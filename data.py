import functools
import logging
from picklable_itertools import imap
import numpy
from fuel.datasets import H5PYDataset
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from fuel.transformers import Transformer, Rename, FilterSources
from utils import cycle, AttributeDict

logger = logging.getLogger('main')


class SemiDataStream(Transformer):
    """ Combines two datastreams into one such that 'target' source (labels)
        is used only from the first one. The second one is renamed
        to avoid collision. Upon iteration, the first one is repeated until
        the second one depletes.
        """
    def __init__(self, data_stream_labeled, data_stream_unlabeled, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.ds_labeled = data_stream_labeled
        self.ds_unlabeled = data_stream_unlabeled

    @property
    def sources(self):
        if hasattr(self, '_sources'):
            return self._sources
        return self.ds_labeled.sources + self.ds_unlabeled.sources

    @sources.setter
    def sources(self, value):
        self._sources = value

    def close(self):
        self.ds_labeled.close()
        self.ds_unlabeled.close()

    def reset(self):
        self.ds_labeled.reset()
        self.ds_unlabeled.reset()

    def next_epoch(self):
        self.ds_labeled.next_epoch()
        self.ds_unlabeled.next_epoch()

    def get_epoch_iterator(self, **kwargs):
        labeled = cycle(self.ds_labeled.get_epoch_iterator, **kwargs)
        unlabeled = self.ds_unlabeled.get_epoch_iterator(**kwargs)
        return imap(self.mergedicts, labeled, unlabeled)

    def mergedicts(self, x, y):
        return dict(list(x.items()) + list(y.items()))


def combine_datastreams(ds_labeled, ds_unlabeled):
    # Rename the sources for clarity
    if ds_labeled is not None:
        names = {'features': 'features_labeled',
                 'targets': 'targets_labeled'}
        if 'mask' in ds_labeled.sources:
            names['mask'] = 'masks_labeled'
        ds_labeled = Rename(ds_labeled, names)

    # Rename the source for input pixels and hide its labels!
    if ds_unlabeled is not None:
        sources = list(ds_unlabeled.sources)
        # Mask away the features
        # Remove targets
        del sources[sources.index('targets')]

        names = {'features': 'features_unlabeled'}
        if 'mask' in ds_unlabeled.sources:
            names['mask'] = 'masks_unlabeled'
        ds_unlabeled = Rename(FilterSources(ds_unlabeled, sources), names=names)

    if ds_labeled is None:
        return ds_unlabeled

    if ds_unlabeled is None:
        return ds_labeled

    return SemiDataStream(data_stream_labeled=ds_labeled,
                          data_stream_unlabeled=ds_unlabeled)


class Dataset(object):
    def __init__(self, classes, p, dim, has_valid):
        self.__dict__.update(locals())

        self.trn, self.val, self.tst = self.build_datasets(self.classes, p)

    def get_train_labels(self):
        # Determine amount of classes if has targets
        train_split = ['train']
        train_set = self.classes(which_sets=train_split, sources=['targets'])

        d = train_set
        h = d.open()
        y = numpy.array(d.get_data(h, list(self.trn.ind))[0])
        d.close(h)
        return y

    def build_datasets(self, dataset_class, p):
        train_split = ['train']
        train_set = dataset_class(which_sets=train_split)

        # Take all indices and permutate them
        all_ind = numpy.arange(train_set.num_examples)
        rng = numpy.random.RandomState(seed=p.seed)
        rng.shuffle(all_ind)

        valid_set = dataset_class(which_sets=["valid"])
        valid_ind = numpy.arange(valid_set.num_examples)
        trn_set_size = p.get('train_set_size', None)
        train_ind = all_ind[:trn_set_size]

        test_split = ['test']
        test_set = dataset_class(which_sets=test_split)
        test_ind = numpy.arange(test_set.num_examples)

        trn = AttributeDict(set=train_set, ind=train_ind, batch_size=p.batch_size)
        val = AttributeDict(set=valid_set, ind=valid_ind, batch_size=p.valid_batch_size)
        tst = AttributeDict(set=test_set,  ind=test_ind,  batch_size=p.valid_batch_size)

        return trn, val, tst

    def get_datastream(self, kind, indices):
        split = {
            'trn': self.trn,
            'val': self.val,
            'tst': self.tst,
        }[kind]
        indices = indices if indices is not None else split.ind
        assert len(set(indices) - set(split.ind)) == 0, 'requested indices outside of split'
        ds = DataStream.default_stream(
            split.set, iteration_scheme=ShuffledScheme(indices, split.batch_size))
        return ds

    def get_in_dim(self):
        return self.dim


def data_dim(p):
    """ Return the dimensionality of the dataset """
    dataset_class = DATASETS[p.dataset]
    return dataset_class(p).get_in_dim()


def setup_data(p, use_unlabeled=True, use_labeled=True):
    assert use_unlabeled or use_labeled, 'Cannot train without cost'
    dataset_class = DATASETS[p.dataset]
    dataset = dataset_class(p)
    train_ind = dataset.trn.ind

    if 'labeled_samples' not in p or p.labeled_samples == 0:
        n_labeled = len(train_ind)
    else:
        n_labeled = p.labeled_samples

    if 'unlabeled_samples' not in p:
        n_unlabeled = len(train_ind)
    else:
        n_unlabeled = p.unlabeled_samples

    assert p.batch_size <= n_labeled, "batch size too large"
    assert len(train_ind) >= n_labeled
    assert len(train_ind) >= n_unlabeled, "not enough training samples"
    assert n_labeled <= n_unlabeled, \
        "at least as many unlabeled samples as number of labeled samples"

    # If not using all labels, let's balance classes
    balance_classes = n_labeled < len(train_ind)

    if balance_classes and use_labeled:
        # Ensure each label is equally represented
        y = dataset.get_train_labels()
        n_classes = numpy.max(y) + 1

        n_from_each_class = n_labeled / n_classes
        logger.info('n_sample_from_each_class {0}'.format(n_from_each_class))
        assert n_labeled % n_classes == 0

        i_labeled = []
        for c in xrange(n_classes):
            i = (train_ind[y[:, 0] == c])[:n_from_each_class]
            if len(i) < n_from_each_class:
                logger.warning('Class {0} : only got {1}'.format(c, len(i)))
            i_labeled += list(i)

    else:
        i_labeled = train_ind[:n_labeled]

    def make_unlabeled_set(train_ind, i_labeled, n_unlabeled):
        """ i_unused_labeled: the labels that are not used in i_labeled.
        n_unlabeled_needed: the number of need for i_unlabeled beyond len(i_labeled)
        """
        i_unused_labeled = list(set(train_ind) - set(i_labeled))
        n_unlabeled_needed = n_unlabeled - len(i_labeled)
        i_unlabeled = i_unused_labeled[:n_unlabeled_needed]
        i_unlabeled.extend(i_labeled)

        return i_unlabeled

    i_unlabeled = make_unlabeled_set(train_ind, i_labeled, n_unlabeled)

    logger.info('Creating data set with %d labeled and %d total samples' %
                (len(i_labeled), len(i_unlabeled)))

    streams = AttributeDict()

    def make(kind, ind_labeled, ind_unlabeled):
        ds_labeled, ds_unlabeled = None, None
        if use_labeled:
            ds_labeled = dataset.get_datastream(kind, ind_labeled)
        if use_unlabeled:
            ds_unlabeled = dataset.get_datastream(kind, ind_unlabeled)

        return combine_datastreams(ds_labeled, ds_unlabeled)

    streams.train = make('trn', i_labeled, i_unlabeled)
    streams.valid = make('val', None, None)  # use all indices
    streams.test = make('tst', None, None)  # use all indices

    return streams

part = functools.partial
DATASETS = {
    'shapes50k20x20':       part(Dataset, part(H5PYDataset, file_or_path='data/shapes50k_20x20_compressed.h5', load_in_memory=True), has_valid=True, dim=(400,)),
    'freq20-2mnist':        part(Dataset, part(H5PYDataset, file_or_path='data/freq20-2MNIST_compressed.h5', load_in_memory=True), has_valid=True, dim=(1, 28, 28)),
    'freq20-1mnist':        part(Dataset, part(H5PYDataset, file_or_path='data/freq20-1MNIST_compressed.h5', load_in_memory=True), has_valid=True, dim=(1, 28, 28)),
}
