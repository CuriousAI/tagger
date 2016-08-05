import os
import pickle
import logging
import theano
import numpy as np
from blocks.extensions import SimpleExtension, Printing
from blocks.extensions.monitoring import MonitoringExtension
from blocks.monitoring.evaluators import DatasetEvaluator
from blocks.roles import add_role
import urllib2
import io

logger = logging.getLogger('main.utils')


def shared_param(init, name, role, **kwargs):
    v = np.float32(init)
    p = theano.shared(v, name=name, **kwargs)
    add_role(p, role)
    return p


class AttributeDict(dict):
    __getattr__ = dict.__getitem__

    def __setattr__(self, a, b):
        self.__setitem__(a, b)

    def __setstate__(self, b):
        self.update(b)


class cycle(object):
    def __init__(self, iter_constructor, **kwargs):
        self.kwargs = kwargs
        self.iter_constructor = iter_constructor
        self.iterator = self.gen_iterator()

    def gen_iterator(self):
        return self.iter_constructor(**self.kwargs)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.iterator.next()
        except StopIteration:
            self.iterator = self.gen_iterator()
            return self.iterator.next()

    def next(self):
        return self.__next__()


def print_p(p):
    def qm(val):
        # Add quotation marks around strings
        if isinstance(val, str):
            return "'" + val + "'"
        return val

    logger.info('# PARAMETERS #')
    [logger.info(" p.{:30}= {:<30}".format(k, qm(v))) for k, v in sorted(p.iteritems())]


def generate_base(save_to):
    base = os.path.join('results', save_to.strip('/.'))
    if not base.endswith('run'):
        base = os.path.join(base, 'run')
    return base


def generate_directory(save_to, root='', i=0):
    base = generate_base(save_to)
    while True:
        relative_name = base + str(i)

        try:
            os.makedirs(os.path.join(root, relative_name))
            break
        except OSError as e:
            if e.errno == 17 and i < 10000:  # os.errno==17 already exist, and avoid infinite loop
                i += 1
                continue
            else:
                raise Exception('failed err: {} iter: {}'.format(os.strerror(e.errno), i))

    return relative_name, i


def save_dict(dir, filename, d):
    with open(os.path.join(dir, filename), 'w') as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
    return


def load_dict(f):
    p = pickle.load(f)
    return AttributeDict(p)


def open_url(url_or_file):
    try:
        handle = urllib2.urlopen(url_or_file)
    except ValueError:
        handle = urllib2.urlopen('file:' + url_or_file)
    return io.BytesIO(handle.read())


class SaveExpParams(SimpleExtension):
    def __init__(self, experiment_params, dir, **kwargs):
        super(SaveExpParams, self).__init__(**kwargs)
        self.dir = dir
        self.experiment_params = experiment_params

    def do(self, which_callback, *args):
        save_dict(self.dir, 'params', self.experiment_params)


class FinalTestMonitoring(SimpleExtension, MonitoringExtension):
    """Monitors validation and test set data with batch norm

    Calculates the training set statistics for batch normalization and adds
    them to the model before calculating the validation and test set values.
    This is done in two steps: First the training set is iterated and the
    statistics are saved in shared variables, then the model iterates through
    the test/validation set using the saved shared variables.
    When the training set is iterated, it is done for the full set, layer by
    layer so that the statistics are correct. This is expensive for very deep
    models, in which case some approximation could be in order
    """
    def __init__(self, output_vars, train_data_stream, test_data_streams, **kwargs):
        super(FinalTestMonitoring, self).__init__(**kwargs)
        if not isinstance(test_data_streams, dict):
            self.tst_streams = {self.prefix, test_data_streams}
        else:
            self.tst_streams = test_data_streams

        self._tst_evaluator = DatasetEvaluator(output_vars)

    def do(self, which_callback, *args):
        """Write the values of monitored variables to the log."""
        # Run on train data and get the statistics

        logger.info("Evaluating final test/validation monitor...")
        for prefix, tst_stream in self.tst_streams.iteritems():
            value_dict = self._tst_evaluator.evaluate(tst_stream)
            self.add_records(self.main_loop.log, value_dict.items(), prefix)

    def _record_name(self, name, prefix=None):
        """The record name for a variable name."""
        PREFIX_SEPARATOR = '_'
        prefix = prefix or self.prefix
        return prefix + PREFIX_SEPARATOR + name if prefix else name

    def add_records(self, log, record_tuples, prefix=None):
        """Helper function to add monitoring records to the log."""
        for name, value in record_tuples:
            if not name:
                raise ValueError("monitor variable without name")
            log.current_row[self._record_name(name, prefix)] = value


class SaveParams(SimpleExtension):
    """Finishes the training process when triggered."""
    def __init__(self, save_freq, ladder, save_path, **kwargs):
        super(SaveParams, self).__init__(**kwargs)
        self.save_path = save_path
        self.params = ladder.params.values()
        self.save_freq = save_freq
        self.add_condition(['after_training'])
        self.add_condition(['on_interrupt'])
        self.curr_epoch = 0

    def save(self, name):
        to_save = {v.name: v.get_value() for v in self.params}
        path = os.path.join(self.save_path, name)
        logger.info('Saving to %s' % path)
        np.savez_compressed(path, **to_save)

    def do(self, which_callback, *args):
        if which_callback == 'after_training' or which_callback == 'on_interrupt':
            self.save('trained_params')
        elif which_callback == 'before_epoch':
            if self.save_freq > 0 and self.curr_epoch % self.save_freq == 0:
                self.save('trained_params_%de' % self.curr_epoch)
            self.curr_epoch += 1
        else:
            assert False, "Unknown condition %s" % which_callback


class ShortPrinting(Printing):
    def __init__(self, to_print, use_log=True, **kwargs):
        self.to_print = to_print
        self.use_log = use_log
        super(ShortPrinting, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        log = self.main_loop.log

        # Iteration
        msg = "e {}, i {}:\n\t".format(
            log.status['epochs_done'],
            log.status['iterations_done'])
        level = 'batch' if 'batch' in which_callback else 'epoch'

        # Requested channels
        items = []
        for k, vars in self.to_print.iteritems():
            for shortname, vars in vars.iteritems():
                if vars is None:
                    continue
                if type(vars) is not list:
                    vars = [vars]

                s = ""
                for num, var in enumerate(vars):
                    try:
                        name = k + '_' + var.name
                        val = log.current_row[name]
                    except:
                        continue
                    try:
                        s += ' ' + ' '.join(["%.3g" % v for v in val])
                        s += ' |' if num < len(vars) - 1 else ''
                    except:
                        s += " %.3g" % val
                if s != "":
                    s += "\n\t"
                    items += [shortname + s]
        try:
            items += ['tr %.2f s ' % log.current_row['time_train_this_' + level]]
            items += ['rd %.2f s ' % log.current_row['time_read_data_this_' + level]]
        except:
            pass
        msg = msg + "".join(sorted(items))
        if self.use_log:
            logger.info(msg)
        else:
            print msg