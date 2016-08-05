import sys
import numpy
import logging
from blocks.algorithms import Adam, GradientDescent
from blocks.extensions import FinishAfter, Timing
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from utils import print_p, generate_directory, load_dict, open_url
from utils import SaveExpParams, FinalTestMonitoring, SaveParams, ShortPrinting
from data import setup_data
from tagger import Tagger

logger = logging.getLogger('main')
logging.basicConfig(level=logging.INFO, format="%(message)s")
sys.setrecursionlimit(50000)


class TaggerExperiment(object):
    """ Holds the state of everything Tagger """

    def __init__(self, p):
        self.save_dir, _ = generate_directory(p.save_to)
        self.p = p

        print_p(self.p)

        self.tagger = Tagger.create_tagger(self.p)

        if 'load_from' in self.p and (self.p.load_from is not None):
            self.load_model(self.p.load_from)

        logger.info('Setting up data...')
        self.streams = setup_data(self.p, use_unlabeled=True, use_labeled=True)

    @classmethod
    def load(cls, load_from, p_override={}):
        def load_p(dir_or_url):
            name = dir_or_url + '/params'
            return load_dict(open_url(name))

        # Start from the defaults.
        p = load_p(load_from)
        p.update(p_override)

        ex = TaggerExperiment(p)

        ex.load_model(load_from)
        return ex

    def train(self):
        """ Setup and train the model """
        to_train = ComputationGraph([self.tagger.total_cost]).parameters
        logger.info('Found the following parameters: %s' % str(to_train))

        step_rule = Adam(learning_rate=self.p.lr)
        training_algorithm = GradientDescent(
            cost=self.tagger.total_cost, parameters=to_train, step_rule=step_rule,
            on_unused_sources='warn',
            theano_func_kwargs={'on_unused_input': 'warn'}
        )

        # TRACKED GRAPH NODES
        train_params = {
            'Train_Denoising_Cost': self.tagger.corr.denoising_cost,
        }
        if self.p.class_cost_x > 0:
            train_params['Train_Classification_Cost'] = self.tagger.corr.class_cost
            train_params['Train_Classification_Error'] = self.tagger.clean.class_error

        valid_params = {
            'Validation_Denoising_Cost': self.tagger.corr.denoising_cost,
        }
        if self.p.class_cost_x > 0:
            valid_params['Validation_Classification_Cost'] = self.tagger.corr.class_cost
            valid_params['Validation_Classification_Error'] = self.tagger.clean.class_error

        test_params = {
            'Test_AMI_Score': self.tagger.clean.ami_score,
            'Test_Denoising_Cost': self.tagger.corr.denoising_cost,
        }
        if self.p.class_cost_x > 0:
            test_params['Test_Classification_Cost'] = self.tagger.corr.class_cost
            test_params['Test_Classification_Error'] = self.tagger.clean.class_error

        short_prints = {
            "train": train_params,
            "valid": valid_params,
            "test": test_params,
        }

        main_loop = MainLoop(
            training_algorithm,
            # Datastream used for training
            self.streams['train'],
            model=Model(self.tagger.total_cost),
            extensions=[
                FinishAfter(after_n_epochs=self.p.num_epochs),
                SaveParams(self.p.get('save_freq', 0), self.tagger, self.save_dir, before_epoch=True),
                DataStreamMonitoring(
                    valid_params.values(),
                    self.streams['valid'],
                    prefix="valid"
                ),
                FinalTestMonitoring(
                    test_params.values(),
                    self.streams['train'],
                    {'valid': self.streams['valid'], 'test': self.streams['test']},
                    after_training=True
                ),
                TrainingDataMonitoring(
                    train_params.values(),
                    prefix="train", after_epoch=True
                ),
                SaveExpParams(self.p, self.save_dir, before_training=True),
                Timing(after_epoch=True),
                ShortPrinting(short_prints, after_epoch=True),
            ])
        logger.info('Running the main loop')
        main_loop.run()

    @staticmethod
    def load_model_params(load_from):
        filename = "trained_params.npz"
        f = open_url(load_from + '/' + filename)
        return numpy.load(f)

    def load_model(self, load_from):
        logger.info('Loading model from {}'.format(load_from))
        loaded = TaggerExperiment.load_model_params(load_from)
        self.tagger.load_params(loaded)
