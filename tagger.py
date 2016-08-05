import logging
import numpy as np
from collections import OrderedDict
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from blocks.roles import PARAMETER, WEIGHT, BIAS
from utils import AttributeDict
from utils import shared_param
import nn
from nn import flatten_first_two_dims, prepend_empty_dim
from nn import ami_score_op
from data import data_dim
from ladder.ladder import LadderAE

logger = logging.getLogger('main')
floatX = theano.config.floatX


class Tagger(object):
    """Tagger model class.

    Everything related to Tagger models & how it uses external parametric
    mapping, namely Ladder here in the code, is codified here.

    Initialization Takes an AttributeDict instance, and then return an
    initialized model, but if one wants to compile the model, #apply function
    has to be called. Therefore, we have a convenience method called
    #create_tagger that takes care of that.
    """

    @staticmethod
    def create_tagger(p):
        """ Create Tagger instance given parameters p """
        tagger = Tagger(p)
        tagger.apply()

        return tagger

    def __init__(self, p):
        logger.debug(theano.config)
        self.p = p
        self.params = OrderedDict()
        self.rstream = RandomStreams(seed=p.seed)
        self.rng = np.random.RandomState(seed=p.seed)
        self.in_dim = data_dim(p)  # input dimensionality

        input_type = T.type.TensorType('float32', [False] * (len(self.in_dim) + 1))
        input_plus_one_type = T.type.TensorType('float32', [False] * (len(self.in_dim) + 2))
        self.x_only = input_type('features_unlabeled')
        self.x = input_type('features_labeled')
        self.y = theano.tensor.lmatrix('targets_labeled')
        self.masks_unlabeled = input_plus_one_type('masks_unlabeled')

        # We noticed that continuous case becomes more stable if we
        # have stable v, i.e. put Ladder's v through a sigmoid
        if p.input_type == 'continuous':
            decoder_spec = ('gauss_stable_v',)
        else:
            decoder_spec = ('gauss',)

        # Ladder Network, a.k.a Parametric Mapping

        ladder_p = AttributeDict({
            'seed': p.seed,
            'encoder_layers': p.encoder_proj,
            'decoder_spec': decoder_spec,
            'denoising_cost_x': (0.0,) * len(p.encoder_proj),
            'act': 'relu',
            # Ladder doesn't add noise to its layers. Tagger handles all corruption.
            'f_local_noise_std': 0.,
            'super_noise_std': 0.,
            'zestbn': 'no',
            'top_c': True,
            'lr': 0.,
        })
        self.ladder = LadderAE(ladder_p)
        # disable logging from ladder
        logging.getLogger('main.model').setLevel(logging.WARNING)

    def weight(self, init, name):
        weight = self.shared(init, name, role=WEIGHT)
        return weight

    def bias(self, init, name):
        b = self.shared(init, name, role=BIAS)
        return b

    def shared(self, init, name, role=PARAMETER, **kwargs):
        p = self.params.get(name)
        if p is None:
            p = shared_param(init, name, role, **kwargs)
            self.params[name] = p
        else:
            assert p.get_value().shape == np.shape(init)
        return p

    def corrupt(self, x):
        """ Corrupt the input x """
        std = self.p.input_noise

        if self.p.input_type == 'binary':
            rnd_noise = self.rstream.binomial(n=1, p=std, size=x.shape, dtype=floatX)
            noise = rnd_noise * (np.float32(1.) - np.float32(2.) * x)
        else:
            noise = self.rstream.normal(size=x.shape, avg=0.0, std=std)

        return x + T.cast(noise, dtype=floatX)

    def rand_init(self, in_dim, out_dim):
        """ Random initialization for fully connected layers """
        W = self.rng.randn(int(in_dim), int(out_dim)) / np.sqrt(int(in_dim))
        return W

    def apply(self):
        """ Build the whole Tagger computation graph """
        x = prepend_empty_dim(self.x)
        x_only = prepend_empty_dim(self.x_only)
        y = self.y

        logger.info('Building graphs')

        # Clean
        # =====
        # Labeled
        self.clean = self.apply_tagger(x, False, y=y)
        # Unlabeled
        self.clean.update(self.apply_tagger(x_only, False))

        # Corrupted
        # =========
        # Labeled
        self.corr = self.apply_tagger(x, True, y=y)
        # Unlabeled
        self.corr.update(self.apply_tagger(x_only, True))

        # update parameters of tagger with that of ladder
        self.params.update(self.ladder.shareds)

        self.total_cost = np.float32(0)

        # Costs
        # =====

        # Total training cost is denoising cost averaged over iterations
        self.total_cost += self.corr.denoising_cost.mean()

        # Total training cost also has optional classification term from the
        # last iteration
        if self.p.class_cost_x > 0:
            class_cost = self.corr.class_cost[-1] * self.p.class_cost_x
            self.total_cost += class_cost

        # Set names for monitoring variables

        self.corr.denoising_cost.name = 'denois'
        self.clean.class_cost.name = 'cost_class_clean'
        self.clean.class_error.name = 'error_rate_clean'
        self.corr.class_cost.name = 'cost_class_corr'
        self.corr.class_error.name = 'error_rate_corr'

        self.clean.ami_score.name = 'ami_score'
        self.clean.ami_score_per_sample.name = 'ami_score_per_sample'

        self.total_cost.name = 'total_cost'

    def apply_tagger(self, x, apply_noise, y=None):
        """ Build one path of Tagger """
        mb_size = x.shape[1]
        input_shape = (self.p.n_groups, mb_size) + self.in_dim
        in_dim = np.prod(self.in_dim)

        # Add noise
        x_corr = self.corrupt(x) if apply_noise else x
        # Repeat input
        x_corr = T.repeat(x_corr, self.p.n_groups, 0)

        # Compute v
        if self.p.input_type == 'binary':
            v = None
        elif self.p.input_type == 'continuous':
            v = self.weight(1., 'v')
            v = v * T.alloc(1., *input_shape)
            # Cap to positive range
            v = nn.exp_inv_sinh(v)

        d = AttributeDict()

        if y:
            d.pred = []
            d.class_error, d.class_cost = [], []
            # here we have the book-keeping of z and m for the visualizations.
            d.z = []
            d.m = []
        else:
            d.denoising_cost, d.ami_score, d.ami_score_per_sample = [], [], []

        assert self.p.n_iterations >= 1

        # z_hat is the value for the next iteration of tagger.
        # z is the current iteration tagger input
        # m is the current iteration mask input
        # m_hat is the value for the next iteration of tagger.
        # m_lh is the mask likelihood.
        # z_delta is the gradient of z, which depends on x, z and m.
        for step in xrange(self.p.n_iterations):
            # Encoder
            # =======

            # Compute m, z and z_hat_pre_bin
            if step == 0:
                # No values from previous iteration, so let's make them up
                m, z = self.init_m_z(input_shape)
                z_hat_pre_bin = None
                # let's keep in the bookkeeping for the visualizations.
                if y:
                    d.z.append(z)
                    d.m.append(m)
            else:
                # Feed in the previous iteration's estimates
                z = z_hat
                m = m_hat

            # Compute m_lh
            m_lh = self.m_lh(x_corr, z, v)
            z_delta = self.f_z_deriv(x_corr, z, m)

            z_tilde = z_hat_pre_bin if z_hat_pre_bin is not None else z
            # Concatenate all inputs
            inputs = [z_tilde, z_delta, m, m_lh]
            inputs = T.concatenate(inputs, axis=2)

            # Projection, batch-normalization and activation to a hidden layer
            z = self.proj(inputs, in_dim * 4, self.p.encoder_proj[0])

            z -= z.mean((0, 1), keepdims=True)
            z /= T.sqrt(z.var((0, 1), keepdims=True) + np.float32(1e-10))

            z += self.bias(0.0 * np.ones(self.p.encoder_proj[0]), 'b')
            h = self.apply_act(z, 'relu')

            # The first dimension is the group. Let's flatten together with
            # minibatch in order to have parametric mapping compute all groups
            # in parallel
            h, undo_flatten = flatten_first_two_dims(h)

            # Parametric Mapping
            # ==================

            self.ladder.apply(None, self.y, h)
            ladder_encoder_output = undo_flatten(self.ladder.act.corr.unlabeled.h[len(self.p.encoder_proj) - 1])
            ladder_decoder_output = undo_flatten(self.ladder.act.est.z[0])

            # Decoder
            # =======

            # compute z_hat
            z_u = self.proj(ladder_decoder_output, self.p.encoder_proj[0], in_dim, scope='z_u')

            z_u -= z_u.mean((0, 1), keepdims=True)
            z_u /= T.sqrt(z_u.var((0, 1), keepdims=True) + np.float32(1e-10))

            z_hat = self.weight(np.ones(in_dim), 'c1') * z_u + self.bias(np.zeros(in_dim), 'b1')
            z_hat = z_hat.reshape(input_shape)

            # compute m_hat
            m_u = self.proj(ladder_decoder_output, self.p.encoder_proj[0], in_dim, scope='m_u')

            m_u -= m_u.mean((0, 1), keepdims=True)
            m_u /= T.sqrt(m_u.var((0, 1), keepdims=True) + np.float32(1e-10))

            c = self.weight(np.float32(1), 'c2')
            m_hat = nn.softmax_n(m_u * c, axis=0)
            m_hat = m_hat.reshape(input_shape)

            # Apply sigmoid activation if input_type is binary
            if self.p.input_type == 'binary':
                z_hat_pre_bin = z_hat
                z_hat = self.apply_act(z_hat, 'sigmoid')

            # Collapse layer
            # ==============

            # Remove the last dim, which is assumed to be class 'None'
            pred = ladder_encoder_output[:, :, :-1]
            # Normalize
            pred /= T.sum(T.sum(pred, axis=2, keepdims=True), axis=0, keepdims=True)

            # Denoising and Classification costs
            # ==================================

            if y:
                class_cost, class_error = self.compute_classification_cost_and_error(pred, y)
                d.pred.append(pred)
                d.class_cost.append(class_cost)
                d.class_error.append(class_error)

                d.m.append(m_hat)
                d.z.append(z_hat)
            else:
                d.denoising_cost.append(self.denoising_cost(z_hat, m_hat, x, v))

                ami_score, ami_score_per_sample = self.mask_accuracy(self.masks_unlabeled, m_hat)
                d.ami_score.append(ami_score)
                d.ami_score_per_sample.append(ami_score_per_sample)

        # stack the list of tensors into one
        d = AttributeDict({key: T.stacklists(val) for key, val in d.iteritems()})

        return d

    def apply_act(self, input, act_name):
        """ Apply activation act_name to input """
        act = {
            'relu': lambda x: T.maximum(0, x),
            'leakyrelu': lambda x: T.switch(x > 0., x, 0.1 * x),
            'linear': lambda x: x,
            'softsign': lambda x: x / (np.float32(1.) + T.abs_(x)),
            'sigmoid': lambda x: T.nnet.sigmoid(x),
            'tanh': lambda x: T.tanh(x),
            'exp': lambda x: T.exp(x),
            'softplus': lambda x: T.log(1 + T.exp(x)),
            'softmax': lambda x: nn.softmax_n(x),
        }.get(act_name)
        assert act, 'unknown act %s' % act_name

        return act(input)

    def denoising_cost(self, z_hat, m_hat, x, v):
        """ Compute the denoising cost between the reconstruction & clean encoder activations """
        if self.p.input_type == 'binary':
            x_hat = (m_hat * z_hat).sum(axis=0, keepdims=True)
            err = nn.soft_binary_crossentropy(x_hat, x, 1e-4)

        elif self.p.input_type == 'continuous':
            assert z_hat is None or x is None or z_hat.ndim == x.ndim

            # Flatten conv activations into one neuronal dimension
            # After flattening, we have (GROUPS, MB, NEURONS)
            z_hat = z_hat.flatten(3)
            x = x.flatten(3)
            v = v.flatten(3) if v is not None else None
            m_hat = m_hat.flatten(3)

            # v represents variance
            #
            # The generative model is as follows
            # p(x|z) = 1/(sqrt(2 * PI * v) * exp((x - ^z)**2 / (2 * v))
            #
            # expectation over groups <p(x|z)>_s
            #
            # Overall cost:
            # C = - log( <p(x|z)>_s )
            #
            # This is computed in a few steps to make it numerically stable:
            # 1. ps = exp(log(p(x|z))) = exp(log_ps)
            # 2. log_max is for the stability
            # 3. p = <p(x|z)>_s

            sqr_err = (x - z_hat) ** 2
            sqr_err /= v

            log_ps = np.float32(-0.5) * (T.log(v) + sqr_err)
            log_ps += T.log(m_hat + np.float32(1e-5))

            log_max = T.max(log_ps, axis=0, keepdims=True)
            ps = T.exp(log_ps - log_max)

            p = T.sum(ps, axis=0, keepdims=True)
            err = -T.log(p) - log_max

        else:
            raise NotImplementedError('Wrong input_type: %s' % self.p.input_type)

        return err.mean()

    def compute_classification_cost_and_error(self, pred, y):
        """ Compute classification cost and error between pred and y """
        if self.p.class_cost_x > 0:
            assert y.ndim in [1, 2]
            tl = y.flatten(2)
            pred = pred.flatten(3)

            if self.p.n_groups > 1:
                pred = T.sum(pred, axis=0, keepdims=False)
            else:
                pred, _ = flatten_first_two_dims(pred)

            if self.p.objects_per_sample > 1:
                error = nn.sigmoid_mis_classification_rate(pred, tl, self.p.objects_per_sample)
            else:
                error = T.neq(T.argmax(pred, 1), tl.flatten()).mean(dtype='floatX') * np.float32(100.)

            cost = nn.categorical_crossentropy(pred, tl, self.p.objects_per_sample)

        else:
            cost, error = 0, 0

        return cost, error

    def f_z_deriv(self, x, z, m):
        """ Compute gradient of z """
        # Assign the bottom-up mask and derivative term
        if self.p.input_type == 'binary':
            n = np.float32(self.p.input_noise)
            # Add uncertainty to z_hat based on noise level
            noise_scale = (np.float32(1.) - np.float32(2.) * n)
            xi_i = z * noise_scale + n

            xi = T.sum(xi_i * m, axis=0, keepdims=True)
            z_deriv = m / (xi - 1. + x) * noise_scale
        elif self.p.input_type == 'continuous':
            # We do not create a separate xi_i because it would be a
            # common term in the cost function as long as we assume same variance
            # for all groups
            z_deriv = m * (x - z)

        return z_deriv

    def init_m_z(self, shape):
        """ Generate intial mask and z_hat values """
        # Random init for mask
        m_raw = self.rstream.normal(size=shape)
        m = nn.softmax_n(m_raw, axis=0)

        z_hat = T.alloc(self.p.zhat_init_value, *shape)

        return m, z_hat

    def mask_accuracy(self, mask_true, mask_est):
        """ Compute AMI score """
        mask_true = mask_true.flatten(3)
        mask_est = mask_est.flatten(3)

        # (MB, GROUP, NEURONS) -> (1, GROUP, MB, NEURONS)
        mask_true = mask_true.dimshuffle('x', 1, 0, 2)

        assert self.p.n_groups > 1
        ami_score_per_sample = ami_score_op(mask_true[0], mask_est)
        ami_score = T.mean(ami_score_per_sample)

        return ami_score, ami_score_per_sample

    def proj(self, x, in_dim, out_dim, scope=''):
        """ Project x from in_dim to out_dim """
        x, undo_flatten = flatten_first_two_dims(x)
        assert x.ndim in [2, 4]
        x = x.flatten(2) if x.ndim > 2 else x
        w_shape = (in_dim, out_dim)
        name = 'W' if scope == '' else scope + '_W'
        name = 'W' if scope == '' else scope + '_W'
        W = self.weight(self.rand_init(*w_shape), name)
        z = T.dot(x, W)
        z = undo_flatten(z)
        return z

    def m_lh(self, x, z, v):
        """ Compute likelihood term m_lh """
        # Evaluate bottom-up mask
        if self.p.input_type == 'binary':
            # self.p.input_noise since the structure might change
            z_tilde = z * np.float32((1 - 2 * self.p.input_noise)) + np.float32(self.p.input_noise)
            loss = nn.soft_binary_crossentropy(z_tilde, x, 1e-4)
        elif self.p.input_type == 'continuous':
            noise_factor = np.float32(self.p.input_noise ** 2) + v ** 2
            # Represents negative log-p
            loss = np.float32(0.5) * T.log(noise_factor) + T.sqr(z - x) / (np.float32(2) * noise_factor)
        else:
            raise NotImplemented

        # normalize
        loss -= T.min(loss, axis=0, keepdims=True)
        normalizer = T.log(T.sum(T.exp(-loss), axis=0, keepdims=True))
        loss += normalizer

        assert loss.ndim in [3, 5]

        m_lh = T.exp(-loss)

        return m_lh

    def load_params(self, loaded):
        """ Load model parameters """
        # If loading before model is set
        to_load = self.params.values()
        if len(to_load) == 0:
            self.params.update({p.name: p for p in loaded})
            logger.info('Loaded params: %s' % ', '.join(self.params.keys()))
            return

        logger.info('Loading parameters: %s' % ', '.join([tl.name for tl in to_load]) +
                    ' - from parameters: %s' % ', '.join(loaded.keys()))
        for param in to_load:
            if param.name not in loaded.keys():
                logger.warning('Not found param %s' % param.name)
                continue
            assert param.get_value().shape == loaded[param.name].shape, \
                '%s needs shape %s, not %s' % (param.name,
                                               param.get_value().shape,
                                               loaded[param.name].shape)
            param.set_value(loaded[param.name])

    def eval_acts(self, inp):
        """ Evaluate Tagger on given inp data

        Here we are only evaluating the clean path."""

        clean_path_kv = [[k, v] for k, v in self.clean.iteritems()]

        givens = {}
        givens.update({self.x: inp['features_labeled']})
        givens.update({self.y: inp['targets_labeled']})
        givens.update({self.x_only: inp['features_unlabeled']})
        givens.update({self.masks_unlabeled: np.float32(inp['masks_unlabeled'])})

        params, args = zip(*givens.iteritems())
        # on_unused_input is set to ignore to omit some warning prints. It can be changed accordingly base on your use
        # cases.
        function = theano.function(params, [k[1] for k in clean_path_kv], on_unused_input='ignore')

        clean_path_v_outputs = function(*args)

        acts = AttributeDict(
            clean=AttributeDict(
                zip([k[0] for k in clean_path_kv], clean_path_v_outputs)
            )
        )

        return acts
