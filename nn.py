import logging

import numpy as np
import theano
from sklearn.metrics import adjusted_mutual_info_score
from theano import tensor as T
from theano.compile.ops import as_op

logger = logging.getLogger('main.nn')


def prepend_empty_dim(x):
    dim = ['x'] + range(x.ndim)
    return x.dimshuffle(*dim)


def flatten_first_two_dims(x):
    f2 = (x.shape[0], x.shape[1])

    def undo(y):
        return y.reshape(f2 + tuple(y.shape[i] for i in range(1, y.ndim)))
    prod = f2[0] * f2[1]
    yy = x.reshape((prod,) + tuple(x.shape[i] for i in range(2, x.ndim)))
    return yy, undo


def softmax_n(x, axis=-1):
    e_x = T.exp(x - x.max(axis=axis, keepdims=True))
    out = e_x / e_x.sum(axis=axis, keepdims=True)
    return out


def soft_clip(var, epsilon):
    # Clip a [0, 1] variable to [0 + epsilon, 1 - epsilon]
    # Useful for binary values that goes into a log that might blow up
    if epsilon > 0:
        epsilon = np.float32(epsilon)
        return var * (np.float32(1) - np.float32(2) * epsilon) + epsilon
    else:
        return var


def soft_binary_crossentropy(est, label, epsilon):
    epsilon = np.float32(epsilon)
    assert epsilon >= 0
    assert epsilon < 0.5
    if est.dtype not in ['float32']:
        est = T.cast(est, 'float32')
    return T.nnet.binary_crossentropy(soft_clip(est, epsilon), label)


def exp_inv_sinh(x):
    y = T.sqrt(x**2 + 1)
    return T.switch(T.le(x, 0), np.float32(1) / (y - x), x + y)


def infer_shape_ami_score(node, input_shapes):
    ashp, bshp = input_shapes
    return [(ashp[1],)]


@as_op(itypes=[theano.tensor.ftensor3, theano.tensor.ftensor3],
       otypes=[theano.tensor.fvector], infer_shape=infer_shape_ami_score)
def ami_score_op(s, s_hat):
    scores = []
    for i in range(s.shape[1]):
        true_labels = s[:, i, :].argmax(0)
        m = s[:, i, :].max(0) > 0.9
        pred_labels = s_hat[:, i, :].argmax(0)
        scores.append(adjusted_mutual_info_score(true_labels[m], pred_labels[m]))
    return np.array(scores, dtype=np.float32)


def sigmoid_mis_classification_rate(predict, target, objects=1, return_full=False):
    assert target.ndim == 2
    assert predict.ndim == 2

    # Simple calculation if we only have one object
    if objects == 1:
        return T.neq(T.argmax(predict, 1), target.reshape((target.shape[0],))).mean(dtype='floatX') * np.float32(100.)

    top_k_diff = T.argsort(-predict, axis=1)[:, :target.shape[1]]
    res_diff = T.cast(T.eq(T.repeat(top_k_diff, target.shape[1], axis=1),
                           T.tile(target, (1, target.shape[1]))), 'float32')
    res_diff = T.sum(res_diff, axis=1) / T.cast(target.shape[1], 'float32')

    assert objects == 2
    top_k_same = top_k_diff[:, 0]  # Complex top-one top_k
    res_same = T.eq(top_k_same, target[:, 0])

    result = (np.float32(1) -
              T.cast(T.switch(T.eq(target[:, 0], target[:, 1]), res_same, res_diff), 'float32')) * np.float32(100.)

    if return_full:
        return top_k_diff, top_k_same, result, result.mean()
    return result.mean()


def categorical_crossentropy(predict, target, objects=1):
    assert target.ndim == 2
    assert predict.ndim == 2
    if objects > 1:
        target = target.reshape((target.shape[0], objects))  # Only do to catch errors where #objects is wrong
        target_k_hot = [T.extra_ops.to_one_hot(target[:, s], predict.shape[1]) for s in xrange(objects)]
        target_k_hot = sum(target_k_hot) / np.float32(objects)
    else:
        target_k_hot = target.reshape((target.shape[0],))  # Simultaneously checks that it's only one class

    return T.nnet.categorical_crossentropy(predict, target_k_hot).mean()
