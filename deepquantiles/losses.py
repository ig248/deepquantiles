import functools
from keras import backend as K  # noqa
from keras.layers import Lambda


def _tilted_loss_tensor(q, y_true, y_pred):
    err = (y_pred - y_true)
    return K.maximum(- q * err, (1 - q) * err)


def _tilted_loss_scalar(q, y_true, y_pred):
    return K.mean(_tilted_loss_tensor(q, y_true, y_pred), axis=-1)


def quantile_loss(q):
    """Return keras loss for quantile `q`."""
    func = functools.partial(_tilted_loss_scalar, q)
    func.__name__ = f'qunatile loss, q={q}'
    return func


QuantileLossLayer = Lambda(
    lambda args: _tilted_loss_tensor(*args)
)
