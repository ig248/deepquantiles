import functools

import numpy as np
from keras import backend as K  # noqa
from keras.layers import Lambda

# Quantiles


def _tilted_loss_tensor(q, y_true, y_pred):
    err = (y_pred - y_true)
    return K.maximum(-q * err, (1 - q) * err)


def _tilted_loss_scalar(q, y_true, y_pred):
    return K.mean(_tilted_loss_tensor(q, y_true, y_pred), axis=-1)


def keras_quantile_loss(q):
    """Return keras loss for quantile `q`."""
    func = functools.partial(_tilted_loss_scalar, q)
    func.__name__ = f'qunatile loss, q={q}'
    return func


QuantileLossLayer = Lambda(lambda args: _tilted_loss_tensor(*args))


def sk_quantile_loss_slow(q, y_true, y_pred):
    err = (y_pred.ravel() - y_true.ravel())
    return np.maximum(-q * err, (1 - q) * err).mean()


def sk_quantile_loss(q, y_true, y_pred):
    err = (y_pred.ravel() - y_true.ravel())
    sign = (err > 0)
    return ((sign - q) * err).mean()


# Mixture Density Networks


def keras_mean_pred_loss(y_true, y_pred):
    return K.mean(y_pred)


def _mdn_phi_tensor(y_true, mu, sigma):
    inv_sigma_2 = 1 / K.square(sigma + K.epsilon())
    phi = inv_sigma_2 * K.exp(-inv_sigma_2 * K.square(y_true - mu))
    return phi


def _mdn_loss_tensor(y_true, w, mu, sigma):
    """Gaussian mixture loss for 1D output."""
    phi = _mdn_phi_tensor(y_true, mu, sigma)
    E = -K.log(K.sum(w * phi, axis=1) + K.epsilon())
    return E


MDNLossLayer = Lambda(lambda args: _mdn_loss_tensor(*args))
