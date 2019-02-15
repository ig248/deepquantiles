import numpy as np
import keras.backend as K  # noqa

from deepquantiles.losses import _mdn_phi_tensor, _mdn_loss_tensor

n = 10  # examples
m = 3  # mixture components


def test_mdn_loss_1D():
    y_val = np.random.random((n, m))
    w_val = np.random.random((n, m))
    w_val = np.exp(w_val)
    w_val = w_val / np.sum(w_val, axis=1, keepdims=True)
    mu_val = np.random.random((n, m))
    sigma_val = np.abs(np.random.random((n, m)))

    y = K.variable(value=y_val)
    w = K.variable(value=w_val)
    mu = K.variable(value=mu_val)
    sigma = K.variable(value=sigma_val)

    phi_tensor = _mdn_phi_tensor(y, mu, sigma)
    assert phi_tensor.shape == (n, m)

    loss_tensor = _mdn_loss_tensor(y, w, mu, sigma)
    assert loss_tensor.shape == (n, )
