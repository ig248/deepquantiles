import pytest

import numpy as np

from deepquantiles import MultiQuantileRegressor, CDFRegressor


@pytest.fixture(scope='module', params=[MultiQuantileRegressor, CDFRegressor])
def Regressor(request):
    return request.param


n = 10
n_features = 1


@pytest.fixture(scope='module')
def Xy():
    X = np.random.rand(n, n_features)
    y = np.random.rand(n, 1)
    return X, y


class TestRegressor:
    @pytest.fixture(scope='class')
    def fitted_model(self, Regressor, Xy):
        model = Regressor()
        X, y = Xy
        model.fit(X, y, epochs=1, verbose=0)
        return model

    def test_fit(self, fitted_model):
        pass

    def test_predict(self, fitted_model, Xy):
        X, y = Xy
        y_hat = fitted_model.predict(X)
        assert y_hat.shape[0] == X.shape[0]
        assert y_hat.shape[1] == len(fitted_model.quantiles)

    def test_sample(self, fitted_model, Xy):
        X, y = Xy
        num_samples = 7
        y_hat = fitted_model.sample(X, num_samples=num_samples)
        assert y_hat.shape == (X.shape[0], num_samples)
