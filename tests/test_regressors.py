import pytest

import numpy as np

from deepquantiles import MultiQuantileRegressor, CDFRegressor, MixtureDensityRegressor


n = 10
n_features = 1


@pytest.fixture(scope='module')
def Xy():
    X = np.random.rand(n, n_features)
    y = np.random.rand(n, 1)
    return X, y


@pytest.fixture(scope='module', params=[MultiQuantileRegressor, CDFRegressor])
def QuantileRegressor(request):
    return request.param


class TestQuantileRegressor:
    @pytest.fixture(scope='class')
    def fitted_model(self, QuantileRegressor, Xy):
        model = QuantileRegressor()
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


class TestMDNRegressor:

    n_components = 3

    @pytest.fixture(scope='class')
    def fitted_model(self, Xy):
        model = MixtureDensityRegressor(n_components=self.n_components)
        X, y = Xy
        model.fit(X, y, epochs=1, verbose=0)
        return model

    def test_fit(self, fitted_model):
        pass

    def test_predict(self, fitted_model, Xy):
        X, y = Xy
        w, mu, sigma = fitted_model.predict(X)
        assert w.shape == (y.shape[0], self.n_components)
        assert mu.shape == (y.shape[0], self.n_components)
        assert sigma.shape == (y.shape[0], self.n_components)

    def test_sample(self, fitted_model, Xy):
        pass
