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
        assert np.isfinite(y_hat).all()
        assert y_hat.shape[0] == X.shape[0]
        assert y_hat.shape[1] == len(fitted_model.quantiles)

    def test_sample(self, fitted_model, Xy):
        X, y = Xy
        num_samples = 7
        y_samples = fitted_model.sample(X, num_samples=num_samples)
        assert np.isfinite(y_samples).all()
        assert y_samples.shape == (X.shape[0], num_samples)


class TestMDNRegressor:

    n_components = 3

    @pytest.fixture(scope='class')
    def unfitted_model(self, Xy):
        model = MixtureDensityRegressor(n_components=self.n_components)
        return model

    @pytest.fixture(scope='class')
    def fitted_model(self, Xy):
        model = MixtureDensityRegressor(n_components=self.n_components)
        X, y = Xy
        model.fit(X, y, epochs=1, verbose=0)
        return model

    def test_predict(self, unfitted_model, Xy):
        X, y = Xy
        w, mu, sigma = unfitted_model.predict(X)
        assert np.isfinite(w).all()
        assert np.isfinite(mu).all()
        assert np.isfinite(sigma).all()
        assert w.shape == (y.shape[0], self.n_components)
        assert mu.shape == (y.shape[0], self.n_components)
        assert sigma.shape == (y.shape[0], self.n_components)
        assert np.allclose(w.sum(axis=1), 1)

    def test_evaluate_loss(self, unfitted_model, Xy):
        X, y = Xy
        losses = unfitted_model.model['loss'].evaluate([X, y], y, verbose=0)
        assert np.isfinite(losses).all()

    def test_fit(self, fitted_model):
        pass

    def test_fit_predict(self, fitted_model, Xy):
        X, y = Xy
        w, mu, sigma = fitted_model.predict(X)
        assert np.isfinite(w).all()
        assert np.isfinite(mu).all()
        assert np.isfinite(sigma).all()
        assert w.shape == (y.shape[0], self.n_components)
        assert mu.shape == (y.shape[0], self.n_components)
        assert sigma.shape == (y.shape[0], self.n_components)
        assert np.allclose(w.sum(axis=1), 1)

    def test_sample(self, fitted_model, Xy):
        pass
