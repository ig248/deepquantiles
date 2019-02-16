import pytest

import numpy as np
import numpy.testing as npt

from deepquantiles import MultiQuantileRegressor, CDFRegressor, MixtureDensityRegressor

MDR = MixtureDensityRegressor

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
        model = MDR(n_components=self.n_components)
        return model

    @pytest.fixture(scope='class')
    def fitted_model(self, Xy):
        model = MDR(n_components=self.n_components)
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
        loss = unfitted_model.evaluate(X, y, verbose=0)
        assert np.isfinite(loss)

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


w = np.array([[1, 0], [0.5, 0.5], [0, 1]])

mu = np.array([[0, 1], [0, 1], [0, 1]])

sigma = np.array([[1, 1], [2, 2], [0.5, 0.5]])

expected_mean = np.array([[0], [0.5], [1]])

num_examples = w.shape[0]


class TestMixtureComputeMethods:
    """Test convenience methods."""

    def test_mean(self):
        mean = MDR.compute_mean(mu, w, sigma)
        npt.assert_equal(mean, expected_mean)

    def test_sample_indicator(self):
        num_samples = 5
        row_idx, col_idx = MDR.compute_sample_indicator(w, num_samples)
        assert row_idx.shape == (num_examples, num_samples)
        assert col_idx.shape == (num_examples, num_samples)
        assert (row_idx == np.arange(num_examples).reshape(-1, 1)).all()
        assert (col_idx[0, :] == 0).all()
        assert (col_idx[-1, :] == 1).all()

    def test_samples(self):
        num_samples = 5
        samples = MDR.compute_samples(w, mu, sigma, num_samples)
        assert samples.shape == (num_examples, num_samples)
