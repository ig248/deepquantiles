import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from sklearn.base import BaseEstimator

from .losses import keras_quantile_loss


class MultiQuantileRegressor(BaseEstimator):
    """Predict several quantiles simultaneously using a multi-output network."""

    def __init__(
        self,
        shared_units=(8, 8),
        quantile_units=(8, ),
        activation='relu',
        quantiles=[0.5],
        lr=0.001,
        epochs=10,
        batch_size=100
    ):
        self._model_instance = None
        self.shared_units = shared_units
        self.quantile_units = quantile_units
        self.activation = activation
        self.quantiles = quantiles
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def _model(self):
        input_features = Input((1, ), name='X')
        # append final output
        intermediate = input_features
        for idx, units in enumerate(self.shared_units):
            intermediate = Dense(
                units=units, activation=self.activation, name=f'dense_{idx}'
            )(intermediate)
        outputs = [intermediate for _ in self.quantiles]
        for idx, units in enumerate(self.quantile_units):
            outputs = [
                Dense(units, activation=self.activation, name=f'q_{q}_dense_{idx}')(output)
                for q, output in zip(self.quantiles, outputs)
            ]
        outputs = [
            Dense(1, name=f'q_{q}_out')(output) for q, output in zip(self.quantiles, outputs)
        ]
        model = Model(input_features, outputs, name='Quantile Regressor')

        model.compile(
            optimizer=Adam(lr=self.lr), loss=[keras_quantile_loss(q) for q in self.quantiles]
        )

        return model

    def _init_model(self):
        self._model_instance = self._model()
        return self._model_instance

    @property
    def model(self):
        return self._model_instance or self._init_model()

    def fit(self, X, y, **kwargs):
        self._init_model()
        y = [y for _ in self.quantiles]
        fit_kwargs = dict(
            epochs=self.epochs,
            batch_size=self.batch_size,
        )
        fit_kwargs.update(kwargs)
        self.model.fit(X, y, **fit_kwargs)

    def predict(self, X, **kwargs):
        predict_kwargs = dict(batch_size=self.batch_size, )
        predict_kwargs.update(kwargs)

        return np.hstack(self.model.predict(X, **predict_kwargs)).reshape(X.shape[0], -1)

    def sample(self, X, num_samples=10, **kwargs):
        predict_kwargs = dict(batch_size=self.batch_size, )
        predict_kwargs.update(kwargs)

        quantiles = self.quantiles
        predictions = self.predict(X, **predict_kwargs)
        samples = [np.interp(np.random.rand(num_samples), quantiles, pred) for pred in predictions]
        return np.vstack(samples)

    @classmethod
    def unroll_samples(cls, X, samples):
        num_samples = samples.shape[1]
        X = np.repeat(X, num_samples, axis=1).ravel()
        y = samples.ravel()
        return X, y
