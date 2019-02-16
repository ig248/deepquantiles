import numpy as np
from keras.layers import BatchNormalization, Concatenate, Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from sklearn.base import BaseEstimator

from .batches import XYQZBatchGenerator
from .losses import QuantileLossLayer, keras_mean_pred_loss


class InverseCDFRegressor(BaseEstimator):
    """Learn conditional CDF by performing a regression on quantile q."""

    def __init__(
        self,
        quantile_units=(8, ),
        feature_units=(8, ),
        shared_units=(8, 8),
        activation='relu',
        batch_norm=True,
        lr=0.001,
        epochs=10,
        batch_size=100,
        q_mode='const',
        shuffle_points=True,
        quantiles=[0.5],
        ada_num_quantiles=10,
    ):
        self._model_instance = None
        self.feature_units = feature_units
        self.quantile_units = quantile_units
        self.shared_units = shared_units
        self.activation = activation
        self.batch_norm = batch_norm
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        if q_mode not in XYQZBatchGenerator.valid_q_modes:
            raise ValueError(f'q_mode must be one of {XYQZBatchGenerator.valid_q_modes}')
        self.q_mode = q_mode
        self.shuffle_points = shuffle_points
        self.quantiles = quantiles
        self.ada_num_quantiles = ada_num_quantiles

    def _model(self):
        input_features = Input((1, ), name='x')
        input_quantile = Input((1, ), name='q')
        input_label = Input((1, ), name='y')
        scaled_features = input_features
        for idx, units in enumerate(self.feature_units):
            scaled_features = Dense(
                units=units, activation=self.activation, name=f'x_dense_{idx}'
            )(scaled_features)
            if self.batch_norm:
                scaled_features = BatchNormalization()(scaled_features)
        scaled_quantile = input_quantile
        for idx, units in enumerate(self.quantile_units):
            scaled_quantile = Dense(
                units=units, activation=self.activation, name=f'q_dense_{idx}'
            )(scaled_quantile)
            if self.batch_norm:
                scaled_quantile = BatchNormalization()(scaled_quantile)
        intermediate = Concatenate()([scaled_features, scaled_quantile])
        for idx, units in enumerate(self.shared_units):
            intermediate = Dense(
                units=units, activation=self.activation, name=f'dense_{idx}'
            )(intermediate)
            if self.batch_norm:
                intermediate = BatchNormalization()(intermediate)
        prediction_output = Dense(1, name='prediction')(intermediate)

        quantile_model = Model(
            [input_features, input_quantile], prediction_output, name='Quantile model'
        )

        loss_output = QuantileLossLayer([input_quantile, input_label, prediction_output])

        loss_model = Model(
            [input_features, input_label, input_quantile], loss_output, name='Loss model'
        )

        loss_model.compile(optimizer=Adam(lr=self.lr), loss=keras_mean_pred_loss)

        return {'loss': loss_model, 'quantile': quantile_model}

    def _init_model(self):
        self._model_instance = self._model()
        return self._model_instance

    @property
    def model(self):
        return self._model_instance or self._init_model()

    def fit(self, X, y, **kwargs):
        self._init_model()
        fit_kwargs = dict(
            epochs=self.epochs,
            batch_size=self.batch_size,
            q_mode=self.q_mode,
            shuffle_points=self.shuffle_points,
            ada_num_quantiles=self.ada_num_quantiles
        )
        fit_kwargs.update(kwargs)

        batch_size = fit_kwargs.pop('batch_size')
        q_mode = fit_kwargs.pop('q_mode')
        shuffle_points = fit_kwargs.pop('shuffle_points')
        ada_num_quantiles = fit_kwargs.pop('ada_num_quantiles')

        self.gen_ = XYQZBatchGenerator(
            X,
            y,
            batch_size=batch_size,
            q_mode=q_mode,
            shuffle_points=shuffle_points,
            model=self,
            ada_num_quantiles=ada_num_quantiles
        )

        # weird hack to use predict inside batch gen
        # see https://github.com/keras-team/keras/issues/5511
        self.model['quantile'].predict([[0], [0]])

        self.model['loss'].fit_generator(self.gen_, **fit_kwargs)

    def predict(self, X, quantiles=None, **kwargs):
        predict_kwargs = dict(batch_size=self.batch_size, )
        predict_kwargs.update(kwargs)

        if quantiles is None:
            quantiles = self.quantiles
        X_tiled = np.repeat(X, len(quantiles), axis=1).reshape(-1, 1)
        q_tiled = np.tile(quantiles, X.shape[0])
        pred = self.model['quantile'].predict([X_tiled, q_tiled], **predict_kwargs)
        pred = pred.reshape(X.shape[0], len(quantiles))
        return pred

    def sample(self, X, num_samples=10, quantiles=None, **kwargs):
        predict_kwargs = dict(batch_size=self.batch_size, )
        predict_kwargs.update(kwargs)
        if quantiles is None:
            quantiles = self.quantiles
        predictions = self.predict(X, quantiles=quantiles, **predict_kwargs)
        samples = [np.interp(np.random.rand(num_samples), quantiles, pred) for pred in predictions]
        return np.vstack(samples)

    @classmethod
    def unroll_samples(cls, X, samples):
        num_samples = samples.shape[1]
        X = np.repeat(X, num_samples, axis=1).ravel()
        y = samples.ravel()
        return X, y
