import numpy as np

from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from kerashistoryplot.callbacks import PlotHistory

from . import MultiQuantileRegressor, InverseCDFRegressor, MixtureDensityRegressor

quantiles = [0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999]

callbacks = [
    ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10, min_delta=0.001, min_lr=0.0001),
    EarlyStopping(monitor='loss', patience=15, min_delta=0.0001),
]

nb_callbacks = callbacks + [PlotHistory(batches=False, n_cols=3, figsize=(15, 7))]

fit_kwargs = dict(
    epochs=100,
    batch_size=100,
    shuffle=True,
    verbose=0,
)

independent_quantile_model = MultiQuantileRegressor(
    shared_units=(),
    quantile_units=(32, 32, 32, 32, 32),
    quantiles=quantiles,
    lr=0.01,
    epochs=100,
    batch_size=1000
)

shared_quantile_model = MultiQuantileRegressor(
    shared_units=(32, 32, 32),
    quantile_units=(32, 32),
    quantiles=quantiles,
    lr=0.01,
    epochs=100,
    batch_size=1000
)

inverse_cdf_model = InverseCDFRegressor(
    feature_units=(32, 32),
    quantile_units=(32, 32),
    shared_units=(32, 32, 32),
    batch_norm=True,
    q_mode='point',
    quantiles=np.linspace(0, 1, 50),
    lr=0.01,
    epochs=100,
    batch_size=1000
)

mixture_model = MixtureDensityRegressor(
    shared_units=(32, 32, 32),
    weight_units=(32, 32),
    mu_units=(32, 32),
    sigma_units=(32, 32),
    lr=0.01,
    epochs=100,
    batch_size=1000,
)
