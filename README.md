[![PyPI version](https://badge.fury.io/py/deepquantiles.svg)](https://badge.fury.io/py/deepquantiles)
[![Build Status](https://travis-ci.com/ig248/deepquantiles.svg?branch=master)](https://travis-ci.com/ig248/deepquantiles)
[![Coverage Status](https://codecov.io/gh/ig248/deepquantiles/branch/master/graph/badge.svg)](https://codecov.io/gh/ig248/deepquantiles)

# Deep Continuous Quantile Regression
This package explores different approaches to learning the uncertainty,
and, more generally, the conditional distribution of the target variable. We introduce a new type of network, the "Deep Continuous Quantile Regression Network", that approximates the inverse conditional CDF directly by a mult-layer perceptron, instead of relying on variational methods which require priors on the functional form of the distribution. In many cases we find that it presents a robust alternative to well-known Mixture Density Networks`.

![](https://raw.githubusercontent.com/ig248/deepquantiles/master/README_pics/comparison_good_MDN_good_CDF.png)

This is particularily important when

- the mean of the target variable is not sufficient for the use case
- the errors are heteroscedastic, i.e. vary depending on input features
- the errors are skewed, making a single summary statistic such as variance inadequate.

![](https://raw.githubusercontent.com/ig248/deepquantiles/master/README_pics/comparison_skewed_samples.png)

We explore two main approches:
1. fitting a mixture density model
2. learning the location of conditional qunatiles, `q`, of the distribution.

Our mixture density network exploits an implementation trick to achieve negative-log-likelihood minimisation in `keras`.

![](https://raw.githubusercontent.com/ig248/deepquantiles/master/README_pics/mdn.png)

Same trick is useed to optimize the "pinball" loss in quantile regression networks, and in fact can be used to optimize an arbitrary loss function of `(X, y, y_hat)`.

Within the quantile-based approach, we further explore:
a. fitting a separate model to predict each quantile
b. fitting a multi-output network to predict multiple quantiles simultaneously
c. learning a regression on `X` and `q` simultanesously, thus effectively
learning the complete (conditional) cumulative density function.


## Installation
Install package from source:

```
pip install git+https://github.com/ig248/deepquantiles
```

Or from PyPi:

```
pip install deepquantiles
```
## Usage
```
from deepquantiles import MultiQuantileRegressor, InverseCDFRegressor, MixtureDensityRegressor
```
As this package is largely an experiment, please explore the Jupyter notebooks and expect to look at the source code.

## Content
- `deepqunatiles.regressors`: implementation of core algorithms
- `deepquantiles.presets`: a collection of pre-configured estimators and settings used in experiments
- `deepquantiles.datasets`: functions used for generating test data
- `deepquantiles.nb_utils`: helper functions used in notebooks
- `notebooks`: Jupyter notebooks with examples and experiments

## Tests

Run
```bash
make dev-install
make lint
make test
```

## References
**Mixture Density Networks**, Christopher M. Bishop, [NCRG/94/004 (1994)](https://publications.aston.ac.uk/373/1/NCRG_94_004.pdf)