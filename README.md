[![PyPI version](https://badge.fury.io/py/deepquantiles.svg)](https://badge.fury.io/py/deepquantiles)
[![Build Status](https://travis-ci.com/ig248/deepquantiles.svg?branch=master)](https://travis-ci.com/ig248/deepquantiles)
[![Coverage Status](https://codecov.io/gh/ig248/deepquantiles/branch/master/graph/badge.svg)](https://codecov.io/gh/ig248/deepquantiles)

# Deep Continuous Quantile Regression
This package explores different approaches to learning the uncertainty,
and, more generally, the conditional distribution of the target variable.

![](https://raw.githubusercontent.com/ig248/deepquantiles/master/README_pics/comparison_good_MDN_good_CDF.png)

This is particularily importent when

- the mean of the target variable is not sufficient for the use case
- the errors are heteroscedastic, i.e. vary depending on input features
- the errors are skewed, making a single descriptor such as variance inadequate.

![](https://raw.githubusercontent.com/ig248/deepquantiles/master/README_pics/comparison_skewed_samples.png)

We explore two main approches:
1. fitting a mixture density model
2. learning the location of conditional qunatiles, `q`, of the distribution.

Our mixture density network exploits an implementation trick to achieve negative-log-likelihood minimisation in `keras`.

![](https://raw.githubusercontent.com/ig248/deepquantiles/master/README_pics/mdn.png)

Same trick is useed to optimize the "pinball" loss in quantile regression networks.

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
