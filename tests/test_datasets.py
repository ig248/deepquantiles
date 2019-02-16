import pytest

from deepquantiles.datasets import bishop_s, xmas_tree, skewed_heteroscedastic


@pytest.fixture(scope='module', params=[bishop_s, xmas_tree, skewed_heteroscedastic])
def data_gen(request):
    return request.param


def test_data_shape(data_gen):
    n_points = 100
    x, y = data_gen(num=n_points)
    assert x.shape == (n_points, 1)
    assert y.shape == (n_points, )


def test_data_sorted(data_gen):
    n_points = 100
    x, y = data_gen(num=n_points, sort=True)
    assert x[0] == x.min()
    assert x[-1] == x.max()
