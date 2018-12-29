import pytest

import numpy as np

from deepquantiles.batches import XYQZBatchGenerator


def x_y_data(n_points):
    X = np.random.rand(n_points, 2)
    y = np.random.rand(n_points, 1)
    return X, y


class TestXYQZBatchGenerator:
    @pytest.mark.parametrize(
        'n_points, batch_size, expected_nb_batches', [
            (0, 2, 0),
            (1, 2, 1),
            (2, 2, 1),
            (4, 2, 2),
            (5, 2, 3),
        ]
    )
    def test_nb_batches(self, n_points, batch_size, expected_nb_batches):
        x, y = x_y_data(n_points)
        generator = XYQZBatchGenerator(
            x=x, y=y, batch_size=batch_size
        )
        assert len(generator) == expected_nb_batches

    @pytest.mark.parametrize(
        'n_points, batch_size, expected_last_batch_size',
        [(4, 2, 2), (5, 2, 1), (7, 3, 1)]
    )
    def test_batch_size(self, n_points, batch_size, expected_last_batch_size):
        x_data, y_data = x_y_data(n_points)
        generator = XYQZBatchGenerator(
            x=x_data, y=y_data, batch_size=batch_size
        )

        for batch_id in range(len(generator)):
            [x, y, q], z = generator[batch_id]
            assert x.shape[1] == x.shape[1]
            assert y.shape[1] == y.shape[1]
            assert q.shape[1] == 1
            assert z.shape[1] == y.shape[1]
            assert all(z == 0)
            assert all(0 <= q) and all(q <= 1)

        for batch_id in range(len(generator) - 1):
            [x, y, q], z = generator[batch_id]
            assert len(x) == batch_size
            assert len(y) == batch_size
            assert len(q) == batch_size
            assert len(z) == batch_size

        for batch_id in range(len(generator) - 1, len(generator)):
            [x, y, q], z = generator[batch_id]
            assert len(x) == expected_last_batch_size
            assert len(y) == expected_last_batch_size
            assert len(q) == expected_last_batch_size
            assert len(z) == expected_last_batch_size
