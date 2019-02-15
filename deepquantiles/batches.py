import numpy as np
from keras.utils import Sequence


class XYQZBatchGenerator(Sequence):
    """Generate batches from X, y arrays.

    Returns batches of ([X, y, q], z), where
    q are uniform random numbers from [0, 1] and
    z are zeros of the same shape as y.
    """

    valid_q_modes = ['point', 'batch', 'const', 'adaptive']

    @classmethod
    def check_q_mode(cls, q_mode):
        if q_mode not in cls.valid_q_modes:
            raise ValueError(f'q_mode must be one of {cls.valid_q_modes}')

    def __init__(
        self,
        x,
        y,
        batch_size=4,
        q_mode='point',
        shuffle_points=True,
        model=None,
        ada_num_quantiles=10
    ):
        self.x = x
        self.y = y
        if len(self.x.shape) == 1:
            self.x = self.x.reshape((-1, 1))
        if len(self.y.shape) == 1:
            self.y = self.y.reshape((-1, 1))
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError('X and y must be same length.')
        # Init q for q_mode in [`const`, `adaptive`]
        self.q = np.random.rand(*self.y.shape)
        self.batch_size = batch_size
        self.check_q_mode(q_mode)
        self.q_mode = q_mode
        self.shuffle_points = shuffle_points
        self.model = model
        self.ada_num_quantiles = ada_num_quantiles

        self.indices = np.arange(self.x.shape[0])
        self.n_rows = self.x.shape[0]

    def __len__(self):
        return self.n_rows // self.batch_size + (1 if self.n_rows % self.batch_size else 0)

    def __getitem__(self, item):
        item = item % len(self)
        start_row = item * self.batch_size
        end_row = min((item + 1) * self.batch_size, self.n_rows)
        inds = self.indices[start_row:end_row]
        x = self.x[inds, :]
        y = self.y[inds, :]
        z = np.zeros(y.shape)
        if self.q_mode in ['const', 'adaptive']:
            q = self.q[inds, :]
        elif self.q_mode == 'batch':
            q = z + np.random.rand()
        elif self.q_mode == 'point':
            q = np.random.rand(*z.shape)
        else:
            self.check_q_mode()
        return [x, y, q], z

    def on_epoch_end(self):
        if self.shuffle_points:
            np.random.shuffle(self.indices)
        if self.q_mode == 'adaptive':
            ada_quantiles = np.linspace(0, 1, num=self.ada_num_quantiles)
            y_limits = self.model.predict(self.x, quantiles=[0., 1.])
            predictions = self.model.predict(self.x, quantiles=ada_quantiles)
            samples = [
                np.interp(
                    y_min + (y_max - y_min) * np.random.rand(),
                    inv_cdf,
                    ada_quantiles,
                ) for (y_min, y_max), inv_cdf in zip(y_limits, predictions)
            ]
            self.q = np.array(samples).reshape(*self.y.shape)
