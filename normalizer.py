"""
Normalizers as preprocessing modules of data.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer


def check_shape(mat, list_names):
    """Check consistency of input ndarray mat and list_names"""
    # mat.shape=[n_samp, len(list_names)], i.e.,
    # input data may be collected from e.g. pd.DataFrame(mat, columns=list_names)
    if mat.shape[1] != len(list_names):
        raise ValueError("Input mat.shape={0}, len(list_names)={1}, not matched!".format(
            mat.shape, len(list_names)))


class ZNormalizer(object):
    """Z normalizer with bound clipper"""
    def __init__(self, clip_lower_quantile=0.01, clip_upper_quantile=0.99):
        self._clip_lower_quantile = clip_lower_quantile
        self._clip_upper_quantile = clip_upper_quantile
        self._df_clip_lb = None
        self._df_clip_ub = None
        self._df_mean = None
        self._df_std = None

    def _fit_bounds(self, mat, list_names):
        check_shape(mat, list_names)
        self._df_clip_lb = pd.DataFrame(
            np.percentile(mat, 100*self._clip_lower_quantile, axis=0, keepdims=True),
            columns=list_names)
        self._df_clip_ub = pd.DataFrame(
            np.percentile(mat, 100*self._clip_upper_quantile, axis=0, keepdims=True),
            columns=list_names)

    def _transform_clip_by_bounds(self, mat, list_names):
        check_shape(mat, list_names)
        if self._df_clip_lb is None or self._df_clip_ub is None:
            raise Exception("Bounds not builded yet")
        if not set(self._df_clip_lb.columns) >= set(list_names):
            raise Exception("The input list_names have not been totally builded in bounds")
        return np.concatenate([
            np.clip(mat[:, i],
                    float(self._df_clip_lb[col]),
                    float(self._df_clip_ub[col])
                   ).reshape([-1, 1])
            for i, col in enumerate(list_names)], 1)

    def _fit_mean_std(self, mat, list_names):
        check_shape(mat, list_names)
        self._df_mean = pd.DataFrame(np.mean(mat, 0, keepdims=True), columns=list_names)
        self._df_std = pd.DataFrame(np.std(mat, 0, keepdims=True), columns=list_names)

    def _transform_mean_std(self, mat, list_names, bool_demean=True, EPSILON=1e-20, BOUND=5.0):
        mat = mat.copy()
        check_shape(mat, list_names)
        if bool_demean:
            if not set(self._df_mean.columns) >= set(list_names):
                raise Exception("The input list_names have not been totally builded in mean")
            mat -= self._df_mean[list_names].values
        if not set(self._df_std.columns) >= set(list_names):
            raise Exception("The input list_names have not been totally builded in std")

        mat /= (EPSILON + self._df_std[list_names].values)

        # avoid memory allocation so a bit ugly....
        BUFFER_SIZE = 10 * 1024 * 1024 # 10 MB buffer size
        chunk_size = BUFFER_SIZE / (mat.shape[1] * mat.itemsize)
        def _chunk_range(len, chunk_size):
            last = 0
            while last < len:
                yield last, min(last + chunk_size, len)
                last += chunk_size

        if BOUND is not None:
            for chunk_start, chunk_end in _chunk_range(len(mat), chunk_size):
                mat_chunk = mat[chunk_start : chunk_end]
                idx_out = np.abs(mat_chunk) > BOUND
                mat_chunk[idx_out] = np.sign(mat_chunk[idx_out]) * BOUND

        return mat

    def fit(self, mat, list_names):
        """Fit the Z-normalizer.

        Args:
            mat: [n_sample, n_features], ndarray
            list_names: n_features-length list, used for store the index of features
        """
        self._fit_bounds(mat, list_names)
        clipped_mat = self._transform_clip_by_bounds(mat, list_names)
        self._fit_mean_std(clipped_mat, list_names)
        return self

    def transform(self, mat, list_names, bool_demean=True):
        """Transform by the Z-normalizer.

        Args:
            mat: [n_sample, n_features], ndarray
            list_names: n_features-length list, used for store the index of features
            bool_demean: bool, whether demean is applied
        Returns:
            ndarray, same size as mat
        """
        clipped_mat = self._transform_clip_by_bounds(mat, list_names)
        return self._transform_mean_std(clipped_mat, list_names, bool_demean=bool_demean)


class QNormalizer(object):
    """Quantile normalizer wrapper"""
    def __init__(self, n_quantiles=20, output_distribution="uniform", **kwargs):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.list_names = None
        self.model_qt = QuantileTransformer(
            n_quantiles=self.n_quantiles,
            output_distribution=self.output_distribution,
            **kwargs)


    def fit(self, mat, list_names, bool_flip=True):
        """Fit the Q-normalizer.

        Args:
            mat: [n_sample, n_features], ndarray
            list_names: n_features-length list, used for store the index of features
            bool_flip: bool, whether mat will be augmented by flipping, default=True to make sure signs are consistent between x and tranform(x)
        """
        check_shape(mat, list_names)
        if self.list_names is not None:
            if self.list_names != list_names:
                raise ValueError("Model already builded, but columns are different")
        self.list_names = list_names
        if bool_flip:
            self.model_qt = self.model_qt.fit(np.concatenate([mat, -mat], 0))
        else:
            self.model_qt = self.model_qt.fit(mat)
        return self

    def transform(self, mat, list_names):
        """Transform by the Q-normalizer.

        Args:
            mat: [n_sample, n_features], ndarray
            list_names: n_features-length list, used for store the index of features
        Returns:
            res: ndarray, same size as mat
            if output_distribution is
            (1) "uniform": res expected to be zero-mean, +1/-1 bounded
            (2) "normal": res expected to be zero-mean, std=1
        """
        check_shape(mat, list_names)
        if self.list_names != list_names:
            raise ValueError("Model already builded, but columns are different")
        res = self.model_qt.transform(mat)
        if self.output_distribution == "uniform":
            # transform it to zero mean, +1/-1 bounded
            res = 2.0 * res - 1.0
        return res


class ZeroNormalizer(object):
    """Set everything to zeros. As the safe choice for no historical data."""
    def __init__(self):
        self._list_names = None

    def fit(self, mat, list_names):
        self._list_names = list(list_names)
        return self

    def transform(self, mat, list_names):
        if self._list_names is not None:
            if not self._list_names == list_names:
                raise ValueError("The input list_names are not consistent with model record")
        return np.zeros_like(mat)
