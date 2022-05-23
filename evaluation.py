# -*- coding: utf-8 -*-
"""
Evaluation functions for training, testing, evaluation.
"""
import os
from collections import OrderedDict
import pandas as pd
import numpy as np
import cupy as cp
import torch
import logging
from typing import Any, Callable, ClassVar, List, Set, Dict, Union, Optional, Tuple

from stml_mft_china_eq.pyutils.misc_utils import get_gpu_device
from stml_mft_china_eq.statarb.utils import scipy_stats_skew, scipy_stats_kurtosis
import stml_mft_china_eq.pyutils.datetime_utils as dtu

EPSILON = 1e-10

LIST_SELF_METRICS = ["mean", "median", "std", "skew", "kurt", "min", "max", "mean/std", "p(neg)", "p(zero)", "p(pos)"]
LIST_COMPARE_METRICS_REGRESSION = ["MSE", "RelMSE", "R2", "CorrR", "IC1", "IC2", "SpearmanRho", "KendallTau", "CC", "slope", "Q(RET)"]
LIST_COMPARE_METRICS_CLASSIFICATION = ["sign_confmat"]

def torch2numpy(x_in):
    if isinstance(x_in, torch.Tensor):
        return x_in.detach().to("cpu").numpy()
    return x_in

###############################################################
#numpy evaluation functions
###############################################################
def eval_stats(
    yhat: Union[np.ndarray, torch.Tensor],
    key_prefix: str = "",
    list_metrics: Optional[List] = None,
    is_res_dataframe: bool = True,
    use_cupy: bool = False
) -> Union[pd.DataFrame, OrderedDict]:
    """Statistics of one signal (np.ndarray/torch.Tensor) itself, compared to NO target y.

    Args:
        yhat: np.ndarray/torch.Tensor of yhat to get flattened and evaluted
        key_prefix: str, the prefix to be added to the key of the result dict
        list_metrics: the list of metric names to get evalutated. Default=None, leads to LIST_SELF_METRICS
        is_res_dataframe: whether the result will be pd.DataFrame (if True) or OrderedDict (if False)
        use_cupy: whether use cupy for potential speedup
    Returns:
        pd.DataFrame or OrderedDict
    """
    if not isinstance(yhat, (torch.Tensor, np.ndarray)):
        raise Exception("Input yhat does not belong to (torch.Tensor, np.ndarray)")
    if list_metrics is None:
        list_metrics = LIST_SELF_METRICS
    else:
        if not (set(list_metrics) <= set(LIST_SELF_METRICS)):
            raise ValueError("There's some metric input but not supported")

    yhat = torch2numpy(yhat)
    yhat = yhat.reshape([-1]).astype(float)
    if is_res_dataframe:
        res = pd.DataFrame(
            np.nan * np.ones((1, len(list_metrics))),
            columns=[key_prefix+metric for metric in list_metrics]
        )
    else:
        res = OrderedDict([])

    if len(yhat) == 0:
        for metric in list_metrics:
            res[key_prefix+metric] = np.nan
        return res

    if not use_cupy:
        for metric in list_metrics:
            if metric == "mean":
                res[key_prefix+metric] = np.nanmean(yhat)
            elif metric == "median":
                res[key_prefix+metric] = np.nanmedian(yhat)
            elif metric == "std":
                res[key_prefix+metric] = np.nanstd(yhat)
            elif metric == "skew":
                res[key_prefix+metric] = scipy_stats_skew(yhat)
            elif metric == "kurt":
                res[key_prefix+metric] = scipy_stats_kurtosis(yhat)
            elif metric == "min":
                res[key_prefix+metric] = np.nanmin(yhat)
            elif metric == "max":
                res[key_prefix+metric] = np.nanmax(yhat)
            elif metric == "mean/std":
                res[key_prefix+metric] = np.nanmean(yhat) / (EPSILON + np.nanstd(yhat))
            elif metric == "p(neg)":
                res[key_prefix+metric] = np.mean(np.sign(yhat)==-1)
            elif metric == "p(zero)":
                res[key_prefix+metric] = np.mean(np.sign(yhat)==0)
            elif metric == "p(pos)":
                res[key_prefix+metric] = np.mean(np.sign(yhat)==1)
            else:
                raise NotImplementedError()
    else:
        def _cupy_skew(x_in):
            _diff = x_in - cp.nanmean(x_in)
            _mu3 = float(cp.mean(_diff**3))
            _mu2 = float(cp.mean(_diff**2))
            return np.sqrt(_mu3**2 / (_mu2**3 + EPSILON))
        def _cupy_kurt(x_in):
            _diff = x_in - cp.nanmean(x_in)
            _mu4 = float(cp.mean(_diff**4))
            _mu2 = float(cp.mean(_diff**2))
            return _mu4 / (_mu2**2 + EPSILON) - 3.0
        device = get_gpu_device()
        with cp.cuda.Device(device):
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=15 * 1024**3)
            yhat = cp.asarray(yhat)
            for metric in list_metrics:
                if metric == "mean":
                    res[key_prefix+metric] = float(cp.nanmean(yhat))
                elif metric == "median":
                    res[key_prefix+metric] = float(cp.nanmedian(yhat))
                elif metric == "std":
                    res[key_prefix+metric] = float(cp.nanstd(yhat))
                elif metric == "skew":
                    res[key_prefix+metric] = _cupy_skew(yhat)
                elif metric == "kurt":
                    res[key_prefix+metric] = _cupy_kurt(yhat)
                elif metric == "min":
                    res[key_prefix+metric] = float(cp.nanmin(yhat))
                elif metric == "max":
                    res[key_prefix+metric] = float(cp.nanmax(yhat))
                elif metric == "mean/std":
                    res[key_prefix+metric] = float(cp.nanmean(yhat)) / (EPSILON + float(cp.nanstd(yhat)))
                elif metric == "p(neg)":
                    res[key_prefix+metric] = float(cp.mean(cp.sign(yhat)==-1))
                elif metric == "p(zero)":
                    res[key_prefix+metric] = float(cp.mean(cp.sign(yhat)==0))
                elif metric == "p(pos)":
                    res[key_prefix+metric] = float(cp.mean(cp.sign(yhat)==1))
                else:
                    raise NotImplementedError()
    return res


def eval_stats_mat(
    yhat_mat: Union[torch.Tensor, np.ndarray],
    key_prefix: str = "",
    list_metrics: Optional[List] = None,
    is_res_dataframe: bool = True,
    use_cupy: bool = False
) -> Union[pd.DataFrame, OrderedDict]:
    """Call eval_stats column-by-column and then take average of each stat."""
    if not isinstance(yhat_mat, (torch.Tensor, np.ndarray)):
        raise Exception("Input yhat_mat does not belong to (torch.Tensor, np.ndarray)")

    yhat_mat = torch2numpy(yhat_mat)
    if yhat_mat.ndim != 2:
        raise ValueError(f"The yhat_mat should be 2dim, but it is input as {yhat_mat.ndim}.")
    if is_res_dataframe:
        res = []
        for i in range(yhat_mat.shape[1]):
            acc_res = eval_stats(yhat_mat[:,i], key_prefix=key_prefix, list_metrics=list_metrics, use_cupy=use_cupy)
            res.append(acc_res)
        return pd.concat(res, axis=0).mean().to_frame().T
    else:
        res = OrderedDict()
        for i in range(yhat_mat.shape[1]):
            acc_res = eval_stats(yhat_mat[:,i], key_prefix=key_prefix, list_metrics=list_metrics)
            for _k,_v in acc_res.items():
                res[_k] = res.get(_k, []) + [_v]
        return OrderedDict([
            (_k, np.nanmean(_v)) for _k,_v in res.items()
        ])


def sign_confmat(y: Union[torch.Tensor, np.ndarray], yhat: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    "c_ij = c(y=i, yhat=j) = c(i->j)"
    from sklearn.metrics import confusion_matrix
    y = torch2numpy(y)
    yhat = torch2numpy(yhat)
    aug_y = np.concatenate([np.reshape(np.sign(y), [-1]),  np.array([-1, 0, 1])])
    aug_yhat = np.concatenate([np.reshape(np.sign(yhat), [-1]),  np.array([-1, 0, 1])])
    confmat = confusion_matrix(aug_y, aug_yhat) - np.eye(3)
    confmat = confmat / np.sum(confmat)
    return confmat


def eval_classification(
    y: Union[torch.Tensor, np.ndarray],
    yhat: Union[torch.Tensor, np.ndarray],
    key_prefix: str = "",
    list_metrics: Optional[List] = None,
) -> OrderedDict:
    """Statistics of groud truth v.s. one signal (np.ndarray).

    Args:
        y, yhat: same size 1d np.ndarray
            NOTE: the order of y and yhat matters!!!
        key_prefix: str, the prefix in the resulted keys
        list_metrics: the list of metrics to evaluate
            Default will lead to ["sign_confmat"]
    Returns:
        OrderedDict
    """
    if not isinstance(y, (torch.Tensor, np.ndarray)) or not isinstance(yhat, (torch.Tensor, np.ndarray)):
        raise Exception("Input are not both np.ndarray")
    y = torch2numpy(y)
    yhat = torch2numpy(yhat)
    if list_metrics is None:
        list_metrics = LIST_COMPARE_METRICS_CLASSIFICATION
    else:
        if not (set(list_metrics) <= set(LIST_COMPARE_METRICS_CLASSIFICATION)):
            raise ValueError("There's some metric input but not supported")

    y_shape = y.shape
    yhat_shape = yhat.shape
    y = y.astype("float64").reshape([-1])
    yhat = yhat.astype("float64").reshape([-1])
    if len(y) != len(yhat):
        raise Exception("The shapes of y={0} and yhat={1} do not match".format(
            y_shape, yhat_shape
        ))

    res_dict = OrderedDict([])
    if len(y) == 0:
        for metric in list_metrics:
            if metric == "sign_confmat":
                res_dict[key_prefix+metric] = np.nan*np.ones((3,3))
            else:
                res_dict[key_prefix+metric] = np.nan
        return res_dict

    for metric in list_metrics:
        if metric == "sign_confmat":
            res_dict[key_prefix+metric] = sign_confmat(y, yhat)
        else:
            raise NotImplementedError()
    return res_dict


def eval_classification_mat(
    y_mat: Union[torch.Tensor, np.ndarray],
    yhat_mat: Union[torch.Tensor, np.ndarray],
    key_prefix: str = "",
    list_metrics: Optional[List] = None,
    is_res_dataframe: bool = True,
) -> Union[pd.DataFrame, OrderedDict]:
    """Call eval_classification column-by-column and then take average of each stat.

    Args:
        y_mat: (n,) or (n,v) shape array
        yhat_mat: (n,) or (n,v) shape array
        key_prefix: str, the prefix in the resulted keys
        list_metrics: the list of metrics to evaluate
            Default will lead to ["sign_confmat"]
    NOTE:
        If y_mat and yhat_mat are both 2d, they should have the same size and be calculated correspondingly.
    Returns:
        dict: each value is the mean of v values
    """
    y_mat = torch2numpy(y_mat)
    yhat_mat = torch2numpy(yhat_mat)
    if y_mat.ndim == 1 and yhat_mat.ndim == 2:
        y_mat = np.tile(y_mat.reshape([-1, 1]), [1, yhat_mat.shape[1]])
    elif yhat_mat.ndim == 1 and y_mat.ndim == 2:
        yhat_mat = np.tile(yhat_mat.reshape([-1, 1]), [1, y_mat.shape[1]])
    elif y_mat.ndim == yhat_mat.ndim == 2:
        assert y_mat.shape == yhat_mat.shape
    else:
        raise ValueError("y_mat.shape={0}, yhat_mat.shape={1}, unsupported".format(
            y_mat.shape, yhat_mat.shape
        ))

    res = OrderedDict()
    for i in range(yhat_mat.shape[1]):
        acc_res = eval_classification(y_mat[:,i], yhat_mat[:,i], key_prefix=key_prefix, list_metrics=list_metrics)
        for _k,_v in acc_res.items():
            if _k != "sign_confmat":
                res[_k] = res.get(_k, []) + [_v]
            else:
                res[_k] = res.get(_k, []) + [np.expand_dims(_v, 2)]
    return OrderedDict([
        (_k, np.nanmean(_v))
        if _k!="sign_confmat" else (_k, np.nanmean(np.concatenate(_v, 2), 2))
        for _k,_v in res.items()
    ])


def _eval_regression(
    y: Union[torch.Tensor, np.ndarray],
    yhat: Union[torch.Tensor, np.ndarray],
    key_prefix: str = "",
    list_metrics: Optional[List] = None,
    is_res_dataframe: bool = True,
) -> Union[pd.DataFrame, OrderedDict]:
    """Statistics of groud truth v.s. one signal (np.ndarray).

    Args:
        y, yhat: same size 1d np.ndarray
            NOTE: the order of y and yhat matters!!!
        key_prefix: str, the prefix in the resulted keys
        list_metrics: the list of metrics to evaluate
            Default will lead to ["MSE", "RelMSE", "R2", "CorrR", "IC1", "IC2", "SpearmanRho", "KendallTau", "CC", "slope", "Q(RET)"]
        is_res_dataframe: whether the result will be pd.DataFrame (if True) or OrderedDict (if False)
    Returns:
        pd.DataFrame or OrderedDict
    """

    from scipy.stats import pearsonr, spearmanr, kendalltau
    if not isinstance(y, (torch.Tensor, np.ndarray)) or not isinstance(yhat, (torch.Tensor, np.ndarray)):
        raise Exception("Input are not both np.ndarray")
    y = torch2numpy(y)
    yhat = torch2numpy(yhat)
    if list_metrics is None:
        list_metrics = LIST_COMPARE_METRICS_REGRESSION
    else:
        if not (set(list_metrics) <= set(LIST_COMPARE_METRICS_REGRESSION)):
            raise ValueError("There's some metric input but not supported")

    y_shape = y.shape
    yhat_shape = yhat.shape
    if len(y) != len(yhat):
        raise Exception("The shapes of y={0} and yhat={1} do not match".format(
            y_shape, yhat_shape
        ))
    y = y.astype(np.float64).reshape([-1])
    yhat = yhat.astype(np.float64).reshape([-1])

    if is_res_dataframe:
        res = pd.DataFrame(
            np.nan * np.ones((1, len(list_metrics))),
            columns=[key_prefix+metric for metric in list_metrics]
        )
    else:
        res = OrderedDict([])

    if len(y) < 3:
        for metric in list_metrics:
            res[key_prefix+metric] = np.nan
        return res

    E_e2 = np.nanmean((y - yhat)**2)
    E_y2 = np.nanmean(y**2)
    E_yhat2 = np.nanmean(yhat**2)
    E_yyhat = np.nanmean(y * yhat)

    for metric in list_metrics:
        if metric == "MSE":
            res[key_prefix+metric] = E_e2
        elif metric == "RelMSE":
            res[key_prefix+metric] = E_e2 / (EPSILON + E_y2)
        elif metric == "R2":
            res[key_prefix+metric] = 1.0 - E_e2 / (EPSILON + E_y2)
        elif metric == "CorrR":
            res[key_prefix+metric] = pearsonr(y, yhat)[0]
        elif metric == "IC1":
            res[key_prefix+metric] = 2.0 * (np.sign(y) == np.sign(yhat)).mean() - 1.0
        elif metric == "IC2":
            res[key_prefix+metric] = 2.0 * (np.sign(y)*np.sign(yhat) > 0.0).mean() - 1.0
        elif metric == "SpearmanRho":
            res[key_prefix+metric] = spearmanr(y, yhat)[0]
        elif metric == "KendallTau":
            res[key_prefix+metric] = kendalltau(y, yhat)[0]
        elif metric == "CC":
            res[key_prefix+metric] = E_yyhat / (EPSILON + np.sqrt(E_y2) * np.sqrt(E_yhat2))
        elif metric == "slope":
            res[key_prefix+metric] = E_yyhat / (EPSILON + E_yhat2)
        elif metric == "Q(RET)":
            res[key_prefix+metric] = E_yyhat / (EPSILON + np.sqrt(E_yhat2))
        else:
            raise NotImplementedError()
    return res


def eval_regression(
    y: Union[torch.Tensor, np.ndarray],
    yhat: Union[torch.Tensor, np.ndarray],
    key_prefix: str = "",
    list_metrics: Optional[List] = None,
    is_res_dataframe: bool = True,
    augment_yhat_sign: bool = False,
) -> Union[pd.DataFrame, OrderedDict]:
    """Statistics of groud truth v.s. one signal (np.ndarray).

    Args:
        y, yhat: same size 1d np.ndarray
            NOTE: the order of y and yhat matters!!!
        key_prefix: str, the prefix in the resulted keys
        list_metrics: the list of metrics to evaluate
            Default will lead to ["MSE", "RelMSE", "R2", "CorrR", "IC1", "IC2", "SpearmanRho", "KendallTau", "CC", "slope", "Q(RET)"]
        is_res_dataframe: whether the result will be pd.DataFrame (if True) or OrderedDict (if False)
        augment_yhat_sign: whether results will be augmented conditioned on different yhat's signs
    Returns:
        pd.DataFrame or OrderedDict
    """
    res = _eval_regression(
        y=y, yhat=yhat, key_prefix=key_prefix, list_metrics=list_metrics, is_res_dataframe=is_res_dataframe)
    if augment_yhat_sign:
        idx_pos = yhat > 0
        res_pos = _eval_regression(
            y=y[idx_pos], yhat=yhat[idx_pos], key_prefix=key_prefix+"L#", list_metrics=list_metrics, is_res_dataframe=is_res_dataframe)
        idx_neg = yhat < 0
        res_neg = _eval_regression(
            y=y[idx_neg], yhat=yhat[idx_neg], key_prefix=key_prefix+"S#", list_metrics=list_metrics, is_res_dataframe=is_res_dataframe)

        if is_res_dataframe:
            res = pd.concat([res, res_pos, res_neg], axis=1)
        else:
            for k,v in res_pos.items():
                res[k] = v
            for k,v in res_neg.items():
                res[k] = v
    return res


def eval_regression_mat(
    y_mat: Union[torch.Tensor, np.ndarray],
    yhat_mat: Union[torch.Tensor, np.ndarray],
    key_prefix: str = "",
    list_metrics: Optional[List] = None,
    is_res_dataframe: bool = True,
    augment_yhat_sign: bool = False
) -> Union[pd.DataFrame, OrderedDict]:
    """Call eval_regression column-by-column and then take average of each stat.

    Args:
        y_mat: (n,) or (n,v) shape array
        yhat_mat: (n,) or (n,v) shape array
        key_prefix: str, the prefix in the resulted keys
        list_metrics: the list of metrics to evaluate
            Default will lead to ["MSE", "RelMSE", "R2", "CorrR", "IC1", "IC2", "SpearmanRho", "KendallTau", "CC", "slope", "Q", "sign_confmat"]
        augment_yhat_sign: whether results will be augmented conditioned on different yhat's signs
    NOTE:
        If y_mat and yhat_mat are both 2d, they should have the same size and be calculated correspondingly.
    Returns:
        dict: each value is the mean of v values
    """
    y_mat = torch2numpy(y_mat)
    yhat_mat = torch2numpy(yhat_mat)
    if y_mat.ndim == 1 and yhat_mat.ndim == 2:
        y_mat = np.tile(y_mat.reshape([-1, 1]), [1, yhat_mat.shape[1]])
    elif yhat_mat.ndim == 1 and y_mat.ndim == 2:
        yhat_mat = np.tile(yhat_mat.reshape([-1, 1]), [1, y_mat.shape[1]])
    elif y_mat.ndim == yhat_mat.ndim == 2:
        assert y_mat.shape == yhat_mat.shape
    else:
        raise ValueError("y_mat.shape={0}, yhat_mat.shape={1}, unsupported".format(
            y_mat.shape, yhat_mat.shape
        ))

    if is_res_dataframe:
        res = []
        for i in range(yhat_mat.shape[1]):
            acc_res = eval_regression(y_mat[:,i], yhat_mat[:,i], key_prefix=key_prefix, list_metrics=list_metrics, augment_yhat_sign=augment_yhat_sign)
            res.append(acc_res)
        return pd.concat(res, axis=0).mean().to_frame().T
    else:
        res = OrderedDict()
        for i in range(yhat_mat.shape[1]):
            acc_res = eval_regression(y_mat[:,i], yhat_mat[:,i], key_prefix=key_prefix, list_metrics=list_metrics, augment_yhat_sign=augment_yhat_sign)
            for _k,_v in acc_res.items():
                res[_k] = res.get(_k, []) + [_v]
        return OrderedDict([
            (_k, np.nanmean(_v))
            if _k!="sign_confmat" else (_k, np.nanmean(np.concatenate(_v, 2), 2))
            for _k,_v in res.items()
        ])


##############################################################
#TODO: tensorflow evaluation functions
###############################################################
