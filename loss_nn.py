import numpy as np
import torch
from torch import nn
from typing import Any, Callable, ClassVar, List, Set, Dict, Union, Optional, Tuple


GEOMETRIES = ('cvar', 'chi-square')
MIN_REL_DIFFERENCE = 1e-5

# alias
MSELoss = nn.MSELoss


def huber(x, y, delta):
    """Huber loss between x and y, given huber threshold delta"""
    abserr = torch.abs(x - y)
    cond = abserr < delta
    return torch.where(cond, 0.5 * abserr ** 2, delta * (abserr - 0.5 * delta))


def chi_square_value(p, v, reg):
    """Returns <p, v> - reg * chi^2(p, uniform) for Torch tensors"""
    m = p.shape[0]

    with torch.no_grad():
        chi2 = (0.5 / m) * reg * (torch.norm(m * p - torch.ones(m, ), p=2) ** 2)

    return torch.dot(p, v) - chi2


def cvar_value(p, v, reg):
    """Returns <p, v> - reg * KL(p, uniform) for Torch tensors"""
    m = p.shape[0]

    with torch.no_grad():
        idx = torch.nonzero(p)  # where is annoyingly backwards incompatible
        kl = np.log(m) + (p[idx] * torch.log(p[idx])).sum()

    return torch.dot(p, v) - reg * kl


def fenchel_kl_cvar(v, alpha):
    """Returns the empirical mean of the Fenchel dual for KL CVaR"""
    v -= np.log(1 / alpha)
    v1 = v[torch.lt(v, 0)]
    v2 = v[torch.ge(v, 0)]
    w1 = torch.exp(v1) / alpha - 1
    w2 = (v2 + 1) * (1 / alpha) - 1
    return (w1.sum() + w2.sum()) / v.shape[0]


def bisection(eta_min, eta_max, f, tol=1e-6, max_iter=500):
    """Expects f an increasing function and return eta in [eta_min, eta_max]
    s.t. |f(eta)| <= tol (or the best solution after max_iter iterations"""
    lower = f(eta_min)
    upper = f(eta_max)

    # until the root is between eta_min and eta_max, double the length of the
    # interval starting at either endpoint.
    while lower > 0 or upper < 0:
        length = eta_max - eta_min
        if lower > 0:
            eta_max = eta_min
            eta_min = eta_min - 2 * length
        if upper < 0:
            eta_min = eta_max
            eta_max = eta_max + 2 * length

        lower = f(eta_min)
        upper = f(eta_max)

    for _ in range(max_iter):
        eta = 0.5 * (eta_min + eta_max)

        v = f(eta)

        if torch.abs(v) <= tol:
            return eta

        if v > 0:
            eta_max = eta
        elif v < 0:
            eta_min = eta

    # if the minimum is not reached in max_iter, returns the current value
    logging.warning('Maximum number of iterations exceeded in bisection')
    return 0.5 * (eta_min + eta_max)


class WeightedLoss(nn.Module):
    def __init__(
        self, loss_type: str, reduction: str = "mean",
        is_focal: bool = False, focal_beta: float = 0.2, focal_gamma: float = 1.0,
        is_poly1: bool = False, poly1_epsilon: float = 1.0,
        huber_delta: float = 0.5
    ):
        """Weighted loss, ranging in l2/l1/huber, focal or not, poly1 or not, weighted or not.

        Args:
            loss_type: str, ["l2", "l1", "huber"]
            reduction: str, ["mean", "sum", "none", "robust"], default mean
            is_focal: bool, whether it is focal
            focal_beta, focal_gamma: the hyperparameter of focal loss (used if is_focal is True)
            is_poly1: bool, whether poly1 is activated
            poly1_epsilon: float, the hyper-parameter of poly1 (if is_poly1 is True)
            huber_delta: the threshold parameter of huber loss (if loss_type is "huber")
        """
        super(WeightedLoss, self).__init__()
        self.loss_type = loss_type.lower()
        if self.loss_type not in ["l2", "l1", "huber"]:
            raise ValueError(f"Unrecognized loss_type={loss_type}")

        self.reduction_str = reduction.lower()
        if self.reduction_str == "mean":
            self.reduction = torch.mean
        elif self.reduction_str == "sum":
            self.reduction = torch.sum
        elif self.reduction_str == "none":
            self.reduction = nn.Identity()
        elif self.reduction_str == "robust":
            self.reduction = JenRobustLoss()
        else:
            raise ValueError(f"Unrecognized reduction={reduction}")

        self.is_focal = is_focal
        if self.is_focal:
            self.focal_beta = focal_beta
            self.focal_gamma = focal_gamma

        self.is_poly1 = is_poly1
        if self.is_poly1:
            if poly1_epsilon <= -1.0:
                raise ValueError("poly1_epsilon should be greater than -1.0 by definition")
            self.poly1_epsilon = poly1_epsilon

        if loss_type == "huber":
            if huber_delta <= 0.0:
                raise ValueError(f"Incorrect huber_delta={huber_delta}")
            self.huber_delta = huber_delta

    @property
    def str_name(self):
        _type_suf = f"#{self.huber_delta}" if self.loss_type == "huber" else ""
        _str_name = f"{self.loss_type}{_type_suf}|{self.reduction_str}"
        if self.is_focal:
            _str_name += f"|focal#{self.focal_beta}#{self.focal_gamma}"
        if self.is_poly1:
            _str_name += f"|poly1#{self.poly1_epsilon}"
        return _str_name

    def forward(self, x: torch.Tensor, y: torch.Tensor, weights: Optional[torch.Tensor] = None):
        if x.shape != y.shape:
            raise ValueError("The shapes of x and y are not matched (TODO: maybe broadcast?)")

        if self.loss_type == "l2":
            loss = (x - y) ** 2
        elif self.loss_type == "l1":
            loss = torch.abs(x - y)
        elif self.loss_type == "huber":
            loss = huber(x, y, self.huber_delta)
        else:
            raise NotImplementedError()

        if self.is_poly1:
            loss += self.poly1_epsilon * (1.0 - torch.exp( - loss))

        if self.is_focal:
            loss *= (torch.tanh(self.focal_beta * torch.abs(x - y))) ** self.focal_gamma

        if weights is not None:
            loss *= weights.expand_as(loss)

        return self.reduction(loss)


class WassersteinNormalUpper(nn.Module):
    def __init__(self, target_mean: float = 0.0, target_std: float = 1.0):
        """Upper bound of Wasserstein distance between batch sample and normal distribution.

        Args:
            target_mean: the mean of Normal density
            target_std: the std of Normal density
        """
        super(WassersteinNormalUpper, self).__init__()
        self.target_mean = target_mean
        self.target_std = target_std

    @property
    def str_name(self):
        return f"WasNormUB|N({self.target_mean},{self.target_std})"

    def forward(self, yhat):
        return (torch.mean(yhat) - self.target_mean)**2 \
            + (torch.std(yhat) - self.target_std)**2


class JenRobustLoss(nn.Module):
    def __init__(self, algorithm: str ="batch", size: float = 0.1, reg: float = 0.01, geometry: str = "cvar"):
        super(JenRobustLoss, self).__init__()
        if algorithm == 'dual':
            self.robust_loss = DualRobustLoss(size, reg, geometry)
        elif algorithm == 'batch':
            self.robust_loss = RobustLoss(size, reg, geometry)
        elif algorithm == 'erm':
            self.robust_loss = RobustLoss(0, 0, 'chi-square')
        else:
            raise ValueError('Unknown algorithm %s' % args.algorithm)

    def forward(self, per_sample_loss):
        return self.robust_loss(per_sample_loss)


class RobustLoss(nn.Module):
    """PyTorch module for the batch robust loss estimator"""
    def __init__(self, size, reg, geometry, tol=1e-4,
                 max_iter=1000, debugging=False):
        """
        Parameters
        ----------
        size : float
            Size of the uncertainty set (\rho for \chi^2 and \alpha for CVaR)
            Set float('inf') for unconstrained
        reg : float
            Strength of the regularizer, entropy if geometry == 'cvar'
            $\chi^2$ divergence if geometry == 'chi-square'
        geometry : string
            Element of GEOMETRIES
        tol : float, optional
            Tolerance parameter for the bisection
        max_iter : int, optional
            Number of iterations after which to break the bisection
        """
        super().__init__()
        self.size = size
        self.reg = reg
        self.geometry = geometry
        self.tol = tol
        self.max_iter = max_iter
        self.debugging = debugging

        self.is_erm = size == 0

        if geometry not in GEOMETRIES:
            raise ValueError('Geometry %s not supported' % geometry)

        if geometry == 'cvar' and self.size > 1:
            raise ValueError(f'alpha should be < 1 for cvar, is {self.size}')

    def best_response(self, v):
        size = self.size
        reg = self.reg
        m = v.shape[0]

        if self.geometry == 'cvar':
            if self.reg > 0:
                if size == 1.0:
                    return torch.ones_like(v) / m

                def p(eta):
                    x = (v - eta) / reg
                    return torch.min(torch.exp(x),
                                     torch.Tensor([1 / size]).type(x.dtype).to(x.device)) / m

                def bisection_target(eta):
                    return 1.0 - p(eta).sum()

                eta_min = reg * torch.logsumexp(v / reg - np.log(m), 0)
                eta_max = v.max()

                if torch.abs(bisection_target(eta_min)) <= self.tol:
                    return p(eta_min)
            else:
                cutoff = int(size * m)
                surplus = 1.0 - cutoff / (size * m)

                p = torch.zeros_like(v)
                idx = torch.argsort(v, descending=True)
                p[idx[:cutoff]] = 1.0 / (size * m)
                if cutoff < m:
                    p[idx[cutoff]] = surplus
                return p

        if self.geometry == 'chi-square':
            if (v.max() - v.min()) / v.max() <= MIN_REL_DIFFERENCE:
                return torch.ones_like(v) / m

            if size == float('inf'):
                assert reg > 0

                def p(eta):
                    return torch.relu(v - eta) / (reg * m)

                def bisection_target(eta):
                    return 1.0 - p(eta).sum()

                eta_min = min(v.sum() - reg * m, v.min())
                eta_max = v.max()

            else:
                assert size < float('inf')

                # failsafe for batch sizes small compared to
                # uncertainty set size
                if m <= 1 + 2 * size:
                    out = (v == v.max()).float()
                    out /= out.sum()
                    return out

                if reg == 0:
                    def p(eta):
                        pp = torch.relu(v - eta)
                        return pp / pp.sum()

                    def bisection_target(eta):
                        pp = p(eta)
                        w = m * pp - torch.ones_like(pp)
                        return 0.5 * torch.mean(w ** 2) - size

                    eta_min = -(1.0 / (np.sqrt(2 * size + 1) - 1)) * v.max()
                    eta_max = v.max()
                else:
                    def p(eta):
                        pp = torch.relu(v - eta)

                        opt_lam = max(
                            reg, torch.norm(pp) / np.sqrt(m * (1 + 2 * size))
                        )

                        return pp / (m * opt_lam)

                    def bisection_target(eta):
                        return 1 - p(eta).sum()

                    eta_min = v.min() - 1
                    eta_max = v.max()

        eta_star = bisection(
            eta_min, eta_max, bisection_target,
            tol=self.tol, max_iter=self.max_iter)

        if self.debugging:
            return p(eta_star), eta_star
        return p(eta_star)

    def forward(self, v):
        """Value of the robust loss
        Note that the best response is computed without gradients
        Parameters
        ----------
        v : torch.Tensor
            Tensor containing the individual losses on the batch of examples
        Returns
        -------
        loss : torch.float
            Value of the robust loss on the batch of examples
        """
        if self.is_erm:
            return v.mean()
        else:
            with torch.no_grad():
                p = self.best_response(v)

            if self.geometry == 'cvar':
                return cvar_value(p, v, self.reg)
            elif self.geometry == 'chi-square':
                return chi_square_value(p, v, self.reg)


class DualRobustLoss(torch.nn.Module):
    """Dual formulation of the robust loss, contains trainable parameter eta"""

    def __init__(self, size, reg, geometry, eta_init=0.0):
        """Constructor for the dual robust loss
        Parameters
        ----------
        size : float
            Size of the uncertainty set (\rho for \chi^2 and \alpha for CVaR)
            Set float('inf') for unconstrained
        reg : float
            Strength of the regularizer, entropy if geometry == 'cvar'
            \chi^2 divergence if geometry == 'chi-square'
        geometry : string
            Element of GEOMETRIES
        eta_init : float
            Initial value for equality constraint Lagrange multiplier eta
        """
        super().__init__()
        self.eta = torch.nn.Parameter(data=torch.Tensor([eta_init]))
        self.geometry = geometry
        self.size = size
        self.reg = reg

        if geometry not in GEOMETRIES:
            raise ValueError('Geometry %s not supported' % geometry)

    def forward(self, v):
        """Value of the dual loss on the batch of examples
        Parameters
        ----------
        v : torch.Tensor
            Tensor containing the individual losses on the batch of examples
        Returns
        -------
        loss : torch.float
            Value of the dual of the robust loss on the batch of examples
        """
        n = v.shape[0]

        if self.geometry == 'cvar':
            if self.reg == 0:
                return self.eta + torch.relu(v - self.eta).mean() / self.size
            else:
                return self.eta + self.reg * fenchel_kl_cvar(
                    (v - self.eta) / self.reg, self.size)

        elif self.geometry == 'chi-square':
            w = torch.relu(v - self.eta)

            if self.size == float('inf'):
                return ((0.5 / self.reg) * (w ** 2).mean()
                        + 0.5 * self.reg + self.eta)
            else:
                if self.reg == 0:
                    return self.eta + np.sqrt(
                        (1 + 2 * self.size) / n) * torch.norm(w, p=2)
                else:
                    return self.eta + 0.5 * self.reg + huber_loss(
                        torch.norm(w, p=2) / np.sqrt(n * self.reg),
                        delta=np.sqrt(self.reg * (1 + 2 * self.size)))


class CorrRWas(nn.Module):
    def __init__(self, was_target_mean: float = 0.0, was_target_std: float = 1.0, was_weight=1.0):
        """Loss of NEGATIVE-correlation with upper bound of Wasserstein distance between batch sample and normal distribution.

        Args:
            was_target_mean: the mean of Normal density for Wasserstein distance usage
            was_target_std: the std of Normal density for Wasserstein distance usage
            was_weight: the weight of was distance (to add to correlation)
        """
        super(CorrRWas, self).__init__()
        self.was_upper = WassersteinNormalUpper(target_mean=was_target_mean, target_std=was_target_std)
        self.was_weight = was_weight

    @property
    def str_name(self):
        return f"CorrRWas|{self.was_weight}|{self.was_upper.str_name}"

    def forward(self, yhat, y, **kwargs):
        cc = torch.corrcoef(torch.cat([yhat.view([1,-1]), y.view([1,-1])], 0))[0,1]
        return -cc + self.was_weight * self.was_upper(yhat)


