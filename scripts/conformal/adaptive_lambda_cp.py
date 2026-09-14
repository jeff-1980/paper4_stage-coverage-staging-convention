import numpy as np
from scipy.stats import spearmanr


def weighted_quantile(values, weights, q):
    values  = np.array(values,  dtype=float)
    weights = np.array(weights, dtype=float)
    idx     = np.argsort(values)
    values  = values[idx]
    weights = weights[idx]
    cdf     = np.cumsum(weights) / weights.sum()
    return float(values[np.searchsorted(cdf, q)])


class AdaptiveLambdaCP:
    """
    Rolling Weighted CP
    设计约束：buf中的样本必须来自同一台发动机的时序
    跨发动机场景改用 split_cp（见下方说明）
    """
    def __init__(self, alpha=0.10, W=300,
                 lambda_init=0.005,
                 alpha_prior=2.0, beta_prior=50.0,
                 sigma_min=1e-3, verbose=False):
        self.alpha   = alpha
        self.W       = W
        self.lam     = lambda_init
        self.a0      = alpha_prior
        self.b0      = beta_prior      # 降低先验强度
        self.sig_min = sigma_min
        self.verbose = verbose
        self.buf     = []
        self.lam_hist= []
        self.sigma2_d= None

    def reset(self):
        """切换到新发动机时重置缓冲区，保留λ_t"""
        self.buf      = []
        self.sigma2_d = None

    def _weights(self, t_now):
        raw = np.array(
            [np.exp(-self.lam*(t_now - ti))
             for ti, _ in self.buf] + [1.0])
        return raw / raw.sum()

    def _nr_update(self, t_now):
        if len(self.buf) < 15:
            return
        times  = np.array([ti for ti, _ in self.buf])
        scores = np.array([si for _, si in self.buf])
        delta_d = np.diff(times)
        delta_d = np.clip(delta_d, 1e-6, None)
        delta_s = np.diff(scores)
        rate    = delta_s / delta_d
        mu_s    = np.median(rate)

        # mu_s过小时NR无意义，跳过（不截断符号）
        if abs(mu_s) < 1e-6:
            return

        if self.verbose and mu_s <= 0:
            print(f'  [λ] t={t_now}: mu_s={mu_s:.4f} '
                  f'recovery phase')

        if (self.sigma2_d is None or
                len(self.buf) % 20 == 0):
            self.sigma2_d = np.var(rate) + 1e-8

        predicted  = self.lam * delta_d * mu_s
        residuals  = delta_s - predicted
        grad_ll    = (np.sum(residuals * delta_d * mu_s)
                      / self.sigma2_d)
        grad_prior = (self.a0 - 1.0)/self.lam - self.b0
        grad       = grad_ll + grad_prior

        hess_ll    = (-np.sum((delta_d*mu_s)**2)
                      / self.sigma2_d)
        hess_prior = -(self.a0 - 1.0) / self.lam**2
        hess       = hess_ll + hess_prior

        if abs(hess) < 1e-12:
            return

        delta_lam = -grad / hess
        delta_lam = np.clip(
            delta_lam, -0.5*self.lam, 0.5*self.lam)
        self.lam  = float(np.clip(
            self.lam + delta_lam, 1e-5, 0.5))

    def predict_interval(self, t_now, mu_t, sigma_t):
        sigma_t = max(float(sigma_t), self.sig_min)
        self._nr_update(t_now)
        self.lam_hist.append((t_now, self.lam))

        if len(self.buf) < 5:
            from scipy.stats import norm
            z = norm.ppf(1 - self.alpha/2)
            return mu_t - z*sigma_t, mu_t + z*sigma_t

        scores = [si for _, si in self.buf] + [np.inf]
        w      = self._weights(t_now)
        q_hat  = weighted_quantile(
            scores, w, 1 - self.alpha)
        return (mu_t - q_hat*sigma_t,
                mu_t + q_hat*sigma_t)

    def update(self, t_now, y_true, mu_t, sigma_t):
        sigma_t = max(float(sigma_t), self.sig_min)
        score   = abs(y_true - mu_t) / sigma_t
        score   = min(score, 10.0)
        self.buf.append((float(t_now), score))
        if len(self.buf) > self.W:
            self.buf.pop(0)
