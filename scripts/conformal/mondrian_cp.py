import numpy as np
from scipy.stats import binomtest
from conformal.adaptive_lambda_cp import (
    AdaptiveLambdaCP, weighted_quantile)


def detect_cp_cusum(scores, k_mult=0.7, h_mult=7.0):
    """
    CUSUM变点检测：找退化加速点
    scores: 得分时序（单台发动机内部）
    返回：变点位置idx（归一化后即阶段边界）
    """
    if len(scores) < 20:
        return len(scores) // 2
    mu0   = np.mean(scores[:10])
    sigma = np.std(scores[:10]) + 1e-8
    k     = k_mult * sigma
    h     = h_mult * sigma
    cusum = 0.0
    cp    = len(scores) - 1
    for i, s in enumerate(scores):
        cusum = max(0.0, cusum + (s - mu0) - k)
        if cusum > h:
            cp = i
            break
    return cp


def assign_stage(t, t_cp, n_total):
    """
    三阶段划分（CUSUM驱动）
    Early:  [0, t_cp//2)
    Middle: [t_cp//2, t_cp)
    Late:   [t_cp, n_total)
    """
    mid = t_cp // 2
    if t < mid:
        return 'early'
    elif t < t_cp:
        return 'middle'
    else:
        return 'late'


class MondirianCP:
    """
    Mondrian CP：每个退化阶段独立校准
    每个阶段维护独立的Rolling WCP实例
    保证：组内边际覆盖率 >= 1-alpha-epsilon
    """
    def __init__(self, alpha=0.10,
                 W_max=300, lambda_init=0.005,
                 total_lifetime_est=None,
                 stage_ratios=(0.15, 0.35, 0.50)):
        self.alpha  = alpha
        self.W_max  = W_max
        self.lam0   = lambda_init
        self.T_est  = total_lifetime_est
        self.ratios = stage_ratios   # early/mid/late比例
        self.stages = ['early', 'middle', 'late']
        self._init_cps()

    def _adaptive_W(self, ratio):
        if self.T_est is None:
            return self.W_max
        n_k = int(self.T_est * ratio)
        W   = min(self.W_max, int(n_k * 0.8))
        return max(W, 20)

    def _init_cps(self):
        self.cp = {}
        self.W_eff = {}
        for stg, ratio in zip(self.stages, self.ratios):
            W = self._adaptive_W(ratio)
            self.W_eff[stg] = W
            self.cp[stg] = AdaptiveLambdaCP(
                alpha=self.alpha,
                W=W,
                lambda_init=self.lam0)

    def predict_interval(self, t_now, mu_t, sigma_t,
                         stage):
        return self.cp[stage].predict_interval(
            t_now, mu_t, sigma_t)

    def update(self, t_now, y_true, mu_t,
               sigma_t, stage):
        self.cp[stage].update(
            t_now, y_true, mu_t, sigma_t)

    def update_lifetime_est(self, T_new):
        self.T_est = T_new
        for stg, ratio in zip(self.stages, self.ratios):
            W_new = self._adaptive_W(ratio)
            if W_new != self.W_eff[stg]:
                self.W_eff[stg] = W_new
                self.cp[stg].W  = W_new
