import torch
import torchmetrics as tm

device = "cuda" if torch.cuda.is_available() else "cpu"

def _get_distribution_mean(d: torch.Tensor):
    return d[:,[0, 2]]

def _get_distribution_covariance(d: torch.Tensor, corr=0.0):
    _d = d[:, [1, 3]]
    # print(_d)
    ret = torch.zeros((len(d), 2, 2), device=device)
    for (i, x) in enumerate(map(lambda x: torch.diag(x), _d)):
        x[0][1] = x[1][0] = corr * x[0][0] * x[1][1]
        ret[i] = x
    return ret

def _calculate_distance(preds: torch.Tensor, target: torch.Tensor):

    p_mean = _get_distribution_mean(preds)
    p_corr = _get_distribution_covariance(preds)

    t_mean = _get_distribution_mean(target)
    t_corr = _get_distribution_covariance(target)

    sum_corr = (t_corr + p_corr) / 2

    sum_corr_inv = torch.inverse(sum_corr)

    _x_mean = p_mean - t_mean
    _x_mean = torch.unsqueeze(_x_mean, 1)
    _x_mean_t = torch.transpose(_x_mean, 1, 2)
    _x = torch.matmul(_x_mean, sum_corr_inv)
    _x = (1/8) * torch.matmul(_x, _x_mean_t)
    _x = torch.squeeze(_x)

    _t = torch.linalg.det(p_corr) * torch.linalg.det(t_corr)
    # print(p_corr)
    # print(t_corr)
    # print(_t)
    _t = torch.sqrt(_t)
    # _t = torch.nan_to_num(_t)
    # print(_t)
    _t = torch.linalg.det(sum_corr) / _t
    # print(_t)
    _t = 0.5 * torch.log(_t)
    # print(_t)
    print("-----------------")
    
    return _x + _t

class BhattacharyyaDistance(tm.Metric):
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("avg_d", default=torch.tensor(0, dtype=torch.float, device=device), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # update metric states
        # preds, target = self._input_format(preds, target)
        # assert preds.shape == target.shape

        d = torch.sum(_calculate_distance(preds, target))
        n = target.numel()
        self.avg_d += d / n

    def compute(self):
        # compute final result
        return self.avg_d