import torch
import torchmetrics as tm

device = "cuda" if torch.cuda.is_available() else "cpu"

def _get_distribution_mean(d: torch.Tensor):
    return d[:,[0, 2]]

def _get_distribution_covariance(d: torch.Tensor, corr=0.0):
    _d = d[:, [1, 3]]
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

    _t = torch.sqrt(torch.linalg.det(p_corr) * torch.linalg.det(t_corr))
    _t = (1/2) * torch.log(torch.linalg.det(sum_corr) / _t)
    
    return _x + _t

class BhattacharyyaDistance(tm.Metric):
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("distance", default=torch.tensor(0, dtype=torch.float, device=device), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, device=device), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # update metric states
        # preds, target = self._input_format(preds, target)
        # assert preds.shape == target.shape

        d = _calculate_distance(preds, target)
        nan_count = torch.sum(torch.isnan(d))
        d = torch.nan_to_num(d, nan=0.0)

        print(nan_count)

        self.distance += torch.sum(d)
        self.total += target.numel() - nan_count

    def compute(self):
        # compute final result
        return self.distance / self.total