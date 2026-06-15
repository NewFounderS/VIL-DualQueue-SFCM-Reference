import torch
import torch.nn as nn
import torch.nn.functional as F

def sobel_edges(img: torch.Tensor) -> torch.Tensor:
    gx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    gy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    (gx, gy) = (gx.repeat(img.shape[1], 1, 1, 1), gy.repeat(img.shape[1], 1, 1, 1))
    (fx, fy) = (F.conv2d(img, gx, padding=1, groups=img.shape[1]), F.conv2d(img, gy, padding=1, groups=img.shape[1]))
    return torch.sqrt(fx * fx + fy * fy + 1e-12)

def ssim_loss(x: torch.Tensor, y: torch.Tensor, c1=0.01 ** 2, c2=0.03 ** 2):
    (mu_x, mu_y) = (F.avg_pool2d(x, 3, 1, 1), F.avg_pool2d(y, 3, 1, 1))
    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    score = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2) / ((mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-12)
    return (1.0 - score).mean()

class DSECLLoss(nn.Module):

    def __init__(self, alpha_early=0.3, beta_final=1.0, lambda_mse=1.0, lambda_ssim=0.2, lambda_edge=0.15, lambda_strong=0.4, lambda_temporal=0.1, strong_echo_thr=0.3, strong_echo_cap=2.0, single_supervision=False):
        super().__init__()
        (self.alpha_early, self.beta_final) = (alpha_early, beta_final)
        (self.lambda_mse, self.lambda_ssim, self.lambda_edge) = (lambda_mse, lambda_ssim, lambda_edge)
        (self.lambda_strong, self.lambda_temporal) = (lambda_strong, lambda_temporal)
        (self.strong_echo_thr, self.strong_echo_cap) = (strong_echo_thr, strong_echo_cap)
        self.single_supervision = single_supervision

    def strong_echo_loss(self, pred, gt):
        weight = 1.0 + torch.clamp((gt - self.strong_echo_thr) / max(1e-06, 1.0 - self.strong_echo_thr), 0.0, 1.0)
        return (weight.clamp(max=self.strong_echo_cap) * (pred - gt) ** 2).mean()

    @staticmethod
    def temporal_consistency_loss(pred, gt):
        return pred.new_tensor(0.0) if pred.shape[1] < 2 else F.l1_loss(pred[:, 1:] - pred[:, :-1], gt[:, 1:] - gt[:, :-1])

    def final_branch_loss(self, pred, gt):
        (mse, ssim_acc, edge_acc) = (F.mse_loss(pred, gt), pred.new_tensor(0.0), pred.new_tensor(0.0))
        for t in range(pred.shape[1]):
            ssim_acc += ssim_loss(pred[:, t], gt[:, t])
            edge_acc += F.l1_loss(sobel_edges(pred[:, t]), sobel_edges(gt[:, t]))
        (ssim, edge) = (ssim_acc / pred.shape[1], edge_acc / pred.shape[1])
        (strong, temporal) = (self.strong_echo_loss(pred, gt), self.temporal_consistency_loss(pred, gt))
        total = self.lambda_mse * mse + self.lambda_ssim * ssim + self.lambda_edge * edge + self.lambda_strong * strong + self.lambda_temporal * temporal
        items = {k: float(v.detach().item()) for (k, v) in {'mse': mse, 'ssim_loss': ssim, 'edge': edge, 'strong': strong, 'temporal': temporal}.items()}
        return (total, items)

    def forward(self, coarse_pred, final_pred, gt):
        early = F.mse_loss(coarse_pred, gt)
        (final_loss, items) = self.final_branch_loss(final_pred, gt)
        total = final_loss if self.single_supervision else self.alpha_early * early + self.beta_final * final_loss
        items.update(early=float(early.detach()), final=float(final_loss.detach()), total=float(total.detach()))
        return (total, items)
MADSCLoss = DSECLLoss
