import math
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import ConvSC, ConvNeXt_block, ConvNeXt_bottle, Attention

class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(1, half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Time_MLP(nn.Module):

    def __init__(self, dim: int=64):
        super().__init__()
        self.embed = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        return self.mlp(self.embed(x))

def stride_generator(N: int, reverse: bool=False) -> List[int]:
    s = ([1, 2] * 10)[:N]
    return list(reversed(s)) if reverse else s

def _as_cl(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous()

def _clamp01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)

class Encoder(nn.Module):

    def __init__(self, C_in: int, C_hid: int, N_S: int):
        super().__init__()
        strides = stride_generator(N_S)
        self.enc = nn.ModuleList([ConvSC(C_in, C_hid, stride=strides[0])] + [ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]])

    def forward(self, x: torch.Tensor):
        y = self.enc[0](x)
        skip = y
        for i in range(1, len(self.enc)):
            y = self.enc[i](y)
        return (y, skip)

class Decoder(nn.Module):

    def __init__(self, C_hid: int, N_S: int):
        super().__init__()
        strides = stride_generator(N_S, reverse=True)
        self.decs = nn.ModuleList([ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]])
        self.final = ConvSC(2 * C_hid, C_hid, stride=strides[-1], transpose=True)

    def forward(self, hid: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        z = hid
        for m in self.decs:
            z = m(z)
        z = self.final(torch.cat([z, skip], dim=1))
        return z

class ObservationChannelAttention(nn.Module):

    def __init__(self, channels: int, reduction: int=8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stat = x.mean(dim=(1, 3, 4))
        w = self.fc(stat)[:, None, :, None, None]
        return x * (0.5 + w)

class FutureTemporalAttention(nn.Module):

    def __init__(self, kernel_size: int=7):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stat = x.mean(dim=(2, 3, 4)).unsqueeze(1)
        w = torch.sigmoid(self.conv(stat)).squeeze(1)
        return x * (0.5 + w[:, :, None, None, None])

class NativeFilledDualQueueFusion(nn.Module):

    def __init__(self, channels: int, future_len: int, weak_echo_thr: float=0.15, mask_prob: float=0.35, time_kernel: int=7):
        super().__init__()
        self.future_len = int(future_len)
        self.weak_echo_thr = weak_echo_thr
        self.mask_prob = mask_prob
        self.obs_attn = ObservationChannelAttention(channels)
        self.future_attn = FutureTemporalAttention(time_kernel)
        self.mask_token = nn.Parameter(torch.zeros(self.future_len, channels, 16, 16))

    def _blank_queue(self, batch: int, height: int, width: int) -> torch.Tensor:
        token = self.mask_token
        if token.shape[-2:] != (height, width):
            token = F.interpolate(token, size=(height, width), mode='bilinear', align_corners=False)
        return token.unsqueeze(0).expand(batch, -1, -1, -1, -1).clone()

    def _adaptive_mask(self, filled: torch.Tensor, raw_filled: torch.Tensor, is_train: bool):
        if not is_train or self.mask_prob <= 0 or filled.shape[1] == 0:
            return filled
        (B, T, C, Hf, Wf) = filled.shape
        raw = raw_filled.reshape(B * T, raw_filled.shape[2], raw_filled.shape[3], raw_filled.shape[4])
        raw = F.interpolate(raw, size=(Hf, Wf), mode='bilinear', align_corners=False)
        raw = raw.reshape(B, T, 1, Hf, Wf)
        weak_mask = (raw < self.weak_echo_thr).float()
        rand_keep = (torch.rand_like(weak_mask) > self.mask_prob).float()
        keep_mask = torch.where(weak_mask > 0, rand_keep, torch.ones_like(weak_mask))
        return filled * keep_mask

    def forward(self, observed: torch.Tensor, raw_observed: torch.Tensor=None, filled_features: torch.Tensor=None, raw_filled: torch.Tensor=None, is_train: bool=False, enable_enhancement: bool=True) -> torch.Tensor:
        (B, T, _, H, W) = observed.shape
        obs_len = max(1, T // 2)
        qo = observed[:, :obs_len]
        qf_seed = observed[:, obs_len:]
        if enable_enhancement:
            qo = self.obs_attn(qo)
            if raw_observed is not None and qf_seed.shape[1] > 0:
                qf_seed = self._adaptive_mask(qf_seed, raw_observed[:, obs_len:], is_train=is_train)
        qf_blank = self._blank_queue(B, H, W)
        if filled_features is not None and filled_features.shape[1] > 0:
            fill_len = min(self.future_len, filled_features.shape[1])
            filled = filled_features[:, :fill_len]
            if raw_filled is not None:
                filled = self._adaptive_mask(filled, raw_filled[:, :fill_len], is_train=is_train)
            qf_blank[:, :fill_len] = filled
        qf = torch.cat([qf_seed, qf_blank], dim=1)
        if enable_enhancement:
            qf = self.future_attn(qf)
        return torch.cat([qo, qf], dim=1)

class TemporalPredictor(nn.Module):

    def __init__(self, Tin: int, C_hid: int, N_T: int):
        super().__init__()
        self.Tin = Tin
        in_ch = Tin * C_hid
        dim = max(1, in_ch // 2)
        if 2 * dim != in_ch:
            self.pre = nn.Conv2d(in_ch, 2 * dim, kernel_size=1)
            ch_for_blocks = 2 * dim
        else:
            self.pre = nn.Identity()
            ch_for_blocks = in_ch
        self.time_mlp = Time_MLP(dim=64)
        self.bottle = ConvNeXt_bottle(dim=ch_for_blocks)
        self.blocks = nn.ModuleList([ConvNeXt_block(dim=ch_for_blocks) for _ in range(N_T)])
        self.proj = nn.Conv2d(ch_for_blocks, C_hid, kernel_size=1)

    def forward(self, z_seq: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
        (B, T, C, H, W) = z_seq.shape
        z = z_seq.reshape(B, T * C, H, W)
        z = _as_cl(z)
        z = self.pre(z)
        t_emb = self.time_mlp(t_scalar)
        z = self.bottle(z, t_emb)
        for blk in self.blocks:
            z = blk(z, t_emb)
        z = self.proj(z)
        return z

class SpatialLongRangeBranch(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(channels, channels, kernel_size=(1, 5, 5), padding=(0, 2, 2), groups=channels, bias=False), nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 2, 2), dilation=(1, 2, 2), bias=False), nn.Conv3d(channels, channels, kernel_size=1, bias=False), nn.GroupNorm(4, channels), nn.SiLU())

    def forward(self, x):
        return self.net(x)

class TemporalDynamicBranch(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.temporal_attn = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True)

    def forward(self, x):
        pooled = x.mean(dim=(-1, -2))
        w = torch.sigmoid(self.temporal_attn(pooled)).unsqueeze(-1).unsqueeze(-1)
        return x * (0.5 + w)

class DynamicKernelAttention(nn.Module):

    def __init__(self, channels: int, strong_thr: float):
        super().__init__()
        self.strong_thr = strong_thr
        self.small = nn.Sequential(nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False), nn.Conv3d(channels, channels, kernel_size=1, bias=False), nn.GroupNorm(4, channels), nn.SiLU())
        self.large_spatial = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=9, padding=4, groups=channels, bias=False), nn.Conv2d(channels, channels, kernel_size=1, bias=False), nn.GroupNorm(4, channels), nn.SiLU())
        self.large_temporal = nn.Sequential(nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=channels, bias=False), nn.Conv3d(channels, channels, kernel_size=1, bias=False), nn.GroupNorm(4, channels), nn.SiLU())

    def forward(self, feat: torch.Tensor, coarse_pred: torch.Tensor):
        strong_mask = (coarse_pred > self.strong_thr).float()
        strong_mask = strong_mask.permute(0, 2, 1, 3, 4).contiguous()
        (B, C, T, H, W) = feat.shape
        out_small = self.small(feat)
        x2d = feat.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        out_large = self.large_spatial(x2d)
        out_large = out_large.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        out_large = self.large_temporal(out_large)
        out = strong_mask * out_small + (1.0 - strong_mask) * out_large
        return (out, strong_mask)

class StrongEchoGate(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv3d(channels + 1, channels, kernel_size=1, bias=True), nn.Sigmoid())

    def forward(self, feat: torch.Tensor, strong_mask: torch.Tensor):
        g = self.gate(torch.cat([feat, strong_mask], dim=1))
        return feat * (1.0 + g)

class MultiScaleResidualRefine(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.branch_spatial = nn.Sequential(nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), groups=channels, bias=False), nn.Conv3d(channels, channels, kernel_size=1, bias=False), nn.GroupNorm(4, channels), nn.SiLU())
        self.branch_temporal = nn.Sequential(nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=channels, bias=False), nn.Conv3d(channels, channels, kernel_size=1, bias=False), nn.GroupNorm(4, channels), nn.SiLU())
        self.branch_joint = nn.Sequential(nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False), nn.Conv3d(channels, channels, kernel_size=1, bias=False), nn.GroupNorm(4, channels), nn.SiLU())
        self.fuse = nn.Sequential(nn.Conv3d(channels * 3, channels, kernel_size=1, bias=False), nn.GroupNorm(4, channels), nn.SiLU())

    def forward(self, x: torch.Tensor, residual_base: torch.Tensor):
        a = self.branch_spatial(x)
        b = self.branch_temporal(x)
        c = self.branch_joint(x)
        fused = self.fuse(torch.cat([a, b, c], dim=1))
        return fused + residual_base

class SpatiotemporalFusionCalibrationModule(nn.Module):

    def __init__(self, in_channels: int, mid_channels: int=32, strong_echo_thr: float=0.3, delta_scale: float=0.5):
        super().__init__()
        self.strong_echo_thr = strong_echo_thr
        self.delta_scale = float(delta_scale)
        self.reshape = nn.Sequential(nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False), nn.GroupNorm(4, mid_channels), nn.SiLU())
        self.coarse_embed = nn.Sequential(nn.Conv3d(1, mid_channels, kernel_size=1, bias=False), nn.GroupNorm(4, mid_channels), nn.SiLU())
        self.spatial_branch = SpatialLongRangeBranch(mid_channels)
        self.temporal_branch = TemporalDynamicBranch(mid_channels)
        self.fuse = nn.Sequential(nn.Conv3d(mid_channels * 2, mid_channels, kernel_size=1, bias=False), nn.GroupNorm(4, mid_channels), nn.SiLU())
        self.dynamic_kernel = DynamicKernelAttention(mid_channels, strong_thr=strong_echo_thr)
        self.strong_gate = StrongEchoGate(mid_channels)
        self.multi_scale = MultiScaleResidualRefine(mid_channels)
        self.out_head = nn.Sequential(nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False), nn.GroupNorm(4, mid_channels), nn.SiLU(), nn.Conv3d(mid_channels, 1, kernel_size=1, bias=True))

    def forward(self, feat_volume: torch.Tensor, coarse_pred: torch.Tensor):
        f = feat_volume.permute(0, 2, 1, 3, 4).contiguous()
        base = self.reshape(f) + self.coarse_embed(coarse_pred.permute(0, 2, 1, 3, 4).contiguous())
        sp = self.spatial_branch(base)
        tp = self.temporal_branch(base)
        fused = self.fuse(torch.cat([sp, tp], dim=1))
        (adapt, strong_mask) = self.dynamic_kernel(fused, coarse_pred)
        gated = self.strong_gate(adapt, strong_mask)
        refined = self.multi_scale(gated, base)
        delta = self.out_head(refined).permute(0, 2, 1, 3, 4).contiguous()
        coarse_logit = torch.logit(coarse_pred.clamp(0.0001, 1.0 - 0.0001))
        return torch.sigmoid(coarse_logit + self.delta_scale * delta)

class MAIAM4VP(nn.Module):

    def __init__(self, in_shape: Tuple[int, int, int, int], hid_S: int=64, hid_T: int=512, N_S: int=4, N_T: int=6, Tout: int=12, weak_echo_thr: float=0.15, adaptive_mask_prob: float=0.35, daqo_time_kernel: int=7, strong_echo_thr: float=0.3, refine_mid_channels: int=32, refiner_delta_scale: float=0.5, disable_ma_daqo: bool=False, disable_ma_scm: bool=False, native_fill_teacher_forcing: float=0.5):
        super().__init__()
        (Tin, C, H, W) = in_shape
        self.Tin = Tin
        self.Tout = Tout
        self.disable_ma_daqo = disable_ma_daqo
        self.disable_ma_scm = disable_ma_scm
        self.native_fill_teacher_forcing = float(native_fill_teacher_forcing)
        self.encoder = Encoder(C_in=C, C_hid=hid_S, N_S=N_S)
        self.fill_encoder = Encoder(C_in=C, C_hid=hid_S, N_S=N_S)
        self.queue_opt = NativeFilledDualQueueFusion(channels=hid_S, future_len=Tout, weak_echo_thr=weak_echo_thr, mask_prob=adaptive_mask_prob, time_kernel=daqo_time_kernel)
        self.temporal = TemporalPredictor(Tin=Tin + Tout, C_hid=hid_S, N_T=N_T)
        self.decoder = Decoder(C_hid=hid_S, N_S=N_S)
        self.attn = Attention(hid_S)
        self.head = nn.Conv2d(hid_S, 1, kernel_size=1)
        self.refiner = SpatiotemporalFusionCalibrationModule(in_channels=hid_S, mid_channels=refine_mid_channels, strong_echo_thr=strong_echo_thr, delta_scale=refiner_delta_scale)

    def forward(self, x_seq: torch.Tensor, y_seq: torch.Tensor=None, is_train: bool=True, teacher_forcing_ratio: float=None):
        assert x_seq.dim() == 5 and x_seq.size(2) == 1, f'expect [B,Tin,1,H,W], got {tuple(x_seq.shape)}'
        (B, Tin, _, H, W) = x_seq.shape
        if teacher_forcing_ratio is None:
            teacher_forcing_ratio = self.native_fill_teacher_forcing
        teacher_forcing_ratio = float(min(1.0, max(0.0, teacher_forcing_ratio)))
        enc_list = []
        skip_first = None
        for t in range(Tin):
            xt = _as_cl(x_seq[:, t])
            (ft, skip) = self.encoder(xt)
            enc_list.append(ft.unsqueeze(1))
            if skip_first is None:
                skip_first = skip
        observed = torch.cat(enc_list, dim=1)
        filled_features = []
        raw_filled = []
        coarse_frames = []
        refined_frames = []
        feat_frames = []
        for step in range(self.Tout):
            filled_tensor = torch.cat(filled_features, dim=1) if filled_features else None
            raw_tensor = torch.cat(raw_filled, dim=1) if raw_filled else None
            queue = self.queue_opt(observed, raw_observed=x_seq, filled_features=filled_tensor, raw_filled=raw_tensor, is_train=is_train, enable_enhancement=not self.disable_ma_daqo)
            t_scalar = torch.full((B,), step * 100.0, device=x_seq.device, dtype=torch.float32)
            hid = self.temporal(queue, t_scalar)
            feat_2d = self.attn(self.decoder(hid, skip_first))
            coarse_frame = torch.sigmoid(self.head(feat_2d)).unsqueeze(1)
            step_feat = feat_2d.unsqueeze(1)
            if self.disable_ma_scm:
                refined_frame = coarse_frame
            else:
                refined_frame = self.refiner(step_feat, coarse_frame)
            coarse_frames.append(coarse_frame)
            refined_frames.append(refined_frame)
            feat_frames.append(step_feat)
            if step + 1 < self.Tout:
                use_gt = is_train and y_seq is not None and (step < y_seq.shape[1]) and (teacher_forcing_ratio > 0.0)
                if use_gt:
                    sample_mask = torch.rand(B, 1, 1, 1, 1, device=x_seq.device) < teacher_forcing_ratio
                    next_raw = torch.where(sample_mask, y_seq[:, step:step + 1], refined_frame.detach())
                else:
                    next_raw = refined_frame.detach()
                (next_feature, _) = self.fill_encoder(_as_cl(next_raw[:, 0]))
                filled_features.append(next_feature.unsqueeze(1))
                raw_filled.append(next_raw)
        coarse_pred = torch.cat(coarse_frames, dim=1)
        final_pred = torch.cat(refined_frames, dim=1)
        feat_volume = torch.cat(feat_frames, dim=1)
        return {'coarse': coarse_pred, 'final': final_pred, 'decoder_feat': feat_volume}
MeteorologicalAdaptiveQueueOptimizer = NativeFilledDualQueueFusion
MASCMRefiner = SpatiotemporalFusionCalibrationModule

class _OutputMixin:

    @staticmethod
    def _pack(pred: torch.Tensor):
        pred = _clamp01(pred)
        return {'coarse': pred, 'final': pred, 'decoder_feat': None}

class IAM4VP(nn.Module, _OutputMixin):

    def __init__(self, in_shape: Tuple[int, int, int, int], hid_S: int=64, hid_T: int=512, N_S: int=4, N_T: int=8, Tout: int=12, **kwargs):
        super().__init__()
        (Tin, C, _, _) = in_shape
        self.Tin = Tin
        self.Tout = Tout
        self.encoder = Encoder(C_in=C, C_hid=hid_S, N_S=N_S)
        self.temporal = TemporalPredictor(Tin=Tin, C_hid=hid_S, N_T=N_T)
        self.decoder = Decoder(C_hid=hid_S, N_S=N_S)
        self.attn = Attention(hid_S)
        self.head = nn.Conv2d(hid_S, Tout, kernel_size=1)

    def forward(self, x_seq: torch.Tensor, is_train: bool=True):
        (B, Tin, _, _, _) = x_seq.shape
        enc_list = []
        skip_first = None
        for t in range(Tin):
            (ft, skip) = self.encoder(_as_cl(x_seq[:, t]))
            enc_list.append(ft.unsqueeze(1))
            if skip_first is None:
                skip_first = skip
        z_seq = torch.cat(enc_list, dim=1)
        t_scalar = torch.full((B,), Tin * 100.0, device=x_seq.device, dtype=torch.float32)
        hid = self.temporal(z_seq, t_scalar)
        feat = self.attn(self.decoder(hid, skip_first))
        pred = torch.sigmoid(self.head(feat)).unsqueeze(2)
        return self._pack(pred)

class SimVPInceptionBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        branch_ch = max(8, channels // 4)
        self.b1 = nn.Conv2d(channels, branch_ch, kernel_size=1)
        self.b3 = nn.Conv2d(channels, branch_ch, kernel_size=3, padding=1)
        self.b5 = nn.Conv2d(channels, branch_ch, kernel_size=5, padding=2)
        self.b7 = nn.Conv2d(channels, branch_ch, kernel_size=7, padding=3)
        self.fuse = nn.Sequential(nn.GELU(), nn.Conv2d(branch_ch * 4, channels, kernel_size=1), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([self.b1(x), self.b3(x), self.b5(x), self.b7(x)], dim=1)
        return x + self.fuse(y)

class SimVP(nn.Module, _OutputMixin):

    def __init__(self, in_shape: Tuple[int, int, int, int], hid_S: int=64, hid_T: int=512, N_S: int=4, N_T: int=8, Tout: int=12, **kwargs):
        super().__init__()
        (Tin, C, _, _) = in_shape
        self.Tin = Tin
        self.Tout = Tout
        self.encoder = Encoder(C_in=C, C_hid=hid_S, N_S=N_S)
        self.decoder = Decoder(C_hid=hid_S, N_S=N_S)
        in_ch = Tin * hid_S
        mid_ch = max(hid_S, min(int(hid_T), 512))
        blocks = [nn.Conv2d(in_ch, mid_ch, kernel_size=1), nn.GELU()]
        for _ in range(max(1, int(N_T))):
            blocks.append(SimVPInceptionBlock(mid_ch))
        blocks.append(nn.Conv2d(mid_ch, hid_S, kernel_size=1))
        self.translator = nn.Sequential(*blocks)
        self.head = nn.Conv2d(hid_S, Tout, kernel_size=1)

    def forward(self, x_seq: torch.Tensor, is_train: bool=True):
        (B, Tin, _, _, _) = x_seq.shape
        feats = []
        skip_first = None
        for t in range(Tin):
            (ft, skip) = self.encoder(x_seq[:, t])
            feats.append(ft)
            if skip_first is None:
                skip_first = skip
        hid = self.translator(torch.cat(feats, dim=1))
        feat = self.decoder(hid, skip_first)
        pred = torch.sigmoid(self.head(feat)).unsqueeze(2)
        return self._pack(pred)

class _ConvLSTMCell(nn.Module):

    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: int=5):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_ch = hidden_ch
        self.conv = nn.Conv2d(in_ch + hidden_ch, hidden_ch * 4, kernel_size, padding=pad)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        gates = self.conv(torch.cat([x, h], dim=1))
        (i, f, o, g) = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return (h, c)

class PredRNNv2(nn.Module, _OutputMixin):

    def __init__(self, in_shape: Tuple[int, int, int, int], hid_S: int=64, N_T: int=8, Tout: int=12, N_S: int=4, **kwargs):
        super().__init__()
        (Tin, C, _, _) = in_shape
        self.Tin = Tin
        self.Tout = Tout
        self.encoder = Encoder(C_in=C, C_hid=hid_S, N_S=N_S)
        self.decoder = Decoder(C_hid=hid_S, N_S=N_S)
        layers = max(1, min(int(N_T), 4))
        cells = []
        for i in range(layers):
            cells.append(_ConvLSTMCell(hid_S, hid_S, kernel_size=5))
        self.cells = nn.ModuleList(cells)
        self.head = nn.Conv2d(hid_S, C, kernel_size=1)

    def forward(self, x_seq: torch.Tensor, is_train: bool=True):
        (B, Tin, _, _, _) = x_seq.shape
        encoded = []
        skip_first = None
        for t in range(Tin):
            (ft, skip) = self.encoder(x_seq[:, t])
            encoded.append(ft)
            if skip_first is None:
                skip_first = skip
        (_, _, H, W) = encoded[0].shape
        h = [x_seq.new_zeros(B, cell.hidden_ch, H, W) for cell in self.cells]
        c = [x_seq.new_zeros(B, cell.hidden_ch, H, W) for cell in self.cells]
        for t in range(Tin):
            inp = encoded[t]
            for (i, cell) in enumerate(self.cells):
                (h[i], c[i]) = cell(inp, h[i], c[i])
                inp = h[i]
        preds = []
        for _ in range(self.Tout):
            inp = h[-1]
            for (i, cell) in enumerate(self.cells):
                (h[i], c[i]) = cell(inp, h[i], c[i])
                inp = h[i]
            frame = torch.sigmoid(self.head(self.decoder(h[-1], skip_first)))
            preds.append(frame.unsqueeze(1))
        return self._pack(torch.cat(preds, dim=1))

class _Conv3DLSTMCell(nn.Module):

    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: Tuple[int, int, int]=(3, 5, 5)):
        super().__init__()
        pad = tuple((k // 2 for k in kernel_size))
        self.hidden_ch = hidden_ch
        self.conv = nn.Conv3d(in_ch + hidden_ch, hidden_ch * 4, kernel_size, padding=pad)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        gates = self.conv(torch.cat([x, h], dim=1))
        (i, f, o, g) = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return (h, c)

class E3DLSTM(nn.Module, _OutputMixin):

    def __init__(self, in_shape: Tuple[int, int, int, int], hid_S: int=64, N_T: int=8, Tout: int=12, N_S: int=4, **kwargs):
        super().__init__()
        (Tin, C, _, _) = in_shape
        self.Tin = Tin
        self.Tout = Tout
        self.encoder = Encoder(C_in=C, C_hid=hid_S, N_S=N_S)
        self.decoder = Decoder(C_hid=hid_S, N_S=N_S)
        layers = max(1, min(int(N_T), 3))
        cells = []
        for _ in range(layers):
            cells.append(_Conv3DLSTMCell(hid_S, hid_S))
        self.cells = nn.ModuleList(cells)
        self.head = nn.Conv3d(hid_S, C, kernel_size=1)

    def forward(self, x_seq: torch.Tensor, is_train: bool=True):
        (B, Tin, _, _, _) = x_seq.shape
        encoded = []
        skip_first = None
        for t in range(Tin):
            (ft, skip) = self.encoder(x_seq[:, t])
            encoded.append(ft.unsqueeze(2))
            if skip_first is None:
                skip_first = skip
        (_, _, _, H, W) = encoded[0].shape
        h = [x_seq.new_zeros(B, cell.hidden_ch, 1, H, W) for cell in self.cells]
        c = [x_seq.new_zeros(B, cell.hidden_ch, 1, H, W) for cell in self.cells]
        for t in range(Tin):
            inp = encoded[t]
            for (i, cell) in enumerate(self.cells):
                (h[i], c[i]) = cell(inp, h[i], c[i])
                inp = h[i]
        preds = []
        for _ in range(self.Tout):
            inp = h[-1]
            for (i, cell) in enumerate(self.cells):
                (h[i], c[i]) = cell(inp, h[i], c[i])
                inp = h[i]
            decoded = self.decoder(h[-1].squeeze(2), skip_first).unsqueeze(2)
            frame = torch.sigmoid(self.head(decoded))
            preds.append(frame.transpose(1, 2))
        return self._pack(torch.cat(preds, dim=1))

def build_model(model_name: str, in_shape: Tuple[int, int, int, int], **kwargs) -> nn.Module:
    name = model_name.lower().replace('_', '-')
    if name in ('ma-iam4vp', 'maiam4vp'):
        return MAIAM4VP(in_shape, **kwargs)
    if name == 'iam4vp':
        return IAM4VP(in_shape, **kwargs)
    if name == 'simvp':
        return SimVP(in_shape, **kwargs)
    if name in ('predrnn-v2', 'predrnnv2'):
        return PredRNNv2(in_shape, **kwargs)
    if name in ('e3d-lstm', 'e3dlstm'):
        return E3DLSTM(in_shape, **kwargs)
    raise ValueError(f'Unknown model_name: {model_name}')
