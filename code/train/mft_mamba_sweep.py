def train_function(params: dict):
    import os
    import sys
    import copy
    import time
    import json
    import csv
    import math
    import itertools
    import numpy as np
    import yaml
    import boto3
    import torch
    import torch.nn as nn
    import torch.utils.data as dataf
    import torch.backends.cudnn as cudnn
    from torch.nn import LayerNorm, Linear, Dropout
    from einops import rearrange
    from scipy import io
    from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
    from operator import truediv
    from pathlib import Path
    from datetime import datetime
    import pytz

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from libs.config.schema import ExperimentConfig
    from libs.utils import upload_directory_to_s3, gpu_mem
    from mamba_ssm.modules.mamba_simple import Mamba

    cudnn.deterministic = True
    cudnn.benchmark = False

    pst = pytz.timezone("America/Los_Angeles")
    training_start_time = datetime.now(pst)
    training_start_str = training_start_time.strftime("%Y-%m-%d_%H:%M")

    # =================================================================
    # Model Classes — MFT Transformer (baseline)
    # =================================================================

    class HetConv(nn.Module):
        def __init__(self, in_channels, out_channels, p=64, g=64):
            super().__init__()
            self.gwc = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, groups=g, padding=1
            )
            self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=p)

        def forward(self, x):
            return self.gwc(x) + self.pwc(x)

    class MCrossAttention(nn.Module):
        def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.1,
            proj_drop=0.1,
        ):
            super().__init__()
            self.num_heads = num_heads
            head_dim = dim // num_heads
            self.scale = qk_scale or head_dim**-0.5
            self.wq = nn.Linear(head_dim, dim, bias=qkv_bias)
            self.wk = nn.Linear(head_dim, dim, bias=qkv_bias)
            self.wv = nn.Linear(head_dim, dim, bias=qkv_bias)
            self.proj = nn.Linear(dim * num_heads, dim)
            self.proj_drop = nn.Dropout(proj_drop)

        def forward(self, x):
            B, N, C = x.shape
            q = self.wq(
                x[:, 0:1, ...].reshape(B, 1, self.num_heads, C // self.num_heads)
            ).permute(0, 2, 1, 3)
            k = self.wk(x.reshape(B, N, self.num_heads, C // self.num_heads)).permute(
                0, 2, 1, 3
            )
            v = self.wv(x.reshape(B, N, self.num_heads, C // self.num_heads)).permute(
                0, 2, 1, 3
            )
            attn = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
            attn = attn.softmax(dim=-1)
            x = torch.einsum("bhij,bhjd->bhid", attn, v).transpose(1, 2)
            x = x.reshape(B, 1, C * self.num_heads)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

    class Mlp(nn.Module):
        def __init__(self, dim, mlp_dim):
            super().__init__()
            self.fc1 = Linear(dim, mlp_dim)
            self.fc2 = Linear(mlp_dim, dim)
            self.act_fn = nn.GELU()
            self.dropout = Dropout(0.1)
            self._init_weights()

        def _init_weights(self):
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.normal_(self.fc1.bias, std=1e-6)
            nn.init.normal_(self.fc2.bias, std=1e-6)

        def forward(self, x):
            x = self.fc1(x)
            x = self.act_fn(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.dropout(x)
            return x

    class Block(nn.Module):
        def __init__(self, dim, num_heads, mlp_dim):
            super().__init__()
            self.attention_norm = LayerNorm(dim, eps=1e-6)
            self.ffn_norm = LayerNorm(dim, eps=1e-6)
            self.ffn = Mlp(dim, mlp_dim)
            self.attn = MCrossAttention(dim, num_heads)

        def forward(self, x):
            h = x
            x = self.attention_norm(x)
            x = self.attn(x)
            x = x + h
            h = x
            x = self.ffn_norm(x)
            x = self.ffn(x)
            x = x + h
            return x

    class TransformerEncoder(nn.Module):
        def __init__(self, dim, num_heads=8, mlp_dim=512, depth=2):
            super().__init__()
            self.layer = nn.ModuleList()
            self.encoder_norm = LayerNorm(dim, eps=1e-6)
            for _ in range(depth):
                layer = Block(dim, num_heads, mlp_dim)
                self.layer.append(copy.deepcopy(layer))

        def forward(self, x):
            for layer_block in self.layer:
                x = layer_block(x)
            encoded = self.encoder_norm(x)
            return encoded[:, 0]

    class MFT(nn.Module):
        def __init__(
            self,
            FM,
            NC,
            NCLidar,
            Classes,
            patch_size,
            ntokens,
            token_type,
            num_heads,
            mlp_dim,
            depth,
        ):
            super().__init__()
            self.ntokens = ntokens
            self.FM = FM
            self.patch_size = patch_size
            self.conv5 = nn.Sequential(
                nn.Conv3d(1, 8, (9, 3, 3), padding=(0, 1, 1), stride=1),
                nn.BatchNorm3d(8),
                nn.ReLU(),
            )
            self.conv6 = nn.Sequential(
                HetConv(
                    8 * (NC - 8),
                    FM * 4,
                    p=1,
                    g=(FM * 4) // 4 if (8 * (NC - 8)) % FM == 0 else (FM * 4) // 8,
                ),
                nn.BatchNorm2d(FM * 4),
                nn.ReLU(),
            )
            self.lidarConv = nn.Sequential(
                nn.Conv2d(NCLidar, FM * 4, 3, 1, 1), nn.BatchNorm2d(FM * 4), nn.GELU()
            )
            self.ca = TransformerEncoder(FM * 4, num_heads, mlp_dim, depth)
            self.out3 = nn.Linear(FM * 4, Classes)
            self.position_embeddings = nn.Parameter(torch.randn(1, ntokens + 1, FM * 4))
            self.dropout = nn.Dropout(0.1)
            torch.nn.init.xavier_uniform_(self.out3.weight)
            torch.nn.init.normal_(self.out3.bias, std=1e-6)
            self.token_wA = nn.Parameter(
                torch.empty(1, ntokens, FM * 4), requires_grad=True
            )
            torch.nn.init.xavier_normal_(self.token_wA)
            self.token_wV = nn.Parameter(
                torch.empty(1, FM * 4, FM * 4), requires_grad=True
            )
            torch.nn.init.xavier_normal_(self.token_wV)
            if token_type == "channel":
                self.token_wA_L = nn.Parameter(
                    torch.empty(1, 1, FM * 4), requires_grad=True
                )
                torch.nn.init.xavier_normal_(self.token_wA_L)
                self.token_wV_L = nn.Parameter(
                    torch.empty(1, FM * 4, FM * 4), requires_grad=True
                )
                torch.nn.init.xavier_normal_(self.token_wV_L)

        def forward(self, x1, x2):
            ps = self.patch_size
            x1 = x1.reshape(x1.shape[0], -1, ps, ps).unsqueeze(1)
            x1 = self.conv5(x1)
            x1 = x1.reshape(x1.shape[0], -1, ps, ps)
            x1 = self.conv6(x1)
            x1 = x1.flatten(2).transpose(-1, -2)
            wa = rearrange(self.token_wA.expand(x1.shape[0], -1, -1), "b h w -> b w h")
            A = rearrange(
                torch.einsum("bij,bjk->bik", x1, wa), "b h w -> b w h"
            ).softmax(dim=-1)
            VV = torch.einsum(
                "bij,bjk->bik", x1, self.token_wV.expand(x1.shape[0], -1, -1)
            )
            T = torch.einsum("bij,bjk->bik", A, VV)
            x2 = x2.reshape(x2.shape[0], -1, ps, ps)
            x2 = self.lidarConv(x2)
            x2 = x2.reshape(x2.shape[0], -1, ps**2).transpose(-1, -2)
            wa_L = rearrange(
                self.token_wA_L.expand(x2.shape[0], -1, -1), "b h w -> b w h"
            )
            A_L = rearrange(
                torch.einsum("bij,bjk->bik", x2, wa_L), "b h w -> b w h"
            ).softmax(dim=-1)
            VV_L = torch.einsum(
                "bij,bjk->bik", x2, self.token_wV_L.expand(x2.shape[0], -1, -1)
            )
            L = torch.einsum("bij,bjk->bik", A_L, VV_L)
            x = torch.cat((L, T), dim=1)
            x = self.dropout(x + self.position_embeddings)
            x = self.ca(x)
            return self.out3(x.reshape(x.shape[0], -1))

    # =================================================================
    # Model Classes — Mamba-1 SSM variant
    # =================================================================

    class BiMambaEncoderBlock(nn.Module):
        def __init__(self, dim, d_state, d_conv, expand):
            super().__init__()
            self.norm = LayerNorm(dim, eps=1e-6)
            self.mamba_fwd = Mamba(
                d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand
            )
            self.mamba_bwd = Mamba(
                d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand
            )

        def forward(self, x):
            z = self.norm(x)
            fwd = self.mamba_fwd(z)
            bwd = self.mamba_bwd(z.flip(1)).flip(1)
            return x + (fwd + bwd) * 0.5

    class MambaEncoder(nn.Module):
        def __init__(self, dim, depth, d_state, d_conv, expand):
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    BiMambaEncoderBlock(dim, d_state, d_conv, expand)
                    for _ in range(depth)
                ]
            )
            self.norm = LayerNorm(dim, eps=1e-6)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return self.norm(x).mean(dim=1)

    class MFTMamba(nn.Module):
        def __init__(
            self,
            FM,
            NC,
            NCLidar,
            Classes,
            patch_size,
            ntokens,
            token_type,
            depth,
            d_state,
            d_conv,
            expand,
        ):
            super().__init__()
            self.ntokens = ntokens
            self.FM = FM
            self.patch_size = patch_size
            self.conv5 = nn.Sequential(
                nn.Conv3d(1, 8, (9, 3, 3), padding=(0, 1, 1), stride=1),
                nn.BatchNorm3d(8),
                nn.ReLU(),
            )
            self.conv6 = nn.Sequential(
                HetConv(
                    8 * (NC - 8),
                    FM * 4,
                    p=1,
                    g=(FM * 4) // 4 if (8 * (NC - 8)) % FM == 0 else (FM * 4) // 8,
                ),
                nn.BatchNorm2d(FM * 4),
                nn.ReLU(),
            )
            self.lidarConv = nn.Sequential(
                nn.Conv2d(NCLidar, FM * 4, 3, 1, 1), nn.BatchNorm2d(FM * 4), nn.GELU()
            )
            self.ca = MambaEncoder(FM * 4, depth, d_state, d_conv, expand)
            self.out3 = nn.Linear(FM * 4, Classes)
            self.position_embeddings = nn.Parameter(torch.randn(1, ntokens + 1, FM * 4))
            self.dropout = nn.Dropout(0.1)
            torch.nn.init.xavier_uniform_(self.out3.weight)
            torch.nn.init.normal_(self.out3.bias, std=1e-6)
            self.token_wA = nn.Parameter(
                torch.empty(1, ntokens, FM * 4), requires_grad=True
            )
            torch.nn.init.xavier_normal_(self.token_wA)
            self.token_wV = nn.Parameter(
                torch.empty(1, FM * 4, FM * 4), requires_grad=True
            )
            torch.nn.init.xavier_normal_(self.token_wV)
            if token_type == "channel":
                self.token_wA_L = nn.Parameter(
                    torch.empty(1, 1, FM * 4), requires_grad=True
                )
                torch.nn.init.xavier_normal_(self.token_wA_L)
                self.token_wV_L = nn.Parameter(
                    torch.empty(1, FM * 4, FM * 4), requires_grad=True
                )
                torch.nn.init.xavier_normal_(self.token_wV_L)

        def forward(self, x1, x2):
            ps = self.patch_size
            x1 = x1.reshape(x1.shape[0], -1, ps, ps).unsqueeze(1)
            x1 = self.conv5(x1)
            x1 = x1.reshape(x1.shape[0], -1, ps, ps)
            x1 = self.conv6(x1)
            x1 = x1.flatten(2).transpose(-1, -2)
            wa = rearrange(self.token_wA.expand(x1.shape[0], -1, -1), "b h w -> b w h")
            A = rearrange(
                torch.einsum("bij,bjk->bik", x1, wa), "b h w -> b w h"
            ).softmax(dim=-1)
            VV = torch.einsum(
                "bij,bjk->bik", x1, self.token_wV.expand(x1.shape[0], -1, -1)
            )
            T = torch.einsum("bij,bjk->bik", A, VV)
            x2 = x2.reshape(x2.shape[0], -1, ps, ps)
            x2 = self.lidarConv(x2)
            x2 = x2.reshape(x2.shape[0], -1, ps**2).transpose(-1, -2)
            wa_L = rearrange(
                self.token_wA_L.expand(x2.shape[0], -1, -1), "b h w -> b w h"
            )
            A_L = rearrange(
                torch.einsum("bij,bjk->bik", x2, wa_L), "b h w -> b w h"
            ).softmax(dim=-1)
            VV_L = torch.einsum(
                "bij,bjk->bik", x2, self.token_wV_L.expand(x2.shape[0], -1, -1)
            )
            L = torch.einsum("bij,bjk->bik", A_L, VV_L)
            x = torch.cat((L, T), dim=1)
            x = self.dropout(x + self.position_embeddings)
            x = self.ca(x)
            return self.out3(x.reshape(x.shape[0], -1))

    # =================================================================
    # FLOP Calculation
    # =================================================================

    def compute_flops_transformer(
        FM, NC, NC_LIDAR, patch_size, ntokens, num_heads, mlp_dim, depth, num_classes
    ):
        """Analytical FLOPs for MFT Transformer (single sample, forward pass)."""
        ps = patch_size
        ps2 = ps * ps
        dim = FM * 4
        seq_len = ntokens + 1
        head_dim = dim // num_heads

        flops = {}
        D_out = NC - 8
        flops["conv5"] = (
            2 * 8 * (9 * 3 * 3 * 1) * D_out * ps * ps + 5 * 8 * D_out * ps * ps
        )
        C_in = 8 * (NC - 8)
        g = dim // 4 if C_in % FM == 0 else dim // 8
        flops["conv6"] = (
            2 * dim * (9 * C_in // g) * ps2 + 2 * dim * C_in * ps2 + 6 * dim * ps2
        )
        flops["hsi_token"] = (
            2 * ps2 * dim * ntokens
            + 5 * ps2 * ntokens
            + 2 * ps2 * dim * dim
            + 2 * ntokens * ps2 * dim
        )
        flops["lidar_conv"] = 2 * dim * 9 * NC_LIDAR * ps2 + 5 * dim * ps2
        flops["lidar_token"] = (
            2 * ps2 * dim + 5 * ps2 + 2 * ps2 * dim * dim + 2 * ps2 * dim
        )
        flops["pos_embed"] = seq_len * dim

        block = {}
        block["attn_norm"] = 5 * seq_len * dim
        block["wq"] = 2 * 1 * num_heads * head_dim * dim
        block["wk"] = 2 * seq_len * num_heads * head_dim * dim
        block["wv"] = 2 * seq_len * num_heads * head_dim * dim
        block["attn_qk"] = 2 * num_heads * seq_len * dim
        block["attn_softmax"] = 6 * num_heads * seq_len
        block["attn_v"] = 2 * num_heads * seq_len * dim
        block["proj"] = 2 * (dim * num_heads) * dim
        block["attn_res"] = seq_len * dim
        block["ffn_norm"] = 5 * seq_len * dim
        block["ffn_fc1"] = 2 * seq_len * dim * mlp_dim
        block["ffn_gelu"] = seq_len * mlp_dim
        block["ffn_fc2"] = 2 * seq_len * mlp_dim * dim
        block["ffn_res"] = seq_len * dim
        block_total = sum(block.values())
        encoder_flops = depth * block_total + 5 * seq_len * dim

        flops["classifier"] = 2 * dim * num_classes
        other_flops = sum(flops.values())
        total_flops = other_flops + encoder_flops
        return total_flops, encoder_flops

    def compute_flops_mamba(
        FM,
        NC,
        NC_LIDAR,
        patch_size,
        ntokens,
        d_state,
        d_conv,
        expand,
        depth,
        num_classes,
    ):
        """Analytical FLOPs for MFT-Mamba (single sample, forward pass)."""
        ps = patch_size
        ps2 = ps * ps
        dim = FM * 4
        seq_len = ntokens + 1
        d_inner = dim * expand
        dt_rank = max(1, (dim + 15) // 16)

        flops = {}
        D_out = NC - 8
        flops["conv5"] = (
            2 * 8 * (9 * 3 * 3 * 1) * D_out * ps * ps + 5 * 8 * D_out * ps * ps
        )
        C_in = 8 * (NC - 8)
        g = dim // 4 if C_in % FM == 0 else dim // 8
        flops["conv6"] = (
            2 * dim * (9 * C_in // g) * ps2 + 2 * dim * C_in * ps2 + 6 * dim * ps2
        )
        flops["hsi_token"] = (
            2 * ps2 * dim * ntokens
            + 5 * ps2 * ntokens
            + 2 * ps2 * dim * dim
            + 2 * ntokens * ps2 * dim
        )
        flops["lidar_conv"] = 2 * dim * 9 * NC_LIDAR * ps2 + 5 * dim * ps2
        flops["lidar_token"] = (
            2 * ps2 * dim + 5 * ps2 + 2 * ps2 * dim * dim + 2 * ps2 * dim
        )
        flops["pos_embed"] = seq_len * dim

        # BiMamba block (2x Mamba — fwd + bwd)
        single_mamba = {}
        single_mamba["in_proj"] = 2 * seq_len * dim * (d_inner * 2)
        single_mamba["conv1d"] = 2 * seq_len * d_inner * d_conv
        single_mamba["silu"] = seq_len * d_inner
        single_mamba["x_proj"] = 2 * seq_len * d_inner * (dt_rank + d_state * 2)
        single_mamba["dt_proj"] = 2 * seq_len * dt_rank * d_inner
        single_mamba["ssm_scan"] = 8 * seq_len * d_inner * d_state
        single_mamba["d_skip"] = 2 * seq_len * d_inner
        single_mamba["out_gate"] = 3 * seq_len * d_inner
        single_mamba["out_proj"] = 2 * seq_len * d_inner * dim
        single_mamba_total = sum(single_mamba.values())

        block_flops = (
            5 * seq_len * dim  # layernorm
            + 2 * single_mamba_total  # fwd + bwd
            + seq_len * dim
        )  # residual
        encoder_flops = depth * block_flops + 5 * seq_len * dim  # final norm

        flops["classifier"] = 2 * dim * num_classes
        other_flops = sum(flops.values())
        total_flops = other_flops + encoder_flops
        return total_flops, encoder_flops

    # =================================================================
    # Helpers
    # =================================================================

    DATASETS_WITH_HSI_PARTS = ["Berlin", "Augsburg"]
    DATA2_LIST = ["SAR", "DSM", "MS"]

    def get_device(config_device="auto"):
        return torch.device(
            "cuda:0"
            if config_device == "auto" and torch.cuda.is_available()
            else "cpu" if config_device == "auto" else config_device
        )

    def aa_and_each_class_accuracy(conf_matrix):
        list_diag = np.diag(conf_matrix)
        list_raw_sum = np.sum(conf_matrix, axis=1)
        each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
        return each_acc, np.mean(each_acc)

    def evaluate(model, x_hsi, x_lidar, y_true, batch_size=500, device="cuda"):
        model.eval()
        pred_y = np.empty(len(y_true), dtype=np.float32)
        num_batches = len(y_true) // batch_size
        with torch.no_grad():
            for i in range(num_batches):
                s, e = i * batch_size, (i + 1) * batch_size
                out = model(x_hsi[s:e].to(device), x_lidar[s:e].to(device))
                pred_y[s:e] = torch.max(out, 1)[1].cpu().numpy()
            if num_batches * batch_size < len(y_true):
                s = num_batches * batch_size
                out = model(x_hsi[s:].to(device), x_lidar[s:].to(device))
                pred_y[s:] = torch.max(out, 1)[1].cpu().numpy()
        pred_y = torch.from_numpy(pred_y).long()
        oa = accuracy_score(y_true, pred_y) * 100
        conf = confusion_matrix(y_true, pred_y)
        each_acc, aa = aa_and_each_class_accuracy(conf)
        kappa = cohen_kappa_score(y_true, pred_y) * 100
        return oa, aa * 100, kappa, each_acc * 100

    def download_s3_prefix(bucket_name, s3_prefix, local_dir):
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                s3_key = obj["Key"]
                relative_path = os.path.relpath(s3_key, s3_prefix)
                local_path = os.path.join(local_dir, relative_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                s3.download_file(bucket_name, s3_key, local_path)
                print(
                    f"  Downloaded: {relative_path} ({os.path.getsize(local_path)/1024/1024:.2f} MB)"
                )

    def load_dataset(config, data_dir):
        dataset_name = config.data.dataset_name.value
        patch_size = config.data.patch_size
        data1_name, data2_name = "", ""
        if dataset_name in ["Houston", "Trento", "MUUFL"]:
            data1_name, data2_name = dataset_name, "LIDAR"
        else:
            for d2 in DATA2_LIST:
                if d2 in dataset_name:
                    data1_name = dataset_name.replace(d2, "")
                    data2_name = d2
                    break
        base = os.path.join(data_dir, f"{data1_name}{patch_size}x{patch_size}")
        hsi_tr = io.loadmat(os.path.join(base, "HSI_Tr.mat"))["Data"].astype(np.float32)
        lidar_tr = io.loadmat(os.path.join(base, f"{data2_name}_Tr.mat"))[
            "Data"
        ].astype(np.float32)
        tr_label = io.loadmat(os.path.join(base, "TrLabel.mat"))["Data"]
        if data1_name in DATASETS_WITH_HSI_PARTS:
            i = 2
            bp = os.path.join(base, "HSI_Te_Part")
            hsi_te = io.loadmat(f"{bp}1.mat")["Data"]
            while Path(f"{bp}{i}.mat").exists():
                hsi_te = np.concatenate(
                    [hsi_te, io.loadmat(f"{bp}{i}.mat")["Data"]], axis=0
                )
                i += 1
        else:
            hsi_te = io.loadmat(os.path.join(base, "HSI_Te.mat"))["Data"]
        hsi_te = hsi_te.astype(np.float32)
        lidar_te = io.loadmat(os.path.join(base, f"{data2_name}_Te.mat"))[
            "Data"
        ].astype(np.float32)
        te_label = io.loadmat(os.path.join(base, "TeLabel.mat"))["Data"]
        NC = hsi_tr.shape[3]
        NC_LIDAR = lidar_tr.shape[3]

        def to_tensors(hsi, lidar, label):
            h = (
                torch.from_numpy(hsi)
                .float()
                .permute(0, 3, 1, 2)
                .reshape(hsi.shape[0], hsi.shape[3], -1)
            )
            l = (
                torch.from_numpy(lidar)
                .float()
                .permute(0, 3, 1, 2)
                .reshape(lidar.shape[0], lidar.shape[3], -1)
            )
            y = torch.from_numpy(label).long().reshape(-1) - 1
            return h, l, y

        train_hsi, train_lidar, train_label = to_tensors(hsi_tr, lidar_tr, tr_label)
        test_hsi, test_lidar, test_label = to_tensors(hsi_te, lidar_te, te_label)
        num_classes = len(np.unique(train_label.numpy()))
        return (
            train_hsi,
            train_lidar,
            train_label,
            test_hsi,
            test_lidar,
            test_label,
            NC,
            NC_LIDAR,
            num_classes,
            data2_name,
        )

    # =================================================================
    # Training functions
    # =================================================================

    def train_baseline_run(
        config,
        train_hsi,
        train_lidar,
        train_label,
        test_hsi,
        test_lidar,
        test_label,
        NC,
        NC_LIDAR,
        num_classes,
        run_idx,
        device,
    ):
        """Train MFT Transformer baseline for one run."""
        seed = config.seed + run_idx
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        model = MFT(
            FM=config.model.feature_maps,
            NC=NC,
            NCLidar=NC_LIDAR,
            Classes=num_classes,
            patch_size=config.data.patch_size,
            ntokens=config.model.num_hsi_tokens,
            token_type="channel",
            num_heads=config.model.num_heads,
            mlp_dim=config.model.mlp_hidden_dim,
            depth=config.model.num_encoder_layers,
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        encoder_params = sum(p.numel() for p in model.ca.parameters())

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
        )
        loss_fn = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler.step_size,
            gamma=config.scheduler.gamma,
        )

        dataset = dataf.TensorDataset(train_hsi, train_lidar, train_label)
        train_loader = dataf.DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
        )

        best_oa, best_aa, best_kappa, best_epoch = 0.0, 0.0, 0.0, 0
        num_epochs = config.num_train_epochs

        torch.cuda.synchronize()
        t0 = time.time()

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            n_steps = 0
            for bx1, bx2, by in train_loader:
                bx1, bx2, by = bx1.to(device), bx2.to(device), by.to(device)
                loss = loss_fn(model(bx1, bx2), by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_steps += 1
            scheduler.step()

            if epoch == 0 or (epoch + 1) % 25 == 0 or epoch == num_epochs - 1:
                oa, aa, kappa, _ = evaluate(
                    model,
                    test_hsi,
                    test_lidar,
                    test_label,
                    batch_size=500,
                    device=device,
                )
                if oa > best_oa:
                    best_oa, best_aa, best_kappa, best_epoch = oa, aa, kappa, epoch + 1

        torch.cuda.synchronize()
        train_time = time.time() - t0

        total_flops, encoder_flops = compute_flops_transformer(
            FM=config.model.feature_maps,
            NC=NC,
            NC_LIDAR=NC_LIDAR,
            patch_size=config.data.patch_size,
            ntokens=config.model.num_hsi_tokens,
            num_heads=config.model.num_heads,
            mlp_dim=config.model.mlp_hidden_dim,
            depth=config.model.num_encoder_layers,
            num_classes=num_classes,
        )

        return {
            "total_params": total_params,
            "encoder_params": encoder_params,
            "total_flops": total_flops,
            "encoder_flops": encoder_flops,
            "best_oa": best_oa,
            "best_aa": best_aa,
            "best_kappa": best_kappa,
            "best_epoch": best_epoch,
            "train_time": train_time,
        }

    def train_mamba_run(
        hp,
        FM,
        NC,
        NC_LIDAR,
        num_classes,
        patch_size,
        ntokens,
        train_hsi,
        train_lidar,
        train_label,
        test_hsi,
        test_lidar,
        test_label,
        num_epochs,
        batch_size,
        run_idx,
        device,
    ):
        """Train one Mamba config for one run."""
        seed = 42 + run_idx
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        model = MFTMamba(
            FM=FM,
            NC=NC,
            NCLidar=NC_LIDAR,
            Classes=num_classes,
            patch_size=patch_size,
            ntokens=ntokens,
            token_type="channel",
            depth=hp["depth"],
            d_state=hp["d_state"],
            d_conv=hp["d_conv"],
            expand=hp["expand"],
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        encoder_params = sum(p.numel() for p in model.ca.parameters())

        optimizer = torch.optim.Adam(
            model.parameters(), lr=hp["lr"], weight_decay=hp["wd"]
        )
        label_smooth = hp.get("label_smooth", 0.0)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smooth)
        warmup_epochs = hp.get("warmup", 0)

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        dataset = dataf.TensorDataset(train_hsi, train_lidar, train_label)
        train_loader = dataf.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

        best_oa, best_aa, best_kappa, best_epoch = 0.0, 0.0, 0.0, 0

        torch.cuda.synchronize()
        t0 = time.time()

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            n_steps = 0
            for bx1, bx2, by in train_loader:
                bx1, bx2, by = bx1.to(device), bx2.to(device), by.to(device)
                loss = loss_fn(model(bx1, bx2), by)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_steps += 1
            scheduler.step()

            if epoch == 0 or (epoch + 1) % 25 == 0 or epoch == num_epochs - 1:
                oa, aa, kappa, _ = evaluate(
                    model,
                    test_hsi,
                    test_lidar,
                    test_label,
                    batch_size=500,
                    device=device,
                )
                if oa > best_oa:
                    best_oa, best_aa, best_kappa, best_epoch = oa, aa, kappa, epoch + 1

        torch.cuda.synchronize()
        train_time = time.time() - t0

        total_flops, encoder_flops = compute_flops_mamba(
            FM=FM,
            NC=NC,
            NC_LIDAR=NC_LIDAR,
            patch_size=patch_size,
            ntokens=ntokens,
            d_state=hp["d_state"],
            d_conv=hp["d_conv"],
            expand=hp["expand"],
            depth=hp["depth"],
            num_classes=num_classes,
        )

        return {
            "total_params": total_params,
            "encoder_params": encoder_params,
            "total_flops": total_flops,
            "encoder_flops": encoder_flops,
            "best_oa": best_oa,
            "best_aa": best_aa,
            "best_kappa": best_kappa,
            "best_epoch": best_epoch,
            "train_time": train_time,
        }

    # =================================================================
    # Grid
    # =================================================================

    SWEEP_GRID = {
        "d_state": [16, 32],
        "d_conv": [4],
        "expand": [1, 2],
        "depth": [1, 2, 3],
        "lr": [1e-4, 5e-5, 2e-5],
        "wd": [5e-3, 1e-2],
        "label_smooth": [0.0, 0.1],
        "warmup": [10],
    }

    NUM_RUNS = 3
    MAMBA_EPOCHS = 400

    # =================================================================
    # Main
    # =================================================================

    config_path = params.get("config_path")
    if not config_path:
        raise ValueError("Must provide 'config_path' in params")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = ExperimentConfig(**config_dict)

    device = get_device(config.device.device)
    CURRENT_DIR_PATH = os.getcwd()
    INPUTS_DIR_PATH = os.path.join(CURRENT_DIR_PATH, "inputs")
    dataset_name = config.data.dataset_name.value
    RUN_NAME = f"mft_mamba_benchmark_{dataset_name}_{training_start_str}"
    OUTPUTS_DIR_PATH = os.path.join(CURRENT_DIR_PATH, "outputs", RUN_NAME)
    os.makedirs(INPUTS_DIR_PATH, exist_ok=True)
    os.makedirs(OUTPUTS_DIR_PATH, exist_ok=True)

    patch_size = config.data.patch_size
    # Extract base dataset name (e.g., "AugsburgSAR" → "Augsburg")
    # so S3 folder matches load_dataset() which uses data1_name
    data1_name = dataset_name
    if dataset_name not in ["Houston", "Trento", "MUUFL"]:
        for d2 in DATA2_LIST:
            if d2 in dataset_name:
                data1_name = dataset_name.replace(d2, "")
                break
    dataset_folder = f"{data1_name}{patch_size}x{patch_size}"
    dataset_s3_prefix = f"{config.inputs.data_s3_prefix.rstrip('/')}/{dataset_folder}"

    # Build grid
    keys = list(SWEEP_GRID.keys())
    values = list(SWEEP_GRID.values())
    all_combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    total_configs = len(all_combos)

    print(f"\n{'='*80}")
    print(f"[MFT-Mamba Benchmark] Dataset: {dataset_name}")
    print(
        f"  Phase 1: MFT Transformer baseline ({NUM_RUNS} runs, {config.num_train_epochs} epochs)"
    )
    print(
        f"  Phase 2: Mamba grid search ({total_configs} configs x {NUM_RUNS} runs = {total_configs * NUM_RUNS} runs, {MAMBA_EPOCHS} epochs)"
    )
    print(f"  Grid:")
    for k, v in SWEEP_GRID.items():
        print(f"    {k}: {v}")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*80}\n", flush=True)

    # --- Download data ---
    print(f"[Step 1] Downloading dataset...")
    download_s3_prefix(
        config.inputs.bucket_name,
        dataset_s3_prefix,
        os.path.join(INPUTS_DIR_PATH, dataset_folder),
    )
    print(f"  Done.\n", flush=True)

    # --- Load data ---
    print(f"[Step 2] Loading data...")
    (
        train_hsi,
        train_lidar,
        train_label,
        test_hsi,
        test_lidar,
        test_label,
        NC,
        NC_LIDAR,
        num_classes,
        data2_name,
    ) = load_dataset(config, INPUTS_DIR_PATH)
    print(f"  HSI: {NC} bands, {data2_name}: {NC_LIDAR} bands, Classes: {num_classes}")
    print(f"  Train: {train_hsi.shape[0]}, Test: {test_hsi.shape[0]}\n", flush=True)

    # CSV setup
    csv_path = os.path.join(OUTPUTS_DIR_PATH, f"benchmark_{dataset_name}.csv")
    csv_fields = [
        "model_type",
        "run_idx",
        "OA",
        "AA",
        "Kappa",
        "total_params",
        "encoder_params",
        "total_FLOPs",
        "encoder_FLOPs",
        "best_epoch",
        "train_time_s",
        "depth",
        "d_state",
        "d_conv",
        "expand",
        "lr",
        "wd",
        "label_smooth",
        "warmup",
        "num_epochs",
        "config_tag",
        "notes",
    ]

    all_rows = []

    # =================================================================
    # Phase 1: MFT Transformer Baseline
    # =================================================================
    print(f"\n{'='*80}")
    print(
        f"[Phase 1] MFT Transformer Baseline ({NUM_RUNS} runs, {config.num_train_epochs} epochs)"
    )
    print(f"{'='*80}")
    sys.stdout.flush()

    baseline_results = []
    for run_idx in range(NUM_RUNS):
        print(f"\n  --- Baseline Run {run_idx+1}/{NUM_RUNS} ---")
        sys.stdout.flush()

        metrics = train_baseline_run(
            config,
            train_hsi,
            train_lidar,
            train_label,
            test_hsi,
            test_lidar,
            test_label,
            NC,
            NC_LIDAR,
            num_classes,
            run_idx,
            device,
        )
        baseline_results.append(metrics)

        print(
            f"    Params: {metrics['total_params']:,} | Encoder: {metrics['encoder_params']:,}"
        )
        print(
            f"    FLOPs: {metrics['total_flops']:,} | Encoder FLOPs: {metrics['encoder_flops']:,}"
        )
        print(
            f"    OA: {metrics['best_oa']:.2f}% | AA: {metrics['best_aa']:.2f}% | "
            f"Kappa: {metrics['best_kappa']:.2f}% | Best@ep{metrics['best_epoch']} | {metrics['train_time']:.0f}s"
        )
        sys.stdout.flush()

        row = {
            "model_type": "transformer",
            "run_idx": run_idx,
            "OA": f"{metrics['best_oa']:.4f}",
            "AA": f"{metrics['best_aa']:.4f}",
            "Kappa": f"{metrics['best_kappa']:.4f}",
            "total_params": metrics["total_params"],
            "encoder_params": metrics["encoder_params"],
            "total_FLOPs": metrics["total_flops"],
            "encoder_FLOPs": metrics["encoder_flops"],
            "best_epoch": metrics["best_epoch"],
            "train_time_s": f"{metrics['train_time']:.1f}",
            "depth": config.model.num_encoder_layers,
            "d_state": "-",
            "d_conv": "-",
            "expand": "-",
            "lr": config.optimizer.learning_rate,
            "wd": config.optimizer.weight_decay,
            "label_smooth": 0.0,
            "warmup": 0,
            "num_epochs": config.num_train_epochs,
            "config_tag": "mft_transformer_baseline",
            "notes": f"heads={config.model.num_heads},mlp_dim={config.model.mlp_hidden_dim},StepLR(step={config.scheduler.step_size},gamma={config.scheduler.gamma})",
        }
        all_rows.append(row)
        torch.cuda.empty_cache()

    # Baseline summary
    bl_oas = [r["best_oa"] for r in baseline_results]
    bl_aas = [r["best_aa"] for r in baseline_results]
    bl_kappas = [r["best_kappa"] for r in baseline_results]
    print(f"\n  Baseline Summary ({NUM_RUNS} runs):")
    print(f"    OA:    {np.mean(bl_oas):.2f} +/- {np.std(bl_oas):.2f}")
    print(f"    AA:    {np.mean(bl_aas):.2f} +/- {np.std(bl_aas):.2f}")
    print(f"    Kappa: {np.mean(bl_kappas):.2f} +/- {np.std(bl_kappas):.2f}")
    print(
        f"    Params: {baseline_results[0]['total_params']:,} | Encoder: {baseline_results[0]['encoder_params']:,}"
    )
    print(
        f"    FLOPs: {baseline_results[0]['total_flops']:,} | Encoder FLOPs: {baseline_results[0]['encoder_flops']:,}"
    )
    sys.stdout.flush()

    baseline_mean_oa = np.mean(bl_oas)
    baseline_std_oa = np.std(bl_oas)
    baseline_mean_aa = np.mean(bl_aas)
    baseline_std_aa = np.std(bl_aas)
    baseline_mean_kappa = np.mean(bl_kappas)
    baseline_std_kappa = np.std(bl_kappas)
    baseline_params = baseline_results[0]["total_params"]
    baseline_encoder_params = baseline_results[0]["encoder_params"]
    baseline_flops = baseline_results[0]["total_flops"]
    baseline_encoder_flops = baseline_results[0]["encoder_flops"]

    # =================================================================
    # Phase 2: Mamba Grid Search
    # =================================================================
    print(f"\n\n{'='*80}")
    print(
        f"[Phase 2] Mamba Grid Search ({total_configs} configs x {NUM_RUNS} runs, {MAMBA_EPOCHS} epochs)"
    )
    print(f"{'='*80}")
    sys.stdout.flush()

    mamba_config_results = []
    sweep_start = time.time()

    for idx, hp in enumerate(all_combos):
        tag = (
            f"d{hp['depth']}_s{hp['d_state']}_e{hp['expand']}"
            f"_lr{hp['lr']}_wd{hp['wd']}_ls{hp['label_smooth']}"
        )
        print(f"\n  [{idx+1}/{total_configs}] {tag}")
        sys.stdout.flush()

        run_results = []
        for run_idx in range(NUM_RUNS):
            try:
                metrics = train_mamba_run(
                    hp=hp,
                    FM=config.model.feature_maps,
                    NC=NC,
                    NC_LIDAR=NC_LIDAR,
                    num_classes=num_classes,
                    patch_size=patch_size,
                    ntokens=config.model.num_hsi_tokens,
                    train_hsi=train_hsi,
                    train_lidar=train_lidar,
                    train_label=train_label,
                    test_hsi=test_hsi,
                    test_lidar=test_lidar,
                    test_label=test_label,
                    num_epochs=MAMBA_EPOCHS,
                    batch_size=config.batch_size,
                    run_idx=run_idx,
                    device=device,
                )
                run_results.append(metrics)

                row = {
                    "model_type": "mamba",
                    "run_idx": run_idx,
                    "OA": f"{metrics['best_oa']:.4f}",
                    "AA": f"{metrics['best_aa']:.4f}",
                    "Kappa": f"{metrics['best_kappa']:.4f}",
                    "total_params": metrics["total_params"],
                    "encoder_params": metrics["encoder_params"],
                    "total_FLOPs": metrics["total_flops"],
                    "encoder_FLOPs": metrics["encoder_flops"],
                    "best_epoch": metrics["best_epoch"],
                    "train_time_s": f"{metrics['train_time']:.1f}",
                    "depth": hp["depth"],
                    "d_state": hp["d_state"],
                    "d_conv": hp["d_conv"],
                    "expand": hp["expand"],
                    "lr": hp["lr"],
                    "wd": hp["wd"],
                    "label_smooth": hp["label_smooth"],
                    "warmup": hp["warmup"],
                    "num_epochs": MAMBA_EPOCHS,
                    "config_tag": tag,
                    "notes": f"cosine_annealing,grad_clip=1.0,BiMamba,mean_pool",
                }
                all_rows.append(row)

            except Exception as e:
                print(f"    Run {run_idx+1} FAILED: {e}")
                row = {
                    "model_type": "mamba",
                    "run_idx": run_idx,
                    "OA": "0",
                    "AA": "0",
                    "Kappa": "0",
                    "total_params": 0,
                    "encoder_params": 0,
                    "total_FLOPs": 0,
                    "encoder_FLOPs": 0,
                    "best_epoch": 0,
                    "train_time_s": "0",
                    "depth": hp["depth"],
                    "d_state": hp["d_state"],
                    "d_conv": hp["d_conv"],
                    "expand": hp["expand"],
                    "lr": hp["lr"],
                    "wd": hp["wd"],
                    "label_smooth": hp["label_smooth"],
                    "warmup": hp["warmup"],
                    "num_epochs": MAMBA_EPOCHS,
                    "config_tag": tag,
                    "notes": f"ERROR: {e}",
                }
                all_rows.append(row)

            torch.cuda.empty_cache()

        if run_results:
            mean_oa = np.mean([r["best_oa"] for r in run_results])
            mean_aa = np.mean([r["best_aa"] for r in run_results])
            mean_kappa = np.mean([r["best_kappa"] for r in run_results])
            std_oa = np.std([r["best_oa"] for r in run_results])
            std_aa = np.std([r["best_aa"] for r in run_results])
            std_kappa = np.std([r["best_kappa"] for r in run_results])
            param_ratio = run_results[0]["total_params"] / baseline_params * 100

            mamba_config_results.append(
                {
                    **hp,
                    "tag": tag,
                    "mean_oa": mean_oa,
                    "std_oa": std_oa,
                    "std_aa": std_aa,
                    "std_kappa": std_kappa,
                    "mean_aa": mean_aa,
                    "mean_kappa": mean_kappa,
                    "total_params": run_results[0]["total_params"],
                    "encoder_params": run_results[0]["encoder_params"],
                    "total_flops": run_results[0]["total_flops"],
                    "encoder_flops": run_results[0]["encoder_flops"],
                    "runs": run_results,
                }
            )

            flops_ratio = run_results[0]["total_flops"] / baseline_flops * 100
            enc_flops_ratio = run_results[0]["encoder_flops"] / baseline_encoder_flops * 100

            print(
                f"    OA:    {mean_oa:.2f} +/- {std_oa:.2f}  (baseline: {baseline_mean_oa:.2f} +/- {baseline_std_oa:.2f})"
            )
            print(
                f"    AA:    {mean_aa:.2f} +/- {std_aa:.2f}  (baseline: {baseline_mean_aa:.2f} +/- {baseline_std_aa:.2f})"
            )
            print(
                f"    Kappa: {mean_kappa:.2f} +/- {std_kappa:.2f}  (baseline: {baseline_mean_kappa:.2f} +/- {baseline_std_kappa:.2f})"
            )
            print(
                f"    Params: {run_results[0]['total_params']:,} ({param_ratio:.0f}%) | "
                f"Encoder: {run_results[0]['encoder_params']:,}  "
                f"(baseline: {baseline_params:,} | {baseline_encoder_params:,})"
            )
            print(
                f"    FLOPs:  {run_results[0]['total_flops']:,} ({flops_ratio:.0f}%) | "
                f"Encoder: {run_results[0]['encoder_flops']:,} ({enc_flops_ratio:.0f}%)  "
                f"(baseline: {baseline_flops:,} | {baseline_encoder_flops:,})"
            )
        sys.stdout.flush()

    sweep_time = time.time() - sweep_start

    # =================================================================
    # Save CSV
    # =================================================================
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  CSV saved: {csv_path} ({len(all_rows)} rows)")

    # =================================================================
    # Save JSON
    # =================================================================
    json_data = {
        "dataset": dataset_name,
        "timestamp": training_start_str,
        "baseline": {
            "model": "MFT_Transformer",
            "num_runs": NUM_RUNS,
            "config": {
                "num_heads": config.model.num_heads,
                "mlp_dim": config.model.mlp_hidden_dim,
                "depth": config.model.num_encoder_layers,
                "lr": config.optimizer.learning_rate,
                "wd": config.optimizer.weight_decay,
                "epochs": config.num_train_epochs,
            },
            "mean_oa": float(np.mean(bl_oas)),
            "std_oa": float(np.std(bl_oas)),
            "mean_aa": float(np.mean(bl_aas)),
            "std_aa": float(np.std(bl_aas)),
            "mean_kappa": float(np.mean(bl_kappas)),
            "std_kappa": float(np.std(bl_kappas)),
            "total_params": baseline_results[0]["total_params"],
            "encoder_params": baseline_results[0]["encoder_params"],
            "total_flops": baseline_results[0]["total_flops"],
            "encoder_flops": baseline_results[0]["encoder_flops"],
            "runs": baseline_results,
        },
        "mamba_sweep": {
            "grid": {k: [str(v) for v in vals] for k, vals in SWEEP_GRID.items()},
            "num_configs": total_configs,
            "num_runs_per_config": NUM_RUNS,
            "mamba_epochs": MAMBA_EPOCHS,
            "sweep_time_s": sweep_time,
            "configs": [],
        },
    }

    mamba_config_results.sort(key=lambda r: r["mean_oa"], reverse=True)
    for r in mamba_config_results:
        json_data["mamba_sweep"]["configs"].append(
            {
                "tag": r["tag"],
                "depth": r["depth"],
                "d_state": r["d_state"],
                "d_conv": r["d_conv"],
                "expand": r["expand"],
                "lr": r["lr"],
                "wd": r["wd"],
                "label_smooth": r["label_smooth"],
                "warmup": r["warmup"],
                "mean_oa": r["mean_oa"],
                "std_oa": r["std_oa"],
                "mean_aa": r["mean_aa"],
                "std_aa": r["std_aa"],
                "mean_kappa": r["mean_kappa"],
                "std_kappa": r["std_kappa"],
                "total_params": r["total_params"],
                "encoder_params": r["encoder_params"],
                "total_flops": r["total_flops"],
                "encoder_flops": r["encoder_flops"],
                "runs": r["runs"],
            }
        )

    json_path = os.path.join(OUTPUTS_DIR_PATH, f"benchmark_{dataset_name}.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"  JSON saved: {json_path}")

    # =================================================================
    # Summary
    # =================================================================
    print(f"\n\n{'='*80}")
    print(f"[BENCHMARK RESULTS] {dataset_name}")
    print(f"{'='*80}")
    print(f"\n  MFT Transformer Baseline ({NUM_RUNS} runs):")
    print(f"    OA:    {np.mean(bl_oas):.2f} +/- {np.std(bl_oas):.2f}%")
    print(f"    AA:    {np.mean(bl_aas):.2f} +/- {np.std(bl_aas):.2f}%")
    print(f"    Kappa: {np.mean(bl_kappas):.2f} +/- {np.std(bl_kappas):.2f}%")
    print(
        f"    Params: {baseline_params:,} | Encoder: {baseline_results[0]['encoder_params']:,}"
    )
    print(
        f"    FLOPs:  {baseline_results[0]['total_flops']:,} | Encoder: {baseline_results[0]['encoder_flops']:,}"
    )

    print(f"\n  Top 10 Mamba Configs (by mean OA, {NUM_RUNS} runs each):")
    print(
        f"  {'Rank':<5} {'OA+/-std':<14} {'AA+/-std':<14} {'Kap+/-std':<14} {'Params':<10} {'%BL':<6} "
        f"{'FLOPs':<12} {'%BL':<6} {'Dep':<4} {'d_s':<4} {'Exp':<4} {'LR':<9} {'WD':<8} {'LS':<5}"
    )
    print(f"  {'-'*145}")

    for rank, r in enumerate(mamba_config_results[:10], 1):
        param_pct = r["total_params"] / baseline_params * 100
        flops_pct = r["total_flops"] / baseline_flops * 100
        print(
            f"  {rank:<5} {r['mean_oa']:.2f}+/-{r['std_oa']:.2f}{'':>3} "
            f"{r['mean_aa']:.2f}+/-{r['std_aa']:.2f}{'':>3} "
            f"{r['mean_kappa']:.2f}+/-{r['std_kappa']:.2f}{'':>1} "
            f"{r['total_params']:<10,} {param_pct:<6.0f} "
            f"{r['total_flops']:<12,} {flops_pct:<6.0f} {r['depth']:<4} {r['d_state']:<4} {r['expand']:<4} "
            f"{r['lr']:<9.1e} {r['wd']:<8.1e} {r['label_smooth']:<5.1f}"
        )

    if mamba_config_results:
        best = mamba_config_results[0]
        param_pct = best["total_params"] / baseline_params * 100
        print(f"\n  Best Mamba vs Baseline:")
        best_flops_pct = best["total_flops"] / baseline_flops * 100
        print(
            f"    Baseline:   OA={baseline_mean_oa:.2f}% | {baseline_params:,} params | {baseline_flops:,} FLOPs"
        )
        print(
            f"    Best Mamba: OA={best['mean_oa']:.2f}+/-{best['std_oa']:.2f}% | "
            f"{best['total_params']:,} params ({param_pct:.0f}%) | "
            f"{best['total_flops']:,} FLOPs ({best_flops_pct:.0f}%)"
        )
        print(f"    OA gap: {best['mean_oa'] - baseline_mean_oa:+.2f}%")

        smaller = [
            r for r in mamba_config_results if r["total_params"] < baseline_params
        ]
        if smaller:
            bs = smaller[0]
            ps = bs["total_params"] / baseline_params * 100
            bs_flops_pct = bs["total_flops"] / baseline_flops * 100
            print(
                f"    Best Mamba (<baseline params): OA={bs['mean_oa']:.2f}+/-{bs['std_oa']:.2f}% | "
                f"{bs['total_params']:,} params ({ps:.0f}%) | "
                f"{bs['total_flops']:,} FLOPs ({bs_flops_pct:.0f}%) | gap: {bs['mean_oa'] - baseline_mean_oa:+.2f}%"
            )

    print(f"\n  Sweep time: {sweep_time:.0f}s ({sweep_time/3600:.1f}h)")
    print(f"  Results: {csv_path}")
    print(f"           {json_path}")
    print(f"{'='*80}\n", flush=True)
    
    return 
