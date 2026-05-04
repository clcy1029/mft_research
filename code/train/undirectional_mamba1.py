import os
import sys
import time
import json
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.utils.data as dataf
import torch.backends.cudnn as cudnn
from torch.nn import LayerNorm
from einops import rearrange
from scipy import io
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from operator import truediv
from pathlib import Path
from datetime import datetime
import pytz
import statistics


from mamba_ssm.modules.mamba_simple import Mamba

from libs.config.schema import ExperimentConfig
from libs.utils import gpu_mem

def train_function(params: dict):
    """
    Unidirectional Mamba-1 experiment (NOT bidirectional).
    """

    cudnn.deterministic = True
    cudnn.benchmark = False

    pst = pytz.timezone("America/Los_Angeles")
    training_start_time = datetime.now(pst)
    training_start_str = training_start_time.strftime("%Y-%m-%d_%H:%M")

    # =================================================================
    # LOAD CONFIG
    # =================================================================
    config_path = params.get("config_path")
    if not config_path:
        raise ValueError("Must provide 'config_path' in params")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    config = ExperimentConfig(**config_dict)

    # Multi-dataset support
    dataset_list = config_dict.get("datasets", None)
    if dataset_list is None:
        dataset_list = [{"dataset_name": config_dict["data"]["dataset_name"],
                         "num_lidar_bands": config_dict["data"]["num_lidar_bands"]}]

    SEEDS = [42, 123, 456]

    device_str = config.device.device
    if device_str == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    CURRENT_DIR_PATH = os.getcwd()
    INPUTS_DIR_PATH = os.path.join(CURRENT_DIR_PATH, "inputs")
    RUN_NAME = f"unidirectional_mamba1_{training_start_str}"
    OUTPUTS_DIR_PATH = os.path.join(CURRENT_DIR_PATH, "outputs", RUN_NAME)
    os.makedirs(INPUTS_DIR_PATH, exist_ok=True)
    os.makedirs(OUTPUTS_DIR_PATH, exist_ok=True)

    # =================================================================
    # Model classes
    # =================================================================
    class HetConv(nn.Module):
        def __init__(self, in_channels, out_channels, p=64, g=64):
            super().__init__()
            self.gwc = nn.Conv2d(in_channels, out_channels, kernel_size=3, groups=g, padding=1)
            self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=p)
        def forward(self, x):
            return self.gwc(x) + self.pwc(x)

    class MambaEncoderBlock(nn.Module):
        """Unidirectional Mamba-1 block: single forward scan only."""
        def __init__(self, dim, d_state, d_conv, expand):
            super().__init__()
            self.norm = LayerNorm(dim, eps=1e-6)
            self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        def forward(self, x):
            return x + self.mamba(self.norm(x))

    class MambaEncoder(nn.Module):
        def __init__(self, dim, depth, d_state, d_conv, expand):
            super().__init__()
            self.layers = nn.ModuleList([
                MambaEncoderBlock(dim, d_state, d_conv, expand) for _ in range(depth)
            ])
            self.norm = LayerNorm(dim, eps=1e-6)
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return self.norm(x).mean(dim=1)

    class MFTMambaAblation(nn.Module):
        """MFTMamba with optional position embedding."""
        def __init__(self, FM, NC, NCLidar, Classes, patch_size, ntokens,
                     depth, d_state, d_conv, expand, use_pos_embed=True):
            super().__init__()
            self.ntokens = ntokens
            self.FM = FM
            self.patch_size = patch_size
            self.use_pos_embed = use_pos_embed

            self.conv5 = nn.Sequential(
                nn.Conv3d(1, 8, (9, 3, 3), padding=(0, 1, 1), stride=1),
                nn.BatchNorm3d(8), nn.ReLU())
            self.conv6 = nn.Sequential(
                HetConv(8 * (NC - 8), FM * 4, p=1,
                        g=(FM * 4) // 4 if (8 * (NC - 8)) % FM == 0 else (FM * 4) // 8),
                nn.BatchNorm2d(FM * 4), nn.ReLU())
            self.lidarConv = nn.Sequential(
                nn.Conv2d(NCLidar, FM * 4, 3, 1, 1),
                nn.BatchNorm2d(FM * 4), nn.GELU())

            self.ca = MambaEncoder(FM * 4, depth, d_state, d_conv, expand)
            self.out3 = nn.Linear(FM * 4, Classes)

            if use_pos_embed:
                self.position_embeddings = nn.Parameter(torch.randn(1, ntokens + 1, FM * 4))
            self.dropout = nn.Dropout(0.1)

            torch.nn.init.xavier_uniform_(self.out3.weight)
            torch.nn.init.normal_(self.out3.bias, std=1e-6)

            # HSI tokenizer
            self.token_wA = nn.Parameter(torch.empty(1, ntokens, FM * 4), requires_grad=True)
            torch.nn.init.xavier_normal_(self.token_wA)
            self.token_wV = nn.Parameter(torch.empty(1, FM * 4, FM * 4), requires_grad=True)
            torch.nn.init.xavier_normal_(self.token_wV)

            # LiDAR tokenizer (channel mode)
            self.token_wA_L = nn.Parameter(torch.empty(1, 1, FM * 4), requires_grad=True)
            torch.nn.init.xavier_normal_(self.token_wA_L)
            self.token_wV_L = nn.Parameter(torch.empty(1, FM * 4, FM * 4), requires_grad=True)
            torch.nn.init.xavier_normal_(self.token_wV_L)

        def forward(self, x1, x2):
            ps = self.patch_size
            x1 = x1.reshape(x1.shape[0], -1, ps, ps).unsqueeze(1)
            x1 = self.conv5(x1)
            x1 = x1.reshape(x1.shape[0], -1, ps, ps)
            x1 = self.conv6(x1)
            x1 = x1.flatten(2).transpose(-1, -2)

            wa = rearrange(self.token_wA.expand(x1.shape[0], -1, -1), 'b h w -> b w h')
            A = rearrange(torch.einsum('bij,bjk->bik', x1, wa), 'b h w -> b w h').softmax(dim=-1)
            VV = torch.einsum('bij,bjk->bik', x1, self.token_wV.expand(x1.shape[0], -1, -1))
            T = torch.einsum('bij,bjk->bik', A, VV)

            x2 = x2.reshape(x2.shape[0], -1, ps, ps)
            x2 = self.lidarConv(x2).reshape(x2.shape[0], -1, ps**2).transpose(-1, -2)

            wa_L = rearrange(self.token_wA_L.expand(x2.shape[0], -1, -1), 'b h w -> b w h')
            A_L = rearrange(torch.einsum('bij,bjk->bik', x2, wa_L), 'b h w -> b w h').softmax(dim=-1)
            VV_L = torch.einsum('bij,bjk->bik', x2, self.token_wV_L.expand(x2.shape[0], -1, -1))
            L = torch.einsum('bij,bjk->bik', A_L, VV_L)

            x = torch.cat((L, T), dim=1)
            if self.use_pos_embed:
                x = x + self.position_embeddings
            x = self.dropout(x)
            x = self.ca(x).reshape(x.shape[0], -1)
            return self.out3(x)

    # =================================================================
    # Transformer Baseline Model (MFT original)
    # =================================================================
    class MCrossAttention(nn.Module):
        def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                     attn_drop=0.1, proj_drop=0.1):
            super().__init__()
            self.num_heads = num_heads
            head_dim = dim // num_heads
            self.scale = qk_scale or head_dim ** -0.5
            self.wq = nn.Linear(head_dim, dim, bias=qkv_bias)
            self.wk = nn.Linear(head_dim, dim, bias=qkv_bias)
            self.wv = nn.Linear(head_dim, dim, bias=qkv_bias)
            self.proj = nn.Linear(dim * num_heads, dim)
            self.proj_drop = nn.Dropout(proj_drop)

        def forward(self, x):
            B, N, C = x.shape
            q = self.wq(x[:, 0:1, ...].reshape(B, 1, self.num_heads, C // self.num_heads)).permute(0, 2, 1, 3)
            k = self.wk(x.reshape(B, N, self.num_heads, C // self.num_heads)).permute(0, 2, 1, 3)
            v = self.wv(x.reshape(B, N, self.num_heads, C // self.num_heads)).permute(0, 2, 1, 3)
            attn = (torch.einsum("bhid,bhjd->bhij", q, k) * self.scale).softmax(dim=-1)
            x = torch.einsum("bhij,bhjd->bhid", attn, v).transpose(1, 2).reshape(B, 1, C * self.num_heads)
            return self.proj_drop(self.proj(x))

    class Mlp(nn.Module):
        def __init__(self, dim, mlp_dim):
            super().__init__()
            self.fc1 = nn.Linear(dim, mlp_dim)
            self.fc2 = nn.Linear(mlp_dim, dim)
            self.act_fn = nn.GELU()
            self.dropout = nn.Dropout(0.1)
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.normal_(self.fc1.bias, std=1e-6)
            nn.init.normal_(self.fc2.bias, std=1e-6)

        def forward(self, x):
            return self.dropout(self.fc2(self.dropout(self.act_fn(self.fc1(x)))))

    class TransformerBlock(nn.Module):
        def __init__(self, dim, num_heads, mlp_dim):
            super().__init__()
            self.attention_norm = LayerNorm(dim, eps=1e-6)
            self.ffn_norm = LayerNorm(dim, eps=1e-6)
            self.ffn = Mlp(dim, mlp_dim)
            self.attn = MCrossAttention(dim, num_heads)

        def forward(self, x):
            h = x
            x = self.attn(self.attention_norm(x)) + h
            h = x
            x = self.ffn(self.ffn_norm(x)) + h
            return x

    class TransformerEncoder(nn.Module):
        def __init__(self, dim, num_heads=8, mlp_dim=512, depth=2):
            super().__init__()
            import copy
            self.layer = nn.ModuleList([copy.deepcopy(TransformerBlock(dim, num_heads, mlp_dim)) for _ in range(depth)])
            self.encoder_norm = LayerNorm(dim, eps=1e-6)

        def forward(self, x):
            for layer_block in self.layer:
                x = layer_block(x)
            return self.encoder_norm(x)[:, 0]

    class MFTBaseline(nn.Module):
        """Original MFT Transformer baseline (from paper)."""
        def __init__(self, FM, NC, NCLidar, Classes, patch_size, ntokens,
                     num_heads=8, mlp_dim=512, depth=2):
            super().__init__()
            self.ntokens = ntokens
            self.FM = FM
            self.patch_size = patch_size

            self.conv5 = nn.Sequential(
                nn.Conv3d(1, 8, (9, 3, 3), padding=(0, 1, 1), stride=1),
                nn.BatchNorm3d(8), nn.ReLU())
            self.conv6 = nn.Sequential(
                HetConv(8 * (NC - 8), FM * 4, p=1,
                        g=(FM * 4) // 4 if (8 * (NC - 8)) % FM == 0 else (FM * 4) // 8),
                nn.BatchNorm2d(FM * 4), nn.ReLU())
            self.lidarConv = nn.Sequential(
                nn.Conv2d(NCLidar, FM * 4, 3, 1, 1),
                nn.BatchNorm2d(FM * 4), nn.GELU())

            self.ca = TransformerEncoder(FM * 4, num_heads, mlp_dim, depth)
            self.out3 = nn.Linear(FM * 4, Classes)
            self.position_embeddings = nn.Parameter(torch.randn(1, ntokens + 1, FM * 4))
            self.dropout = nn.Dropout(0.1)

            torch.nn.init.xavier_uniform_(self.out3.weight)
            torch.nn.init.normal_(self.out3.bias, std=1e-6)

            # Channel tokenizer (same as Mamba version)
            self.token_wA = nn.Parameter(torch.empty(1, ntokens, FM * 4), requires_grad=True)
            torch.nn.init.xavier_normal_(self.token_wA)
            self.token_wV = nn.Parameter(torch.empty(1, FM * 4, FM * 4), requires_grad=True)
            torch.nn.init.xavier_normal_(self.token_wV)
            self.token_wA_L = nn.Parameter(torch.empty(1, 1, FM * 4), requires_grad=True)
            torch.nn.init.xavier_normal_(self.token_wA_L)
            self.token_wV_L = nn.Parameter(torch.empty(1, FM * 4, FM * 4), requires_grad=True)
            torch.nn.init.xavier_normal_(self.token_wV_L)

        def forward(self, x1, x2):
            ps = self.patch_size
            x1 = x1.reshape(x1.shape[0], -1, ps, ps).unsqueeze(1)
            x1 = self.conv5(x1)
            x1 = x1.reshape(x1.shape[0], -1, ps, ps)
            x1 = self.conv6(x1)
            x1 = x1.flatten(2).transpose(-1, -2)

            wa = rearrange(self.token_wA.expand(x1.shape[0], -1, -1), 'b h w -> b w h')
            A = rearrange(torch.einsum('bij,bjk->bik', x1, wa), 'b h w -> b w h').softmax(dim=-1)
            VV = torch.einsum('bij,bjk->bik', x1, self.token_wV.expand(x1.shape[0], -1, -1))
            T = torch.einsum('bij,bjk->bik', A, VV)

            x2 = x2.reshape(x2.shape[0], -1, ps, ps)
            x2 = self.lidarConv(x2).reshape(x2.shape[0], -1, ps**2).transpose(-1, -2)

            wa_L = rearrange(self.token_wA_L.expand(x2.shape[0], -1, -1), 'b h w -> b w h')
            A_L = rearrange(torch.einsum('bij,bjk->bik', x2, wa_L), 'b h w -> b w h').softmax(dim=-1)
            VV_L = torch.einsum('bij,bjk->bik', x2, self.token_wV_L.expand(x2.shape[0], -1, -1))
            L = torch.einsum('bij,bjk->bik', A_L, VV_L)

            x = torch.cat((L, T), dim=1)
            x = x + self.position_embeddings
            x = self.dropout(x)
            x = self.ca(x).reshape(x.shape[0], -1)
            return self.out3(x)

    
    # =================================================================
    # Helpers (same as mft_mamba.py)
    # =================================================================
    DATASETS_WITH_HSI_PARTS = ['Berlin', 'Augsburg']
    DATA2_LIST = ['SAR', 'DSM', 'MS']

    def aa_and_each_class_accuracy(conf_matrix):
        list_diag = np.diag(conf_matrix)
        list_raw_sum = np.sum(conf_matrix, axis=1)
        each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
        return each_acc, np.mean(each_acc)

    def evaluate(model, x_hsi, x_lidar, y_true, batch_size=256, device="cuda"):
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

    def load_dataset(config, data_dir):
        dataset_name = config.data.dataset_name.value
        patch_size = config.data.patch_size
        data1_name, data2_name = '', ''
        if dataset_name in ["Houston", "Trento", "MUUFL"]:
            data1_name = dataset_name
            data2_name = "LIDAR"
        else:
            for d2 in DATA2_LIST:
                if d2 in dataset_name:
                    data1_name = dataset_name.replace(d2, "")
                    data2_name = d2
                    break
        base = os.path.join(data_dir, f"{data1_name}{patch_size}x{patch_size}")
        hsi_tr = io.loadmat(os.path.join(base, 'HSI_Tr.mat'))['Data'].astype(np.float32)
        lidar_tr = io.loadmat(os.path.join(base, f'{data2_name}_Tr.mat'))['Data'].astype(np.float32)
        tr_label = io.loadmat(os.path.join(base, 'TrLabel.mat'))['Data']
        if data1_name in DATASETS_WITH_HSI_PARTS:
            i = 2
            bp = os.path.join(base, 'HSI_Te_Part')
            hsi_te = io.loadmat(f'{bp}1.mat')['Data']
            while Path(f'{bp}{i}.mat').exists():
                hsi_te = np.concatenate([hsi_te, io.loadmat(f'{bp}{i}.mat')['Data']], axis=0)
                i += 1
        else:
            hsi_te = io.loadmat(os.path.join(base, 'HSI_Te.mat'))['Data']
        hsi_te = hsi_te.astype(np.float32)
        lidar_te = io.loadmat(os.path.join(base, f'{data2_name}_Te.mat'))['Data'].astype(np.float32)
        te_label = io.loadmat(os.path.join(base, 'TeLabel.mat'))['Data']
        NC = hsi_tr.shape[3]
        NC_LIDAR = lidar_tr.shape[3]
        def to_tensors(hsi, lidar, label):
            h = torch.from_numpy(hsi).float().permute(0,3,1,2).reshape(hsi.shape[0], hsi.shape[3], -1)
            l = torch.from_numpy(lidar).float().permute(0,3,1,2).reshape(lidar.shape[0], lidar.shape[3], -1)
            y = torch.from_numpy(label).long().reshape(-1) - 1
            return h, l, y
        train_hsi, train_lidar, train_label = to_tensors(hsi_tr, lidar_tr, tr_label)
        test_hsi, test_lidar, test_label = to_tensors(hsi_te, lidar_te, te_label)
        num_classes = len(np.unique(train_label.numpy()))
        return (train_hsi, train_lidar, train_label,
                test_hsi, test_lidar, test_label, NC, NC_LIDAR, num_classes)

    def count_params(model):
        """Detailed parameter breakdown."""
        groups = {
            'conv5': 0, 'conv6': 0, 'lidarConv': 0,
            'tokenization': 0, 'pos_embed': 0,
            'encoder': 0, 'classifier': 0
        }
        for name, p in model.named_parameters():
            n = p.numel()
            if 'conv5' in name or 'batch_norm5' in name:
                groups['conv5'] += n
            elif 'conv6' in name or 'batch_norm6' in name:
                groups['conv6'] += n
            elif 'lidarConv' in name:
                groups['lidarConv'] += n
            elif 'token_w' in name:
                groups['tokenization'] += n
            elif 'position_embeddings' in name:
                groups['pos_embed'] += n
            elif 'ca.' in name:
                groups['encoder'] += n
            elif 'out3' in name:
                groups['classifier'] += n
            else:
                groups['encoder'] += n  # default
        groups['cnn'] = groups['conv5'] + groups['conv6'] + groups['lidarConv']
        groups['total'] = sum(p.numel() for p in model.parameters())
        return groups

    # =================================================================
    # Training loop: Baseline Transformer (original MFT recipe)
    # lr=5e-4, StepLR(step=50, gamma=0.9), 200 epochs, no label smooth
    # =================================================================
    def train_one_run_baseline(model, train_hsi, train_lidar, train_label,
                               test_hsi, test_lidar, test_label, device):
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-3)
        loss_fn = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

        dataset = dataf.TensorDataset(train_hsi, train_lidar, train_label)
        loader = dataf.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

        best_oa = 0.0
        best_state = None
        best_epoch = 0
        start = time.time()

        for epoch in range(200):
            model.train()
            for bx1, bx2, by in loader:
                bx1, bx2, by = bx1.to(device), bx2.to(device), by.to(device)
                loss = loss_fn(model(bx1, bx2), by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            if (epoch + 1) % 25 == 0 or epoch == 199:
                oa, aa, kappa, each_acc = evaluate(model, test_hsi, test_lidar, test_label,
                                                    batch_size=256, device=device)
                if oa > best_oa:
                    best_oa = oa
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = epoch + 1

        train_time = time.time() - start
        model.load_state_dict(best_state)
        model.to(device)
        oa, aa, kappa, each_acc = evaluate(model, test_hsi, test_lidar, test_label,
                                            batch_size=256, device=device)
        return {
            'oa': round(oa, 4), 'aa': round(aa, 4), 'kappa': round(kappa, 4),
            'each_acc': [round(float(x), 4) for x in each_acc],
            'best_epoch': best_epoch, 'time_s': round(train_time, 1),
            'best_state': best_state
        }

    # =================================================================
    # Training loop: Mamba (with cosine+warmup+grad_clip)
    # =================================================================
    def train_one_run_mamba(model, train_hsi, train_lidar, train_label,
                      test_hsi, test_lidar, test_label, device):
        optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.learning_rate, weight_decay=config.optimizer.weight_decay)

        if 0.1 > 0:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        else:
            loss_fn = nn.CrossEntropyLoss()

        # Cosine annealing with warmup
        warmup_epochs = 10
        total_epochs = config.num_train_epochs

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        dataset = dataf.TensorDataset(train_hsi, train_lidar, train_label)
        loader = dataf.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

        best_oa = 0.0
        best_state = None
        best_epoch = 0
        start = time.time()

        for epoch in range(total_epochs):
            model.train()
            for bx1, bx2, by in loader:
                bx1, bx2, by = bx1.to(device), bx2.to(device), by.to(device)
                out = model(bx1, bx2)
                loss = loss_fn(out, by)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            # Eval every 25 epochs
            if (epoch + 1) % 25 == 0 or epoch == total_epochs - 1:
                oa, aa, kappa, each_acc = evaluate(model, test_hsi, test_lidar, test_label,
                                                    batch_size=256, device=device)
                if oa > best_oa:
                    best_oa = oa
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = epoch + 1

        train_time = time.time() - start

        # Restore best and final eval
        model.load_state_dict(best_state)
        model.to(device)
        oa, aa, kappa, each_acc = evaluate(model, test_hsi, test_lidar, test_label,
                                            batch_size=256, device=device)
        return {
            'oa': round(oa, 4), 'aa': round(aa, 4), 'kappa': round(kappa, 4),
            'each_acc': [round(float(x), 4) for x in each_acc],
            'best_epoch': best_epoch, 'time_s': round(train_time, 1),
            'best_state': best_state
        }

    # =================================================================
    # DEVICE
    # =================================================================
    print(f"\n{'='*80}")
    print(f"[Setup] Unidirectional Mamba-1 Experiment")
    print(f"  Config: {config_path}")
    print(f"  Datasets: {[d['dataset_name'] for d in dataset_list]}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Outputs: {OUTPUTS_DIR_PATH}")
    print(f"{'='*80}\n", flush=True)

    FM = config.model.feature_maps
    ntokens = config.model.num_hsi_tokens
    test_bs = config.evaluation.test_batch_size

    # =================================================================
    # MAIN LOOP
    # =================================================================
    all_results = {
        'experiment': 'unidirectional_mamba1',
        'timestamp': training_start_str,
        'config': config_path,
        'seeds': SEEDS,
        'datasets': {}
    }

    for ds_idx, ds_entry in enumerate(dataset_list):
        dataset_name = ds_entry["dataset_name"]
        num_lidar_bands = ds_entry["num_lidar_bands"]

        # Override config for this dataset
        config_dict["data"]["dataset_name"] = dataset_name
        config_dict["data"]["num_lidar_bands"] = num_lidar_bands
        config = ExperimentConfig(**config_dict)

        print(f"\n{'='*70}")
        print(f"DATASET {ds_idx+1}/{len(dataset_list)}: {dataset_name}")
        print(f"{'='*70}")
        sys.stdout.flush()

        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        np.random.seed(config.seed)

        # Load data
        (train_hsi, train_lidar, train_label,
         test_hsi, test_lidar, test_label,
         NC, NC_LIDAR, num_classes) = load_dataset(config, INPUTS_DIR_PATH)

        print(f"  Train: {train_hsi.shape[0]}, Test: {test_hsi.shape[0]}, "
              f"NC={NC}, NC_LIDAR={NC_LIDAR}, Classes={num_classes}")

        # Move to CPU to save GPU memory
        test_hsi = test_hsi.cpu()
        test_lidar = test_lidar.cpu()
        train_hsi = train_hsi.cpu()
        train_lidar = train_lidar.cpu()

        ds_result = {
            'num_classes': num_classes, 'NC': NC, 'NC_LIDAR': NC_LIDAR,
            'train_samples': train_hsi.shape[0], 'test_samples': test_hsi.shape[0],
        }
        baseline_total_params = None  # will be set after first baseline run

        for condition in ['baseline', 'mamba1']:
            if condition == 'baseline':
                pe_label = "Transformer (baseline)"
            else:
                pe_label = "Unidirectional Mamba-1 (no PE)"
            print(f"\n  --- Condition: {pe_label} ---")
            sys.stdout.flush()

            runs = []
            param_info = None

            for run_idx, seed in enumerate(SEEDS):
                print(f"    Run {run_idx+1}/3 (seed={seed}) ...", end=" ", flush=True)

                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                if condition == 'baseline':
                    model = MFTBaseline(
                        FM=config.model.feature_maps, NC=NC, NCLidar=NC_LIDAR, Classes=num_classes,
                        patch_size=config.data.patch_size, ntokens=config.model.num_hsi_tokens,
                        num_heads=8, mlp_dim=512, depth=2
                    ).to(device)
                else:
                    model = MFTMambaAblation(
                        FM=config.model.feature_maps, NC=NC, NCLidar=NC_LIDAR, Classes=num_classes,
                        patch_size=config.data.patch_size, ntokens=config.model.num_hsi_tokens,
                        depth=config.model.num_encoder_layers, d_state=config.mamba.d_state,
                        d_conv=config.mamba.d_conv, expand=config.mamba.expand,
                        use_pos_embed=False
                    ).to(device)

                if param_info is None:
                    param_info = count_params(model)
                    if condition == 'baseline':
                        baseline_total_params = param_info['total']
                    pct_str = ""
                    if baseline_total_params and baseline_total_params > 0:
                        pct_str = f" ({param_info['total'] / baseline_total_params * 100:.1f}%)"
                    print(f"\n      Params: total={param_info['total']:,}{pct_str} "
                          f"(encoder={param_info['encoder']:,}, cnn={param_info['cnn']:,}, "
                          f"tok={param_info['tokenization']:,}, pe={param_info['pos_embed']:,}, "
                          f"cls={param_info['classifier']:,})")
                    print(f"    Run 1/3 (seed={seed}) ...", end=" ", flush=True)

                if condition == 'baseline':
                    result = train_one_run_baseline(model, train_hsi, train_lidar, train_label,
                                                    test_hsi, test_lidar, test_label, device)
                else:
                    result = train_one_run_mamba(model, train_hsi, train_lidar, train_label,
                                                 test_hsi, test_lidar, test_label, device)
                runs.append(result)
                print(f"OA={result['oa']:.2f}, AA={result['aa']:.2f}, "
                      f"K={result['kappa']:.2f}, epoch={result['best_epoch']}, "
                      f"time={result['time_s']:.0f}s")
                sys.stdout.flush()

                # Save best checkpoint
                ckpt_dir = os.path.join(OUTPUTS_DIR_PATH, "checkpoints", dataset_name.lower())
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"{condition}_run{run_idx}.pt")
                torch.save(result['best_state'], ckpt_path)
                print(f"      Checkpoint saved: {ckpt_path}")

                # Save predictions for run0 (seed=42) — used for classification maps
                if run_idx == 0:
                    model_tmp = model.__class__.__new__(model.__class__)
                    # Re-instantiate to load best state for prediction
                    if condition == 'baseline':
                        model_pred = MFTBaseline(
                            FM=config.model.feature_maps, NC=NC, NCLidar=NC_LIDAR, Classes=num_classes,
                            patch_size=config.data.patch_size, ntokens=config.model.num_hsi_tokens,
                            num_heads=8, mlp_dim=512, depth=2
                        ).to(device)
                    else:
                        model_pred = MFTMambaAblation(
                            FM=config.model.feature_maps, NC=NC, NCLidar=NC_LIDAR, Classes=num_classes,
                            patch_size=config.data.patch_size, ntokens=config.model.num_hsi_tokens,
                            depth=config.model.num_encoder_layers, d_state=config.mamba.d_state,
                            d_conv=config.mamba.d_conv, expand=config.mamba.expand,
                            use_pos_embed=False
                        ).to(device)
                    model_pred.load_state_dict(result['best_state'])
                    model_pred.eval()
                    # Get predictions
                    all_preds = np.empty(len(test_label), dtype=np.int64)
                    nb = len(test_label) // 256
                    with torch.no_grad():
                        for bi in range(nb):
                            s, e = bi*256, (bi+1)*256
                            all_preds[s:e] = torch.max(model_pred(test_hsi[s:e].to(device), test_lidar[s:e].to(device)), 1)[1].cpu().numpy()
                        if nb*256 < len(test_label):
                            s = nb*256
                            all_preds[s:] = torch.max(model_pred(test_hsi[s:].to(device), test_lidar[s:].to(device)), 1)[1].cpu().numpy()
                    pred_dir = os.path.join(OUTPUTS_DIR_PATH, "predictions", dataset_name.lower())
                    os.makedirs(pred_dir, exist_ok=True)
                    np.save(os.path.join(pred_dir, f"{condition}_preds.npy"), all_preds)
                    np.save(os.path.join(pred_dir, "test_gt.npy"), test_label.numpy())
                    print(f"      Predictions saved: {pred_dir}/{condition}_preds.npy")
                    del model_pred; torch.cuda.empty_cache()

                del result['best_state']  # free memory after saving

                # Free GPU
                del model
                torch.cuda.empty_cache()

            # Aggregate
            oas = [r['oa'] for r in runs]
            aas = [r['aa'] for r in runs]
            kappas = [r['kappa'] for r in runs]
            all_each = np.array([r['each_acc'] for r in runs])

            ds_result[condition] = {
                'runs': runs,
                'mean_oa': round(np.mean(oas), 4),
                'std_oa': round(np.std(oas), 4),
                'mean_aa': round(np.mean(aas), 4),
                'std_aa': round(np.std(aas), 4),
                'mean_kappa': round(np.mean(kappas), 4),
                'std_kappa': round(np.std(kappas), 4),
                'mean_each_acc': [round(float(x), 4) for x in np.mean(all_each, axis=0)],
                'std_each_acc': [round(float(x), 4) for x in np.std(all_each, axis=0)],
                'params': param_info,
            }

            print(f"    Summary {pe_label}: OA={np.mean(oas):.2f}±{np.std(oas):.2f}, "
                  f"AA={np.mean(aas):.2f}±{np.std(aas):.2f}, "
                  f"Kappa={np.mean(kappas):.2f}±{np.std(kappas):.2f}")
            sys.stdout.flush()

        all_results['datasets'][dataset_name] = ds_result

        # Aggressive GPU cleanup between datasets
        del train_hsi, train_lidar, train_label, test_hsi, test_lidar, test_label
        torch.cuda.empty_cache()
        import gc; gc.collect()
        print(f"\n  GPU memory cleared.")

    # =================================================================
    # SAVE RESULTS
    # =================================================================
    json_path = os.path.join(OUTPUTS_DIR_PATH, 'unidirectional_mamba1.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nJSON saved: {json_path}")

    # Human-readable summary
    summary_path = os.path.join(OUTPUTS_DIR_PATH, 'unidirectional_mamba1_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"=== Position Embedding Ablation Study ===\n")
        f.write(f"Timestamp: {training_start_str}\n")
        f.write(f"Config: depth={config.model.num_encoder_layers}, d_state={config.mamba.d_state}, "
                f"expand={config.mamba.expand}, lr={config.optimizer.learning_rate}, wd={config.optimizer.weight_decay}, "
                f"ls={0.1}\n")
        f.write(f"Epochs: {config.num_train_epochs}, Warmup: {10}, "
                f"Seeds: {SEEDS}\n\n")

        # Main table
        f.write(f"{'Dataset':<15} {'Model':<14} {'OA ± std':>16} {'AA ± std':>16} "
                f"{'Kappa ± std':>16} {'Params':>10} {'%BL':>6}\n")
        f.write("-" * 95 + "\n")

        for ds_entry in dataset_list:
            dataset_name = ds_entry['dataset_name']
            ds = all_results['datasets'][dataset_name]
            bl_params = ds['baseline']['params']['total'] if 'baseline' in ds else 1
            for cond in ['baseline', 'mamba1']:
                d = ds[cond]
                if cond == 'baseline':
                    label = "Transformer"
                else:
                    label = "Mamba-1 (uni)"
                pct = d['params']['total'] / bl_params * 100
                f.write(f"{dataset_name:<15} {label:<14} "
                        f"{d['mean_oa']:>7.2f} ± {d['std_oa']:<5.2f}  "
                        f"{d['mean_aa']:>7.2f} ± {d['std_aa']:<5.2f}  "
                        f"{d['mean_kappa']:>7.2f} ± {d['std_kappa']:<5.2f}  "
                        f"{d['params']['total']:>10,} {pct:>5.1f}%\n")
            f.write("\n")

        # Per-class tables
        for ds_entry in dataset_list:
            dataset_name = ds_entry['dataset_name']
            ds = all_results['datasets'][dataset_name]
            nc = ds['num_classes']
            f.write(f"\n--- {dataset_name}: Per-Class Accuracy ---\n")
            header = f"{'Class':>8}"
            for cond in ['baseline', 'mamba1']:
                if cond == 'baseline':
                    lbl = "Transf"
                else:
                    lbl = "Mamba1"
                header += f"  {lbl+' mean':>10} {lbl+' std':>8}"
            f.write(header + "\n")
            for c in range(nc):
                line = f"{'C'+str(c+1):>8}"
                for cond in ['baseline', 'mamba1']:
                    d = ds[cond]
                    line += f"  {d['mean_each_acc'][c]:>10.2f} {d['std_each_acc'][c]:>8.2f}"
                f.write(line + "\n")

        # Param breakdown
        f.write(f"\n--- Parameter Breakdown ---\n")
        for ds_entry in dataset_list:
            dataset_name = ds_entry['dataset_name']
            ds = all_results['datasets'][dataset_name]
            f.write(f"\n{dataset_name}:\n")
            for cond in ['baseline', 'mamba1']:
                p = ds[cond]['params']
                if cond == 'baseline':
                    lbl = "Transformer"
                else:
                    lbl = "Mamba-1 (uni)"
                f.write(f"  {lbl}: total={p['total']:,} | cnn={p['cnn']:,} | "
                        f"tok={p['tokenization']:,} | pe={p['pos_embed']:,} | "
                        f"enc={p['encoder']:,} | cls={p['classifier']:,}\n")

    print(f"Summary saved: {summary_path}")

    print("\n=== DONE ===")
    return OUTPUTS_DIR_PATH
