#!/usr/bin/env python3
"""
Generate dataset visualization figures for the paper.
For each dataset: (a) Pseudo-color Map, (b) LiDAR/SAR/DSM Map,
(c) Training Samples, (d) Test Samples — similar to Figures 4-7 in MFT paper.

Usage:
    python visualize_datasets.py

Requires raw scene-level data (not 11x11 patches).
Currently supports: Augsburg (HSI+SAR, HSI+DSM)
Houston and Trento need raw scene files added to DATA_DIR.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import scipy.io as sio
import os

# ── Config ──
DATA_DIR = "/Users/chang/Explore/research_1/MFT_Data"
OUT_DIR = "/Users/chang/Explore/research_1/mft_research/dataset_figures"
os.makedirs(OUT_DIR, exist_ok=True)

DPI = 300


def normalize_band(band):
    """Min-max normalize a single band to [0, 1]."""
    mn, mx = np.nanmin(band), np.nanmax(band)
    if mx - mn < 1e-10:
        return np.zeros_like(band, dtype=np.float32)
    return ((band - mn) / (mx - mn)).astype(np.float32)


def make_pseudo_color(hsi, bands_rgb):
    """Create pseudo-color RGB from HSI using specified band indices."""
    r = normalize_band(hsi[:, :, bands_rgb[0]])
    g = normalize_band(hsi[:, :, bands_rgb[1]])
    b = normalize_band(hsi[:, :, bands_rgb[2]])
    # Slight gamma for better visibility
    gamma = 0.8
    rgb = np.stack([r**gamma, g**gamma, b**gamma], axis=-1)
    return np.clip(rgb, 0, 1)


def make_label_map(labels, class_colors, class_names):
    """Create colored label map from integer labels.
    label=0 is background (black)."""
    h, w = labels.shape
    img = np.zeros((h, w, 3), dtype=np.float32)
    for cls_id, color in enumerate(class_colors):
        if cls_id == 0:
            continue  # skip background
        mask = labels == cls_id
        img[mask] = [c / 255.0 for c in color]
    return img


def make_legend_patches(class_names, class_colors, counts=None):
    """Create legend patches for the class labels."""
    patches = []
    for i, (name, color) in enumerate(zip(class_names, class_colors)):
        if i == 0 and name == "Background":
            continue
        c = [c / 255.0 for c in color]
        label = name
        if counts is not None and i < len(counts):
            label = f"{name} ({counts[i]:,})"
        patches.append(mpatches.Patch(facecolor=c, edgecolor='gray', label=label, linewidth=0.5))
    return patches


def save_4panel(pseudo_rgb, lidar_img, train_map, test_map,
                legend_patches, dataset_name, lidar_label,
                train_counts, test_counts, class_names, class_colors):
    """Save a 4-panel figure like the MFT paper."""

    # Create dataset-specific subfolder
    safe_name = dataset_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    ds_dir = os.path.join(OUT_DIR, safe_name)
    os.makedirs(ds_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(20, 6), dpi=DPI)
    titles = [
        f"(a) Pseudo-color Map",
        f"(b) {lidar_label} Map",
        f"(c) Training Samples",
        f"(d) Test Samples",
    ]
    images = [pseudo_rgb, lidar_img, train_map, test_map]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')

    plt.suptitle(f"{dataset_name} Dataset", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save without legend (clean image)
    path = os.path.join(ds_dir, f"4panel.png")
    fig.savefig(path, dpi=DPI, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    print(f"  Saved: {path}")

    # Also save individual panels at higher quality
    for i, (img, title) in enumerate(zip(images, titles)):
        fig_s, ax_s = plt.subplots(1, 1, figsize=(6, 6), dpi=DPI)
        ax_s.imshow(img)
        ax_s.set_title(title, fontsize=12, fontweight='bold')
        ax_s.axis('off')
        panel_name = ['pseudocolor', 'lidar', 'train_samples', 'test_samples'][i]
        p = os.path.join(ds_dir, f"{panel_name}.png")
        fig_s.savefig(p, dpi=DPI, bbox_inches='tight', pad_inches=0.05, facecolor='white')
        plt.close(fig_s)
        print(f"  Saved: {p}")

    # Save legend + class table as separate image
    fig_leg, ax_leg = plt.subplots(1, 1, figsize=(8, 4), dpi=DPI)
    ax_leg.axis('off')

    # Build table data
    table_data = []
    col_colors = []
    for i in range(1, len(class_names)):
        c = [c/255.0 for c in class_colors[i]]
        tr = train_counts.get(i, 0)
        te = test_counts.get(i, 0)
        table_data.append([class_names[i], f"{tr:,}", f"{te:,}"])
        col_colors.append(c)

    table = ax_leg.table(
        cellText=table_data,
        colLabels=["Land Cover", "Train", "Test"],
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Color the first column cells
    for i in range(len(table_data)):
        cell = table[i + 1, 0]
        cell.set_facecolor(col_colors[i])
        # White text on dark backgrounds
        r, g, b = col_colors[i]
        if r * 0.299 + g * 0.587 + b * 0.114 < 0.5:
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_text_props(fontweight='bold')

    ax_leg.set_title(f"{dataset_name} — Class Distribution", fontsize=11, fontweight='bold')
    p = os.path.join(ds_dir, f"legend.png")
    fig_leg.savefig(p, dpi=DPI, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close(fig_leg)
    print(f"  Saved: {p}")

    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# AUGSBURG (HSI + SAR + DSM) — raw scene data available
# ═══════════════════════════════════════════════════════════
def visualize_augsburg():
    print("\n=== Augsburg ===")
    base = os.path.join(DATA_DIR, "HS-SAR-DSM Augsburg")

    hsi = sio.loadmat(os.path.join(base, "data_HS_LR.mat"))["data_HS_LR"].astype(np.float32)
    sar = sio.loadmat(os.path.join(base, "data_SAR_HR.mat"))["data_SAR_HR"].astype(np.float32)
    dsm = sio.loadmat(os.path.join(base, "data_DSM.mat"))["data_DSM"].astype(np.float32)
    train_gt = sio.loadmat(os.path.join(base, "TrainImage.mat"))["TrainImage"].astype(np.int32)
    test_gt = sio.loadmat(os.path.join(base, "TestImage.mat"))["TestImage"].astype(np.int32)

    print(f"  HSI: {hsi.shape}, SAR: {sar.shape}, DSM: {dsm.shape}")
    print(f"  Train GT: {train_gt.shape}, unique={np.unique(train_gt)}")
    print(f"  Test GT:  {test_gt.shape}, unique={np.unique(test_gt)}")

    # Augsburg classes (7 classes, 0=background)
    class_names = ["Background", "Forest", "Residential", "Industrial",
                   "Low Plants", "Allotment", "Commercial", "Water"]
    class_colors = [
        (0, 0, 0),          # 0: Background
        (34, 139, 34),      # 1: Forest - dark green
        (255, 99, 71),      # 2: Residential - tomato red
        (255, 165, 0),      # 3: Industrial - orange
        (144, 238, 144),    # 4: Low Plants - light green
        (255, 215, 0),      # 5: Allotment - gold
        (138, 43, 226),     # 6: Commercial - blue violet
        (30, 144, 255),     # 7: Water - dodger blue
    ]

    # Pseudo-color from HSI (bands ~R=80, G=40, B=10 for 180-band Augsburg)
    pseudo_rgb = make_pseudo_color(hsi, bands_rgb=[80, 40, 10])

    # SAR composite (use bands 0,1,2 as RGB)
    sar_rgb = np.stack([normalize_band(sar[:,:,0]),
                        normalize_band(sar[:,:,1]),
                        normalize_band(sar[:,:,2])], axis=-1)
    sar_rgb = np.clip(sar_rgb ** 0.6, 0, 1)  # gamma for SAR

    # DSM as grayscale/terrain colormap
    dsm_norm = normalize_band(dsm)
    dsm_colored = plt.cm.terrain(dsm_norm)[:, :, :3]

    # Label maps
    train_map = make_label_map(train_gt, class_colors, class_names)
    test_map = make_label_map(test_gt, class_colors, class_names)

    # Count samples per class
    train_counts = {c: int(np.sum(train_gt == c)) for c in range(len(class_names))}
    test_counts = {c: int(np.sum(test_gt == c)) for c in range(len(class_names))}
    print(f"  Train counts: {train_counts}")
    print(f"  Test counts:  {test_counts}")

    legend_patches = make_legend_patches(class_names, class_colors)

    # HSI + SAR version
    save_4panel(pseudo_rgb, sar_rgb, train_map, test_map,
                legend_patches, "Augsburg (HSI+SAR)", "SAR",
                train_counts, test_counts, class_names, class_colors)

    # HSI + DSM version
    save_4panel(pseudo_rgb, dsm_colored, train_map, test_map,
                legend_patches, "Augsburg (HSI+DSM)", "DSM",
                train_counts, test_counts, class_names, class_colors)


# ═══════════════════════════════════════════════════════════
# HOUSTON — needs raw scene files
# Expected files: Houston_HSI.mat, Houston_LiDAR.mat, Houston_GT.mat
# OR: Houston.mat with keys 'HSI', 'LiDAR', 'GT'
# ═══════════════════════════════════════════════════════════
def visualize_houston():
    """Generate Houston 2013 visualizations from prepared .npy files."""
    raw_dir = "/Users/chang/Explore/research_1/MFT_Data/Houston_raw"
    if not os.path.exists(raw_dir + "/hsi.npy"):
        print("\n=== Houston ===\n  SKIPPED: hsi.npy not found.")
        return
    print("\n=== Houston ===")
    hsi = np.load(raw_dir + "/hsi.npy")
    lidar = np.load(raw_dir + "/lidar.npy")
    train_gt = np.load(raw_dir + "/train_gt.npy")
    test_gt = np.load(raw_dir + "/test_gt.npy") if os.path.exists(raw_dir + "/test_gt.npy") else None
    print(f"  HSI: {hsi.shape}, LiDAR: {lidar.shape}, Train GT: {train_gt.shape}")
    class_names = ["Background", "Grass-healthy", "Grass-stressed", "Grass-synthetic",
                   "Tree", "Soil", "Water", "Residential", "Commercial", "Road",
                   "Highway", "Railway", "Parking-lot1", "Parking-lot2",
                   "Tennis-court", "Running-track"]
    class_colors = [(0,0,0),(0,205,0),(127,255,0),(46,139,87),(0,100,0),(255,165,79),
        (0,0,255),(255,0,0),(216,191,216),(128,128,128),(255,0,255),(0,255,255),
        (255,255,0),(238,154,0),(85,26,139),(255,127,80)]
    pseudo_rgb = make_pseudo_color(hsi, bands_rgb=[64, 43, 22])
    lidar_colored = plt.cm.gray(normalize_band(lidar))[:, :, :3]
    train_map = make_label_map(train_gt, class_colors, class_names)
    train_counts = {c: int(np.sum(train_gt == c)) for c in range(len(class_names))}
    test_map = make_label_map(test_gt, class_colors, class_names) if test_gt is not None else None
    test_counts = {c: int(np.sum(test_gt == c)) for c in range(len(class_names))} if test_gt is not None else {}
    ds_dir = os.path.join(OUT_DIR, "houston"); os.makedirs(ds_dir, exist_ok=True)
    if test_gt is not None:
        # 4-panel with test samples
        legend_patches = make_legend_patches(class_names, class_colors)
        save_4panel(pseudo_rgb, lidar_colored, train_map, test_map,
                    legend_patches, "Houston", "LiDAR DSM",
                    train_counts, test_counts, class_names, class_colors)
    else:
        # 3-panel without test
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=DPI)
        for ax, img, title in zip(axes, [pseudo_rgb, lidar_colored, train_map],
                ["(a) Pseudo-color Map", "(b) LiDAR DSM Map", "(c) Training Samples"]):
            ax.imshow(img); ax.set_title(title, fontsize=11, fontweight='bold', pad=8); ax.axis('off')
        plt.suptitle("Houston 2013 Dataset", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(ds_dir, "4panel.png"), dpi=DPI, bbox_inches='tight', pad_inches=0.1, facecolor='white')
        print(f"  Saved: {ds_dir}/4panel.png"); plt.close(fig)
    # Individual panels
    panels = [("pseudocolor", pseudo_rgb, "(a) Pseudo-color Map"),
              ("lidar", lidar_colored, "(b) LiDAR DSM Map"),
              ("train_samples", train_map, "(c) Training Samples")]
    if test_map is not None:
        panels.append(("test_samples", test_map, "(d) Test Samples"))
    for pname, img, title in panels:
        fig_s, ax_s = plt.subplots(1, 1, figsize=(8, 3), dpi=DPI)
        ax_s.imshow(img); ax_s.set_title(title, fontsize=12, fontweight='bold'); ax_s.axis('off')
        p = os.path.join(ds_dir, pname + ".png")
        fig_s.savefig(p, dpi=DPI, bbox_inches='tight', pad_inches=0.05, facecolor='white')
        plt.close(fig_s); print(f"  Saved: {p}")
    fig_leg, ax_leg = plt.subplots(1, 1, figsize=(8, 5), dpi=DPI); ax_leg.axis('off')
    td = [[class_names[i], f"{train_counts.get(i,0):,}"] for i in range(1, len(class_names))]
    cc = [[v/255.0 for v in class_colors[i]] for i in range(1, len(class_names))]
    table = ax_leg.table(cellText=td, colLabels=["Land Cover","Train"], cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.4)
    for i in range(len(td)):
        cell = table[i+1, 0]; cell.set_facecolor(cc[i])
        if cc[i][0]*0.299+cc[i][1]*0.587+cc[i][2]*0.114 < 0.5:
            cell.set_text_props(color='white', fontweight='bold')
        else: cell.set_text_props(fontweight='bold')
    ax_leg.set_title("Houston 2013 — Class Distribution", fontsize=11, fontweight='bold')
    p = os.path.join(ds_dir, "legend.png")
    fig_leg.savefig(p, dpi=DPI, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close(fig_leg); print(f"  Saved: {p}")


# ═══════════════════════════════════════════════════════════
# TRENTO — raw scene files
# ═══════════════════════════════════════════════════════════
def visualize_trento():
    base = "/Users/chang/Explore/research_1/TrentoDateset/TrentoDataset"
    if not os.path.exists(base):
        print("\n=== Trento ===")
        print("  SKIPPED: Raw scene files not found.")
        return

    print("\n=== Trento ===")
    hsi = sio.loadmat(os.path.join(base, "HSI_Trento.mat"))["HSI_Trento"].astype(np.float32)
    lidar = sio.loadmat(os.path.join(base, "Lidar_Trento.mat"))["Lidar_Trento"].astype(np.float32)
    gt = sio.loadmat(os.path.join(base, "GT_Trento.mat"))["GT_Trento"].astype(np.int32)

    print(f"  HSI: {hsi.shape}, LiDAR: {lidar.shape}, GT: {gt.shape}")
    print(f"  GT unique: {np.unique(gt)}")

    # Trento: 6 classes (0=background)
    class_names = ["Background", "Apples", "Buildings", "Ground",
                   "Woods", "Vineyard", "Roads"]
    class_colors = [
        (0, 0, 0),          # 0: Background
        (255, 0, 0),        # 1: Apples - red
        (0, 0, 255),        # 2: Buildings - blue
        (255, 165, 0),      # 3: Ground - orange
        (34, 139, 34),      # 4: Woods - green
        (148, 0, 211),      # 5: Vineyard - purple
        (128, 128, 128),    # 6: Roads - gray
    ]

    # Train/test split: use the same split as in the paper (random 5%)
    # Since we only have GT, split by taking labeled pixels
    np.random.seed(42)
    labeled_mask = gt > 0
    labeled_indices = np.argwhere(labeled_mask)
    n_labeled = len(labeled_indices)

    # 5% train
    perm = np.random.permutation(n_labeled)
    n_train = max(1, int(n_labeled * 0.05))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    for idx in train_idx:
        r, c = labeled_indices[idx]
        train_gt[r, c] = gt[r, c]
    for idx in test_idx:
        r, c = labeled_indices[idx]
        test_gt[r, c] = gt[r, c]

    print(f"  Labeled pixels: {n_labeled}, Train: {n_train}, Test: {n_labeled - n_train}")

    # Pseudo-color (bands ~40, 20, 10 for 63-band Trento)
    pseudo_rgb = make_pseudo_color(hsi, bands_rgb=[40, 20, 10])

    # LiDAR as terrain colormap
    lidar_norm = normalize_band(lidar)
    lidar_colored = plt.cm.gray(lidar_norm)[:, :, :3]

    train_map = make_label_map(train_gt, class_colors, class_names)
    test_map = make_label_map(test_gt, class_colors, class_names)

    train_counts = {c: int(np.sum(train_gt == c)) for c in range(len(class_names))}
    test_counts = {c: int(np.sum(test_gt == c)) for c in range(len(class_names))}
    print(f"  Train counts: {train_counts}")
    print(f"  Test counts:  {test_counts}")

    legend_patches = make_legend_patches(class_names, class_colors)
    save_4panel(pseudo_rgb, lidar_colored, train_map, test_map,
                legend_patches, "Trento", "LiDAR",
                train_counts, test_counts, class_names, class_colors)


# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

def visualize_muufl():
    raw_dir = "/Users/chang/Explore/research_1/MFT_Data/MUUFL_raw"
    if not os.path.exists(raw_dir + "/hsi.npy"):
        print("\n=== MUUFL ===\n  SKIPPED: hsi.npy not found.")
        return
    print("\n=== MUUFL ===")
    hsi = np.load(raw_dir + "/hsi.npy")
    lidar = np.load(raw_dir + "/lidar.npy")
    gt = np.load(raw_dir + "/gt.npy")
    print(f"  HSI: {hsi.shape}, LiDAR: {lidar.shape}, GT: {gt.shape}")
    print(f"  GT unique: {np.unique(gt)}")

    # 11 classes, -1=unlabeled, 0=none
    class_names = ["Unlabeled", "Trees", "Grass-Pure", "Grass-Ground", "Dirt-Sand",
                   "Road", "Water", "Shadow-Building", "Buildings", "Sidewalk",
                   "Yellow-Curb", "Cloth-Panels"]
    class_colors = [
        (0, 0, 0),          # 0/-1: unlabeled
        (34, 139, 34),      # 1: Trees
        (0, 255, 0),        # 2: Grass-Pure
        (144, 238, 144),    # 3: Grass-Ground
        (210, 180, 140),    # 4: Dirt-Sand
        (128, 128, 128),    # 5: Road
        (0, 0, 255),        # 6: Water
        (64, 64, 64),       # 7: Shadow-Building
        (255, 0, 0),        # 8: Buildings
        (200, 200, 200),    # 9: Sidewalk
        (255, 255, 0),      # 10: Yellow-Curb
        (255, 0, 255),      # 11: Cloth-Panels
    ]

    # Remap -1 to 0 for visualization
    gt_vis = gt.copy()
    gt_vis[gt_vis < 0] = 0

    # Train/test split: 5% train
    labeled_mask = gt_vis > 0
    labeled_indices = np.argwhere(labeled_mask)
    n_labeled = len(labeled_indices)
    np.random.seed(42)
    perm = np.random.permutation(n_labeled)
    n_train = max(1, int(n_labeled * 0.05))

    train_gt = np.zeros_like(gt_vis)
    test_gt = np.zeros_like(gt_vis)
    for idx in perm[:n_train]:
        r, c = labeled_indices[idx]
        train_gt[r, c] = gt_vis[r, c]
    for idx in perm[n_train:]:
        r, c = labeled_indices[idx]
        test_gt[r, c] = gt_vis[r, c]

    print(f"  Labeled: {n_labeled}, Train: {n_train}, Test: {n_labeled - n_train}")

    # Pseudo-color (bands 40, 20, 10 for 64-band MUUFL)
    pseudo_rgb = make_pseudo_color(hsi, bands_rgb=[40, 20, 10])

    # LiDAR: use first channel as grayscale
    lidar_colored = plt.cm.gray(normalize_band(lidar[:, :, 0]))[:, :, :3]

    train_map = make_label_map(train_gt, class_colors, class_names)
    test_map = make_label_map(test_gt, class_colors, class_names)
    train_counts = {c: int(np.sum(train_gt == c)) for c in range(len(class_names))}
    test_counts = {c: int(np.sum(test_gt == c)) for c in range(len(class_names))}

    legend_patches = make_legend_patches(class_names, class_colors)
    save_4panel(pseudo_rgb, lidar_colored, train_map, test_map,
                legend_patches, "MUUFL", "LiDAR",
                train_counts, test_counts, class_names, class_colors)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Output dir: {OUT_DIR}")
    visualize_augsburg()
    visualize_houston()
    visualize_trento()
    visualize_muufl()
    print(f"\nAll figures saved to: {OUT_DIR}")
