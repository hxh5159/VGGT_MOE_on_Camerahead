"""
================================================================================
VGGT Sample Inference — 一键推理脚本
================================================================================

用法:
    python run_inference.py --image_dir ./my_photos --output_dir ./results

    # 指定微调后的 checkpoint
    python run_inference.py --image_dir ./my_photos --output_dir ./results \
        --checkpoint logs/ablation_D_no_grad/ckpts/checkpoint_epoch_10.pth

    # CPU 推理
    python run_inference.py --image_dir ./my_photos --device cpu

输入:  任意尺寸/比例 的 JPG/PNG 图片文件夹（至少2张）
输出:
    depth_visualization.png        — 每帧深度图对比
    merged_pointcloud.png          — 所有帧融合的 3D 重建
    pose_enc.npy / depth.npy 等    — 数值结果
    reconstructed_world_points.npy — 从深度图反投影重建的 3D 点云（融合）
================================================================================
"""

import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path

import torch
from PIL import Image


def load_images_from_folder(image_dir, img_size=518):
    """从文件夹加载图片，pad 为正方形后 resize 到 img_size。"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}
    paths = sorted(
        [p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts]
    )
    if len(paths) < 2:
        print(f"错误: 需要至少 2 张图片，只找到 {len(paths)} 张")
        sys.exit(1)

    print(f"找到 {len(paths)} 张图片:")
    for p in paths:
        print(f"  {p.name}")

    tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        # Pad 到正方形
        w, h = img.size
        max_side = max(w, h)
        new_img = Image.new("RGB", (max_side, max_side), (255, 255, 255))
        new_img.paste(img, ((max_side - w) // 2, (max_side - h) // 2))
        # Resize
        new_img = new_img.resize((img_size, img_size), Image.BILINEAR)
        arr = np.array(new_img, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1))  # C, H, W

    # Stack: [S, C, H, W] → [1, S, C, H, W]
    images = torch.stack(tensors)  # S, C, H, W
    print(f"预处理完成: {images.shape} (S, C, H, W)")
    print(f"数值范围: [{images.min():.3f}, {images.max():.3f}]")
    return images.unsqueeze(0)  # B, S, C, H, W


def reconstruct_3d_from_depth(depth, depth_conf, extrinsics, intrinsics,
                              conf_thresh=0.5, sample_step=4):
    """
    从深度图 + 相机参数重建并融合 3D 点云。

    Args:
        depth:          (1, S, H, W, 1) 深度图
        depth_conf:     (1, S, H, W)    深度置信度
        extrinsics:     (1, S, 3, 4)    外参矩阵（cam from world, OpenCV）
        intrinsics:     (1, S, 3, 3)    内参矩阵
        conf_thresh:    置信度阈值，低于此值的像素被丢弃
        sample_step:    采样步长，每隔 step 个像素取一个（减少点数）

    Returns:
        merged_xyz:  (N, 3) 融合后的世界坐标点
        merged_colors: (N, 3) 对应 RGB 颜色（归一化 [0,1]）
    """
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    S = depth.shape[1]
    all_xyz = []
    all_colors = []

    for i in range(S):
        # 每帧单独反投影（保留帧维度：使用 i:i+1 切片而非 i 索引）
        pts_world = unproject_depth_map_to_point_map(
            depth[0, i:i+1].cpu().numpy(),         # (1, H, W, 1) — 保留帧维
            extrinsics[0, i:i+1].cpu().numpy(),    # (1, 3, 4)
            intrinsics[0, i:i+1].cpu().numpy(),    # (1, 3, 3)
        )  # (1, H, W, 3)
        pts_world = pts_world[0]  # 去掉帧维 → (H, W, 3)

        conf = depth_conf[0, i].cpu().numpy()  # (H, W)

        # 置信度过滤 + 下采样
        H, W = pts_world.shape[:2]
        mask = conf > conf_thresh
        for y in range(0, H, sample_step):
            for x in range(0, W, sample_step):
                if mask[y, x]:
                    all_xyz.append(pts_world[y, x])

    if len(all_xyz) == 0:
        print("  ⚠ 没有通过置信度过滤的点，尝试降低 conf_thresh")
        return None, None

    all_xyz = np.stack(all_xyz, axis=0)  # (N, 3)

    # 裁掉离群点（超出 3σ 范围的点）
    centroid = np.median(all_xyz, axis=0)
    dists = np.linalg.norm(all_xyz - centroid, axis=1)
    thresh = np.median(dists) + 3.0 * np.std(dists)
    inlier = dists < thresh
    all_xyz = all_xyz[inlier]

    print(f"  融合点云: {all_xyz.shape[0]} 个点（{S} 帧合并）")
    return all_xyz, dists[inlier]  # 用距离代替颜色


def main():
    parser = argparse.ArgumentParser(description="VGGT 一键推理")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="图片文件夹路径")
    parser.add_argument("--output_dir", type=str, default="inference_output",
                        help="输出目录")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="微调 checkpoint 路径（可选，不填则用预训练权重）")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--conf_thresh", type=float, default=0.5,
                        help="深度置信度阈值（低于此值不参与重建）")
    parser.add_argument("--sample_step", type=int, default=2,
                        help="点云下采样步长（1=全像素，2=每隔一个）")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========== Step 1: 加载图片 ==========
    print("\n[1/4] 加载并预处理图片...")
    images = load_images_from_folder(args.image_dir, args.img_size)

    # ========== Step 2: 加载模型 ==========
    # 注意: 模型结构需要与训练时一致
    # Co3D 消融/全参微调: 只训练 camera + depth head
    print("\n[2/4] 加载 VGGT 模型...")
    from vggt.models.vggt import VGGT

    model = VGGT(
        enable_camera=True,
        enable_depth=True,
        enable_point=False,  # Co3D 微调未训练 point head
        enable_track=False,  # Co3D 微调未训练 track head
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"  加载 checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
    else:
        print("  使用 HuggingFace 预训练权重")

    model = model.to(args.device)
    model.eval()
    images = images.to(args.device)
    print(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # ========== Step 3: 推理 ==========
    print("\n[3/4] 推理中...")
    t0 = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            predictions = model(images=images)
    print(f"  耗时 {time.time() - t0:.1f}s")

    # ========== 保存 .npy ==========
    print("\n保存数值结果...")
    if "pose_enc" in predictions:
        np.save(output_dir / "pose_enc.npy",
                predictions["pose_enc"].cpu().numpy())
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        ext, intr = pose_encoding_to_extri_intri(
            predictions["pose_enc"],
            (args.img_size, args.img_size),
            pose_encoding_type="absT_quaR_FoV")
        np.save(output_dir / "extrinsics.npy", ext.cpu().numpy())
        np.save(output_dir / "intrinsics.npy", intr.cpu().numpy())
        print(f"  ✓ pose_enc.npy, extrinsics.npy, intrinsics.npy")

    if "depth" in predictions:
        np.save(output_dir / "depth.npy",
                predictions["depth"].cpu().numpy())
        np.save(output_dir / "depth_conf.npy",
                predictions["depth_conf"].cpu().numpy())
        print(f"  ✓ depth.npy, depth_conf.npy")

    # ========== Step 4: 深度反投影 → 融合 3D 重建 ==========
    print("\n[4/4] 深度图 → 3D 重建...")
    merged_xyz = None
    if "depth" in predictions and "pose_enc" in predictions:
        merged_xyz, depth_dists = reconstruct_3d_from_depth(
            predictions["depth"],
            predictions["depth_conf"],
            ext,
            intr,
            conf_thresh=args.conf_thresh,
            sample_step=args.sample_step,
        )
        if merged_xyz is not None:
            np.save(output_dir / "reconstructed_world_points.npy", merged_xyz)
            print(f"  ✓ reconstructed_world_points.npy")

    # ========== 画图 ==========
    print("\n生成可视化...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        _, S = images.shape[:2]
        depth_np = predictions["depth"].cpu().numpy()
        images_np = images[0].cpu().permute(0, 2, 3, 1).numpy()

        # ---- 图1: 深度图（原始 vs 预测） ----
        n_show = min(S, 4)
        fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
        if n_show == 1:
            axes = axes.reshape(2, 1)
        for i in range(n_show):
            axes[0, i].imshow(np.clip(images_np[i], 0, 1))
            axes[0, i].set_title(f"Frame {i}")
            axes[0, i].axis("off")
            d = depth_np[0, i, ..., 0]
            im = axes[1, i].imshow(d, cmap="inferno")
            axes[1, i].set_title(f"Depth {i}")
            axes[1, i].axis("off")
            plt.colorbar(im, ax=axes[1, i], fraction=0.046)
        plt.suptitle("VGGT — Depth Prediction", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "depth_visualization.png", dpi=150)
        plt.close()
        print(f"  ✓ depth_visualization.png")

        # ---- 图2: 融合 3D 重建 ----
        if merged_xyz is not None and len(merged_xyz) > 0:
            fig = plt.figure(figsize=(14, 7))

            # 视角1: 正面
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            n_pts = min(len(merged_xyz), 50000)
            idx = np.random.choice(len(merged_xyz), n_pts, replace=False)
            colors = depth_dists[idx] if depth_dists is not None else merged_xyz[idx, 2]
            ax1.scatter(merged_xyz[idx, 0], merged_xyz[idx, 1], merged_xyz[idx, 2],
                        c=colors, cmap="viridis", s=0.5, alpha=0.6)
            ax1.set_title("Fused 3D Reconstruction (Front View)")
            ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
            ax1.view_init(elev=10, azim=0)

            # 视角2: 俯视
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            ax2.scatter(merged_xyz[idx, 0], merged_xyz[idx, 1], merged_xyz[idx, 2],
                        c=colors, cmap="viridis", s=0.5, alpha=0.6)
            ax2.set_title("Fused 3D Reconstruction (Top View)")
            ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
            ax2.view_init(elev=80, azim=0)

            plt.suptitle(f"VGGT — 3D Reconstruction ({S} frames fused)", fontsize=14)
            plt.tight_layout()
            plt.savefig(output_dir / "merged_pointcloud.png", dpi=200)
            plt.close()
            print(f"  ✓ merged_pointcloud.png ({n_pts} points shown)")

    except Exception as e:
        print(f"  ⚠ 可视化失败: {e}")

    print(f"\n{'=' * 50}")
    print(f"完成！结果保存在 {output_dir}/")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name} ({mb:.1f} MB)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
