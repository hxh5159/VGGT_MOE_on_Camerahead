"""
================================================================================
VGGT Out-of-Sample Inference Script
================================================================================

Performs inference on custom images (not from Co3D/training distribution)
to demonstrate VGGT's generalization capability.

This script:
1. Loads any set of images (e.g., phone photos, web images)
2. Runs VGGT to predict camera poses, depth, and 3D points
3. Visualizes the results

Usage:
    # On a folder of images
    python inference_custom.py --image_dir ./my_photos --output_dir ./results

    # Using the demo viser interface
    python demo_viser.py

    # Using gradio web interface
    python demo_gradio.py
"""

import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path

import torch
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="VGGT Out-of-Sample Inference")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="inference_output",
                        help="Directory to save results")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to fine-tuned checkpoint (optional)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--img_size", type=int, default=518,
                        help="Input image size for the model")
    parser.add_argument("--query_mode", type=str, default="grid",
                        choices=["grid", "random", "none"],
                        help="Query point mode for tracking")
    parser.add_argument("--num_query_points", type=int, default=256,
                        help="Number of query points for tracking")

    args = parser.parse_args()

    # Validate input
    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        print(f"ERROR: {image_dir} is not a valid directory")
        sys.exit(1)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}
    image_files = sorted(
        [f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions]
    )

    if len(image_files) < 2:
        print(f"ERROR: Need at least 2 images, found {len(image_files)}")
        print(f"Supported formats: {image_extensions}")
        sys.exit(1)

    print(f"Found {len(image_files)} images:")
    for f in image_files:
        print(f"  {f.name}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Load images
    # =========================================================================
    print("\n[1/4] Loading and preprocessing images...")
    from vggt.utils.load_fn import load_and_preprocess_images

    images_np = []
    for img_path in image_files:
        img = np.array(Image.open(img_path).convert("RGB"))
        images_np.append(img)
        print(f"  Loaded {img_path.name}: {img.shape}")

    # Preprocess: pad to square, resize to 518
    images_tensor, _ = load_and_preprocess_images(images_np, mode="pad")
    print(f"  Preprocessed: {images_tensor.shape} (B, S, C, H, W)")
    print(f"  Value range: [{images_tensor.min():.3f}, {images_tensor.max():.3f}]")

    # =========================================================================
    # Step 2: Load model
    # =========================================================================
    print("\n[2/4] Loading VGGT model...")
    from vggt.models.vggt import VGGT

    model = VGGT(
        enable_camera=True,
        enable_depth=True,
        enable_point=True,
        enable_track=True,
    )

    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"  Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        print("  Using pre-trained weights from HuggingFace (default)")
        print("  (To use your fine-tuned weights, specify --checkpoint)")

    model = model.to(args.device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params / 1e6:.1f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.1f}M")

    # =========================================================================
    # Step 3: Prepare query points (optional tracking)
    # =========================================================================
    print("\n[3/4] Preparing inputs...")
    images_tensor = images_tensor.to(args.device)

    query_points = None
    if args.query_mode == "grid":
        H, W = images_tensor.shape[-2:]
        # Create a grid of query points
        grid_size = int(np.sqrt(args.num_query_points))
        y = torch.linspace(0, H - 1, grid_size)
        x = torch.linspace(0, W - 1, grid_size)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        query_points = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        query_points = query_points[:args.num_query_points]  # Limit
        print(f"  Grid query points: {query_points.shape}")
    elif args.query_mode == "random":
        H, W = images_tensor.shape[-2:]
        query_points = torch.rand(args.num_query_points, 2) * torch.tensor([W, H])
        print(f"  Random query points: {query_points.shape}")

    if query_points is not None:
        query_points = query_points.to(args.device)

    # =========================================================================
    # Step 4: Run inference
    # =========================================================================
    print("\n[4/4] Running inference...")
    start_time = time.time()

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            predictions = model(images=images_tensor, query_points=query_points)

    elapsed = time.time() - start_time
    print(f"  Inference completed in {elapsed:.2f}s")

    # =========================================================================
    # Save results
    # =========================================================================
    print("\nSaving results...")

    # Camera poses
    if "pose_enc" in predictions:
        pose_enc = predictions["pose_enc"].cpu().numpy()
        print(f"  Pose encoding: {pose_enc.shape} (B, S, 9)")
        print(f"    Format: [Tx, Ty, Tz, Qx, Qy, Qz, Qw, FoV_H, FoV_W]")
        np.save(output_dir / "pose_enc.npy", pose_enc)

        # Decode pose to human-readable format
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            predictions["pose_enc"],
            images_tensor.shape[-2:],
            pose_encoding_type="absT_quaR_FoV"
        )
        print(f"  Extrinsics: {extrinsics.shape}")
        print(f"  Intrinsics: {intrinsics.shape}")
        np.save(output_dir / "extrinsics.npy", extrinsics.cpu().numpy())
        np.save(output_dir / "intrinsics.npy", intrinsics.cpu().numpy())

    # Depth
    if "depth" in predictions:
        depth = predictions["depth"].cpu().numpy()
        depth_conf = predictions["depth_conf"].cpu().numpy()
        print(f"  Depth: {depth.shape} (B, S, H, W, 1)")
        print(f"  Depth confidence: {depth_conf.shape} (B, S, H, W)")
        np.save(output_dir / "depth.npy", depth)
        np.save(output_dir / "depth_conf.npy", depth_conf)

    # 3D World Points
    if "world_points" in predictions:
        pts3d = predictions["world_points"].cpu().numpy()
        pts3d_conf = predictions["world_points_conf"].cpu().numpy()
        print(f"  World points: {pts3d.shape} (B, S, H, W, 3)")
        print(f"  Points confidence: {pts3d_conf.shape} (B, S, H, W)")
        np.save(output_dir / "world_points.npy", pts3d)
        np.save(output_dir / "world_points_conf.npy", pts3d_conf)

    # Tracks
    if "track" in predictions and predictions["track"] is not None:
        track = predictions["track"].cpu().numpy()
        vis = predictions.get("vis", torch.zeros(1)).cpu().numpy()
        conf = predictions.get("conf", torch.zeros(1)).cpu().numpy()
        print(f"  Tracks: {track.shape} (B, S, N, 2)")
        np.save(output_dir / "tracks.npy", track)
        np.save(output_dir / "track_vis.npy", vis)
        np.save(output_dir / "track_conf.npy", conf)

    # =========================================================================
    # Generate visualizations
    # =========================================================================
    print("\nGenerating visualizations...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        B, S = images_tensor.shape[:2]
        viz_frames = min(S, 4)  # Visualize up to 4 frames

        # 1. Depth visualization
        if "depth" in predictions:
            fig, axes = plt.subplots(2, viz_frames, figsize=(4 * viz_frames, 8))
            if viz_frames == 1:
                axes = axes.reshape(2, 1)

            for i in range(viz_frames):
                # Top: input image
                img = images_tensor[0, i].cpu().permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                axes[0, i].imshow(img)
                axes[0, i].set_title(f"Frame {i}")
                axes[0, i].axis("off")

                # Bottom: depth
                d = depth[0, i, ..., 0]
                im = axes[1, i].imshow(d, cmap="inferno")
                axes[1, i].set_title(f"Depth {i}")
                axes[1, i].axis("off")
                plt.colorbar(im, ax=axes[1, i], fraction=0.046)

            plt.suptitle("VGGT Depth Prediction", fontsize=14)
            plt.tight_layout()
            plt.savefig(output_dir / "depth_visualization.png", dpi=150)
            plt.close()
            print(f"  Saved depth_visualization.png")

        # 2. 3D point cloud visualization (simple)
        if "world_points" in predictions:
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(12, 6))
            for i in range(min(viz_frames, 2)):
                ax = fig.add_subplot(1, 2, i + 1, projection='3d')
                pts = pts3d[0, i]
                conf_mask = pts3d_conf[0, i] > 0.5

                # Sample points to avoid overcrowding
                step = max(1, pts.shape[0] // 50)
                pts_sample = pts[::step, ::step][conf_mask[::step, ::step]]

                if len(pts_sample) > 0:
                    ax.scatter(
                        pts_sample[:, 0],
                        pts_sample[:, 1],
                        pts_sample[:, 2],
                        c=pts_sample[:, 2],
                        cmap="viridis",
                        s=1,
                        alpha=0.5,
                    )
                ax.set_title(f"3D Points - Frame {i}")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")

            plt.suptitle("VGGT 3D Point Cloud Prediction", fontsize=14)
            plt.tight_layout()
            plt.savefig(output_dir / "pointcloud_visualization.png", dpi=150)
            plt.close()
            print(f"  Saved pointcloud_visualization.png")

        # 3. Track visualization (if available)
        if "track" in predictions and predictions["track"] is not None:
            fig, axes = plt.subplots(1, min(viz_frames, 4), figsize=(4 * min(viz_frames, 4), 4))
            if min(viz_frames, 4) == 1:
                axes = [axes]

            track_np = track[0]  # S x N x 2
            vis_np = vis[0]  # S x N

            for i in range(min(viz_frames, 4)):
                img = images_tensor[0, i].cpu().permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                axes[i].imshow(img)

                # Draw tracks visible in this frame
                for j in range(min(track_np.shape[1], 50)):
                    if vis_np[i, j] > 0.5:
                        axes[i].scatter(track_np[i, j, 0], track_np[i, j, 1],
                                        c='r', s=10, alpha=0.7)

                axes[i].set_title(f"Tracks Frame {i}")
                axes[i].axis("off")

            plt.suptitle("VGGT Point Tracking", fontsize=14)
            plt.tight_layout()
            plt.savefig(output_dir / "track_visualization.png", dpi=150)
            plt.close()
            print(f"  Saved track_visualization.png")

    except Exception as e:
        print(f"  Warning: Visualization failed: {e}")
        print(f"  Numerical results are still saved in {output_dir}/")

    print(f"\n{'=' * 60}")
    print(f"Inference complete! Results saved to {output_dir}/")
    print(f"Files saved:")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name} ({size_mb:.1f} MB)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
