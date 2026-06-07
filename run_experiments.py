"""
================================================================================
VGGT CV Course Assignment - Experiment Runner
================================================================================

This script orchestrates:
1. Full-parameter fine-tuning on Co3D
2. Ablation experiments to identify which loss function is removable
3. Out-of-sample inference
4. Hardware usage monitoring

Usage:
    # Full fine-tuning (H100/A100 required for full params)
    python run_experiments.py --mode full_finetune --co3d_dir /path/to/co3d --anno_dir /path/to/anno

    # Ablation experiments (can run on 4090 with frozen backbone)
    python run_experiments.py --mode ablation --co3d_dir /path/to/co3d --anno_dir /path/to/anno

    # Inference on custom images
    python run_experiments.py --mode inference --image_dir /path/to/images

    # Hardware check (estimate memory requirements)
    python run_experiments.py --mode check_hardware
"""

import argparse
import os
import sys
import subprocess
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# Ablation Experiment Definitions
# ==============================================================================

# Each experiment varies which loss components are active
# The goal is to determine which loss can be removed during fine-tuning
ABLATION_EXPERIMENTS = {
    "A_baseline": {
        "description": "Full loss (camera + depth with grad)",
        "loss_config": {
            "camera": {"weight": 5.0, "loss_type": "l1"},
            "depth": {"weight": 1.0, "gradient_loss_fn": "grad", "valid_range": 0.98},
            "point": None,
            "track": None,
        },
    },
    "B_camera_only": {
        "description": "Camera loss ONLY (no depth loss)",
        "loss_config": {
            "camera": {"weight": 5.0, "loss_type": "l1"},
            "depth": None,  # <-- Depth loss removed
            "point": None,
            "track": None,
        },
    },
    "C_depth_only": {
        "description": "Depth loss ONLY (no camera loss)",
        "loss_config": {
            "camera": None,  # <-- Camera loss removed
            "depth": {"weight": 1.0, "gradient_loss_fn": "grad", "valid_range": 0.98},
            "point": None,
            "track": None,
        },
    },
    "D_no_grad_depth": {
        "description": "Depth loss WITHOUT gradient smoothness term",
        "loss_config": {
            "camera": {"weight": 5.0, "loss_type": "l1"},
            "depth": {"weight": 1.0, "gradient_loss_fn": None, "valid_range": 0.98},
            "point": None,
            "track": None,
        },
    },
    "E_no_reg_depth": {
        "description": "Depth loss WITHOUT L2 regression term (grad only)",
        "loss_config": {
            "camera": {"weight": 5.0, "loss_type": "l1"},
            "depth": {"weight": 1.0, "gradient_loss_fn": "grad", "valid_range": 0.98,
                       "_skip_reg": True},  # custom flag
            "point": None,
            "track": None,
        },
    },
}


def generate_exp_config(exp_name, exp_def, co3d_dir, anno_dir, ckpt_path, output_dir, config_name=None):
    """
    Generate a Hydra config file for a specific ablation experiment.

    Args:
        exp_name: Experiment name for logging directory
        exp_def: Experiment definition dict
        co3d_dir: Path to Co3D dataset
        anno_dir: Path to Co3D annotations
        ckpt_path: Path to pre-trained checkpoint
        output_dir: Root output directory for logs/ckpts
        config_name: Config file name (without .yaml). Written to training/config/.
                     If None, uses exp_name.

    Returns:
        The config file name (without path)
    """
    import yaml

    if config_name is None:
        config_name = exp_name

    config = {
        "defaults": ["default_dataset"],
        "exp_name": exp_name,
        "img_size": 518,
        "num_workers": 4,
        "seed_value": 42,
        "accum_steps": 4,
        "patch_size": 14,
        "val_epoch_freq": 5,
        "max_img_per_gpu": 24,
        "limit_train_batches": 400,
        "limit_val_batches": 200,
        "data": {
            "train": {
                "_target_": "data.dynamic_dataloader.DynamicTorchDataset",
                "num_workers": "${num_workers}",
                "max_img_per_gpu": "${max_img_per_gpu}",
                "common_config": {
                    "img_size": "${img_size}",
                    "patch_size": "${patch_size}",
                    "debug": False,
                    "repeat_batch": False,
                },
                "dataset": {
                    "_target_": "data.composed_dataset.ComposedDataset",
                    "dataset_configs": [{
                        "_target_": "data.datasets.co3d.Co3dDataset",
                        "split": "train",
                        "min_num_images": 2,  # 适配小数据集
                        "CO3D_DIR": co3d_dir,
                        "CO3D_ANNOTATION_DIR": anno_dir,
                    }],
                },
            },
            "val": {
                "_target_": "data.dynamic_dataloader.DynamicTorchDataset",
                "num_workers": "${num_workers}",
                "max_img_per_gpu": "${max_img_per_gpu}",
                "common_config": {
                    "img_size": "${img_size}",
                    "patch_size": "${patch_size}",
                    "debug": False,
                },
                "dataset": {
                    "_target_": "data.composed_dataset.ComposedDataset",
                    "dataset_configs": [{
                        "_target_": "data.datasets.co3d.Co3dDataset",
                        "split": "test",
                        "min_num_images": 2,  # 适配小数据集
                        "CO3D_DIR": co3d_dir,
                        "CO3D_ANNOTATION_DIR": anno_dir,
                    }],
                },
            },
        },
        "logging": {
            "log_dir": f"{output_dir}/{exp_name}/logs",
            "log_visuals": False,
            "log_freq": 1,
            "log_level_primary": "DEBUG",
            "log_level_secondary": "WARNING",
            "all_ranks": False,
            "tensorboard_writer": {
                "_target_": "train_utils.tb_writer.TensorBoardLogger",
                "path": "${logging.log_dir}/tensorboard",
            },
            "scalar_keys_to_log": {
                "train": {"keys_to_log": [
                    "loss_objective", "loss_camera", "loss_T", "loss_R", "loss_FL",
                    "loss_conf_depth", "loss_reg_depth", "loss_grad_depth"
                ]},
                "val": {"keys_to_log": [
                    "loss_objective", "loss_camera", "loss_T", "loss_R", "loss_FL",
                    "loss_conf_depth", "loss_reg_depth", "loss_grad_depth"
                ]},
            },
        },
        "checkpoint": {
            "save_dir": f"{output_dir}/{exp_name}/ckpts",
            "save_freq": 5,
            "resume_checkpoint_path": ckpt_path,
            "strict": False,
        },
        "loss": {
            "_target_": "loss.MultitaskLoss",
            **exp_def["loss_config"],
        },
        "optim": {
            "param_group_modifiers": False,
            "optimizer": {
                "_target_": "torch.optim.AdamW",
                "lr": 5e-5,
                "weight_decay": 0.05,
            },
            "frozen_module_names": ["*aggregator*"],
            "amp": {"enabled": True, "amp_dtype": "bfloat16"},
            "gradient_clip": {
                "_target_": "train_utils.gradient_clip.GradientClipper",
                "configs": [
                    {"module_name": ["aggregator"], "max_norm": 1.0, "norm_type": 2},
                    {"module_name": ["depth"], "max_norm": 1.0, "norm_type": 2},
                    {"module_name": ["camera"], "max_norm": 1.0, "norm_type": 2},
                ],
            },
            "options": {
                "lr": [{"scheduler": {
                    "_target_": "fvcore.common.param_scheduler.CompositeParamScheduler",
                    "schedulers": [
                        {"_target_": "fvcore.common.param_scheduler.LinearParamScheduler",
                         "start_value": 1e-8, "end_value": 5e-5},
                        {"_target_": "fvcore.common.param_scheduler.CosineParamScheduler",
                         "start_value": 5e-5, "end_value": 1e-8},
                    ],
                    "lengths": [0.05, 0.95],
                    "interval_scaling": ["rescaled", "rescaled"],
                }}],
                "weight_decay": [{"scheduler": {
                    "_target_": "fvcore.common.param_scheduler.ConstantParamScheduler",
                    "value": 0.05,
                }}],
            },
        },
        "max_epochs": 10,
        "model": {
            "_target_": "vggt.models.vggt.VGGT",
            "enable_camera": True,
            "enable_depth": True,
            "enable_point": False,
            "enable_track": False,
        },
        "distributed": {
            "backend": "nccl",
            "comms_dtype": None,
            "find_unused_parameters": False,
            "timeout_mins": 30,
            "gradient_as_bucket_view": True,
            "bucket_cap_mb": 25,
            "broadcast_buffers": True,
        },
        "cuda": {
            "cudnn_deterministic": False,
            "cudnn_benchmark": False,
            "allow_tf32": True,
        },
    }

    # Save config to training/config/ so launch.py can load it
    config_dir = "training/config"
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f"{config_name}.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_name


def run_full_finetune(args):
    """Run full-parameter fine-tuning on Co3D."""
    logger.info("=" * 60)
    logger.info("Running FULL-PARAMETER FINE-TUNING on Co3D")
    logger.info("=" * 60)
    logger.info(f"Co3D dir: {args.co3d_dir}")
    logger.info(f"Annotation dir: {args.anno_dir}")
    logger.info(f"Checkpoint: {args.checkpoint_path}")

    # Check hardware capability first
    check_hardware()

    # Run training with full finetune config
    # launch.py uses --config <name> which loads training/config/<name>.yaml
    cmd = [
        "torchrun",
        "--nproc_per_node", str(args.nproc),
        "--master_port", str(args.master_port),
        "training/launch.py",
        "--config", "co3d_full_finetune",
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    if not args.dry_run:
        subprocess.run(cmd, check=True)


def run_ablation_experiments(args):
    """Run all ablation experiments sequentially."""
    import yaml

    logger.info("=" * 60)
    logger.info("Running ABLATION EXPERIMENTS on Co3D")
    logger.info("=" * 60)

    results = {}

    for exp_name, exp_def in ABLATION_EXPERIMENTS.items():
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Experiment: {exp_name}")
        logger.info(f"Description: {exp_def['description']}")
        logger.info(f"{'=' * 40}")

        output_dir = args.output_dir or "logs/ablation"

        # Generate config into training/config/ so launch.py can find it
        config_name = f"ablation_{exp_name}"
        generate_exp_config(
            exp_name, exp_def,
            args.co3d_dir, args.anno_dir,
            args.checkpoint_path,
            output_dir,
            config_name=config_name,
        )

        # launch.py uses --config <name> to load training/config/<name>.yaml
        cmd = [
            "torchrun",
            "--nproc_per_node", str(args.nproc),
            "--master_port", str(args.master_port),
            "training/launch.py",
            "--config", config_name,
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        start_time = time.time()
        if not args.dry_run:
            try:
                # Write experiment info
                info_path = os.path.join(
                    output_dir, exp_name, "experiment_info.json"
                )
                os.makedirs(os.path.dirname(info_path), exist_ok=True)
                with open(info_path, "w") as f:
                    json.dump({
                        "experiment_name": exp_name,
                        "description": exp_def["description"],
                        "loss_config": {k: v for k, v in exp_def["loss_config"].items()},
                    }, f, indent=2)

                subprocess.run(cmd, check=True)
                status = "completed"
            except subprocess.CalledProcessError as e:
                logger.error(f"Experiment {exp_name} failed: {e}")
                status = "failed"
        else:
            status = "dry_run"

        elapsed = time.time() - start_time
        results[exp_name] = {
            "status": status,
            "elapsed_sec": elapsed,
            "description": exp_def["description"],
        }

        logger.info(f"Experiment {exp_name}: {status} ({elapsed:.1f}s)")

    # Save summary
    summary_path = f"{output_dir}/ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nAblation summary saved to {summary_path}")
    logger.info("Analysis: Compare tensorboard logs to determine which loss is removable.")
    logger.info("Look for:")
    logger.info("  1. Camera loss vs Depth loss: which contributes more to final performance?")
    logger.info("  2. Gradient loss: does removing it hurt depth quality?")
    logger.info("  3. Quality-efficiency tradeoff: which loss gives best perf/FLOPs?")


def run_inference(args):
    """Run out-of-sample inference."""
    logger.info("=" * 60)
    logger.info("Running OUT-OF-SAMPLE INFERENCE")
    logger.info("=" * 60)

    image_dir = args.image_dir
    if not image_dir or not os.path.isdir(image_dir):
        logger.error(f"Image directory not found: {image_dir}")
        return

    from vggt.utils.load_fn import load_and_preprocess_images
    import torch
    import numpy as np
    from PIL import Image

    # Load model
    logger.info("Loading VGGT model...")
    from vggt.models.vggt import VGGT

    model = VGGT(
        enable_camera=True,
        enable_depth=True,
        enable_point=True,
        enable_track=True,
    )

    # Load checkpoint if provided
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        logger.info(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

    model = model.cuda()
    model.eval()

    # Find images
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith(image_extensions)]
    )

    if len(image_files) < 2:
        logger.error(f"Need at least 2 images for inference, found {len(image_files)}")
        return

    logger.info(f"Found {len(image_files)} images for inference")

    # Load and preprocess
    images = []
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        img = np.array(Image.open(img_path).convert("RGB"))
        images.append(img)

    # Use the model's preprocessing
    images_tensor, _ = load_and_preprocess_images(images, mode="pad")
    images_tensor = images_tensor.cuda()

    # Run inference
    logger.info("Running inference...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            predictions = model(images=images_tensor)

    # Save results
    output_dir = args.output_dir or "inference_output"
    os.makedirs(output_dir, exist_ok=True)

    # Extract and save predictions
    pose_enc = predictions.get("pose_enc")
    depth = predictions.get("depth")
    depth_conf = predictions.get("depth_conf")
    world_points = predictions.get("world_points")
    world_points_conf = predictions.get("world_points_conf")

    np.savez(
        os.path.join(output_dir, "predictions.npz"),
        pose_enc=pose_enc.cpu().numpy() if pose_enc is not None else None,
        depth=depth.cpu().numpy() if depth is not None else None,
        depth_conf=depth_conf.cpu().numpy() if depth_conf is not None else None,
        world_points=world_points.cpu().numpy() if world_points is not None else None,
        world_points_conf=world_points_conf.cpu().numpy() if world_points_conf is not None else None,
    )

    # Generate visualizations
    if depth is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        for i in range(min(depth.shape[1], 4)):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Original image
            axes[0].imshow(images[min(i, len(images)-1)])
            axes[0].set_title(f"Input Image {i}")
            axes[0].axis("off")

            # Depth map
            depth_np = depth[0, i, ..., 0].cpu().numpy()
            im = axes[1].imshow(depth_np, cmap="inferno")
            axes[1].set_title(f"Predicted Depth {i}")
            axes[1].axis("off")
            plt.colorbar(im, ax=axes[1])

            # Confidence
            conf_np = depth_conf[0, i].cpu().numpy()
            im2 = axes[2].imshow(conf_np, cmap="viridis", vmin=0, vmax=1)
            axes[2].set_title(f"Depth Confidence {i}")
            axes[2].axis("off")
            plt.colorbar(im2, ax=axes[2])

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"depth_vis_{i}.png"), dpi=150)
            plt.close()

    logger.info(f"Inference complete! Results saved to {output_dir}")
    logger.info("Files:")
    logger.info(f"  - {output_dir}/predictions.npz (all numerical predictions)")
    logger.info(f"  - {output_dir}/depth_vis_*.png (depth visualizations)")
    logger.info("\nTo visualize camera poses and 3D points, use:")
    logger.info(f"  python demo_viser.py --image_dir {image_dir}")


def check_hardware():
    """Check and report hardware specifications."""
    logger.info("=" * 60)
    logger.info("HARDWARE CHECK")
    logger.info("=" * 60)

    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_mem / (1024 ** 3)
                logger.info(f"GPU {i}: {props.name}")
                logger.info(f"  Memory: {mem_gb:.1f} GB")
                logger.info(f"  Compute Capability: {props.major}.{props.minor}")
                logger.info(f"  SMs: {props.multi_processor_count}")
    except ImportError:
        logger.warning("PyTorch not installed, cannot check hardware")

    try:
        from training.train_utils.hardware_monitor import estimate_gpu_memory
        from vggt.models.vggt import VGGT

        logger.info("\nEstimating memory for VGGT training...")
        model = VGGT(enable_camera=True, enable_depth=True, enable_point=False, enable_track=False)

        # Full params
        est_full = estimate_gpu_memory(model, img_size=518, batch_size=1, seq_len=12)
        logger.info(f"\nFull-parameter fine-tuning (bf16):")
        logger.info(f"  Model: {est_full['model_params_gb']:.1f} GB")
        logger.info(f"  Optimizer: {est_full['optimizer_states_gb']:.1f} GB")
        logger.info(f"  Gradients: {est_full['gradients_gb']:.1f} GB")
        logger.info(f"  Activations (est): {est_full['activations_est_gb']:.1f} GB")
        logger.info(f"  TOTAL (est): {est_full['total_est_gb']:.1f} GB")
        logger.info(f"  Trainable params: {est_full['param_stats']['trainable'] / 1e6:.1f}M")

        # With frozen backbone
        logger.info(f"\nWith frozen aggregator (LoRA-like head tuning):")
        logger.info(f"  ~{est_full['total_est_gb'] * 0.3:.1f} GB (approximate)")

        logger.info(f"\nRECOMMENDATION:")
        if est_full['total_est_gb'] > 24:
            logger.info(f"  ❌ 4090 (24GB) CANNOT support full-param fine-tuning")
            logger.info(f"  ✅ H100 (80GB) or A100 (40/80GB) REQUIRED for full-param")
            logger.info(f"  ✅ 4090 can run with frozen backbone or small batch sizes")
        else:
            logger.info(f"  ✅ 4090 (24GB) CAN support full-param fine-tuning")

    except Exception as e:
        logger.warning(f"Could not estimate memory: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="VGGT CV Course Assignment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check hardware capability
  python run_experiments.py --mode check_hardware

  # Full fine-tuning
  python run_experiments.py --mode full_finetune \\
      --co3d_dir /data/co3d --anno_dir /data/co3d_anno \\
      --checkpoint_path /path/to/vggt_checkpoint.pt

  # Ablation experiments
  python run_experiments.py --mode ablation \\
      --co3d_dir /data/co3d --anno_dir /data/co3d_anno

  # Inference on custom images
  python run_experiments.py --mode inference \\
      --image_dir /path/to/my_images --checkpoint_path /path/to/checkpoint.pt
        """,
    )

    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["full_finetune", "ablation", "inference", "check_hardware"],
        help="Experiment mode"
    )
    parser.add_argument("--co3d_dir", type=str, help="Path to Co3D dataset directory")
    parser.add_argument("--anno_dir", type=str, help="Path to Co3D annotations directory")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to pre-trained VGGT checkpoint")
    parser.add_argument("--image_dir", type=str, help="Path to images for inference")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--nproc", type=int, default=1,
                        help="Number of GPUs to use")
    parser.add_argument("--master_port", type=int, default=29500,
                        help="Master port for distributed training")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")

    args = parser.parse_args()

    if args.mode == "check_hardware":
        check_hardware()
    elif args.mode == "full_finetune":
        if not args.co3d_dir or not args.anno_dir:
            parser.error("--co3d_dir and --anno_dir are required for full_finetune")
        run_full_finetune(args)
    elif args.mode == "ablation":
        if not args.co3d_dir or not args.anno_dir:
            parser.error("--co3d_dir and --anno_dir are required for ablation")
        run_ablation_experiments(args)
    elif args.mode == "inference":
        if not args.image_dir:
            parser.error("--image_dir is required for inference")
        run_inference(args)


if __name__ == "__main__":
    main()
