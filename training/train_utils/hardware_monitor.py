# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified for CV course assignment - hardware monitoring utilities.
"""
Hardware monitoring utilities for VGGT training.
Records GPU memory, utilization, temperature, and CPU/disk usage.
"""

import time
import threading
import logging
import json
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class HardwareMonitor:
    """
    Monitors and records hardware usage during training.

    Records GPU metrics (memory, utilization, temperature) and CPU metrics
    at configurable intervals. Saves summary to JSON for later analysis.

    Usage:
        monitor = HardwareMonitor(log_dir="logs/hardware")
        monitor.start()
        # ... training ...
        monitor.stop()
        monitor.save_summary()
    """

    def __init__(
        self,
        log_dir: str = "logs/hardware",
        interval_sec: float = 1.0,
        gpu_ids: Optional[List[int]] = None,
    ):
        self.log_dir = log_dir
        self.interval_sec = interval_sec
        self.gpu_ids = gpu_ids or [0]

        self.records: List[Dict] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.start_time = None

        os.makedirs(log_dir, exist_ok=True)

    def _get_gpu_info(self, gpu_id: int) -> Dict:
        """Collect GPU metrics using pynvml."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

            try:
                temperature = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except pynvml.NVMLError:
                temperature = -1

            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Watts
            except pynvml.NVMLError:
                power = -1.0

            try:
                clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            except pynvml.NVMLError:
                clock = -1

            return {
                "gpu_id": gpu_id,
                "memory_used_gb": mem_info.used / (1024 ** 3),
                "memory_total_gb": mem_info.total / (1024 ** 3),
                "memory_pct": (mem_info.used / mem_info.total) * 100,
                "gpu_util_pct": utilization.gpu,
                "mem_util_pct": utilization.memory,
                "temperature_c": temperature,
                "power_w": power,
                "sm_clock_mhz": clock,
            }
        except ImportError:
            # Fallback: use torch API
            import torch
            return {
                "gpu_id": gpu_id,
                "memory_used_gb": torch.cuda.memory_allocated(gpu_id) / (1024 ** 3),
                "memory_total_gb": torch.cuda.get_device_properties(gpu_id).total_mem / (1024 ** 3),
                "memory_pct": torch.cuda.memory_allocated(gpu_id) / torch.cuda.get_device_properties(gpu_id).total_mem * 100,
                "gpu_util_pct": -1,
                "mem_util_pct": -1,
                "temperature_c": -1,
                "power_w": -1,
                "sm_clock_mhz": -1,
            }

    def _get_cpu_info(self) -> Dict:
        """Collect CPU and system metrics."""
        import psutil

        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "cpu_percent": cpu_percent,
            "ram_used_gb": mem.used / (1024 ** 3),
            "ram_total_gb": mem.total / (1024 ** 3),
            "ram_pct": mem.percent,
            "disk_used_gb": disk.used / (1024 ** 3),
            "disk_total_gb": disk.total / (1024 ** 3),
            "disk_pct": disk.percent,
        }

    def _record_loop(self):
        """Background thread loop that records hardware metrics."""
        while not self._stop_event.is_set():
            elapsed = time.time() - self.start_time if self.start_time else 0

            record = {"timestamp": elapsed}

            for gpu_id in self.gpu_ids:
                gpu_info = self._get_gpu_info(gpu_id)
                for key, value in gpu_info.items():
                    record[f"gpu{gpu_id}_{key}"] = value

            try:
                cpu_info = self._get_cpu_info()
                record.update(cpu_info)
            except Exception:
                pass

            # Also record PyTorch peak memory
            try:
                import torch
                for gpu_id in self.gpu_ids:
                    record[f"gpu{gpu_id}_peak_allocated_gb"] = (
                        torch.cuda.max_memory_allocated(gpu_id) / (1024 ** 3)
                    )
                    record[f"gpu{gpu_id}_peak_reserved_gb"] = (
                        torch.cuda.max_memory_reserved(gpu_id) / (1024 ** 3)
                    )
            except Exception:
                pass

            self.records.append(record)
            self._stop_event.wait(self.interval_sec)

    def start(self):
        """Start background hardware monitoring."""
        self.start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        logger.info(f"Hardware monitor started. Logging to {self.log_dir}")

    def stop(self):
        """Stop hardware monitoring."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.info(f"Hardware monitor stopped. {len(self.records)} records collected.")

    def save_summary(self):
        """Save monitoring summary to JSON file."""
        if not self.records:
            logger.warning("No records to save.")
            return

        # Compute summary statistics
        summary = self._compute_summary()

        summary_path = os.path.join(self.log_dir, "hardware_summary.json")
        records_path = os.path.join(self.log_dir, "hardware_records.json")

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Save only every 10th record to keep file size manageable
        sampled_records = self.records[::max(1, len(self.records) // 1000)]
        with open(records_path, "w") as f:
            json.dump(sampled_records, f, indent=2, default=str)

        logger.info(f"Hardware summary saved to {summary_path}")
        logger.info(f"Hardware records (sampled) saved to {records_path}")

        return summary

    def _compute_summary(self) -> Dict:
        """Compute summary statistics from recorded data."""
        if not self.records:
            return {}

        # Numeric keys to summarize
        numeric_keys = set()
        for record in self.records:
            for key, value in record.items():
                if isinstance(value, (int, float)) and key != "timestamp":
                    numeric_keys.add(key)

        summary = {
            "num_records": len(self.records),
            "duration_sec": self.records[-1]["timestamp"] - self.records[0]["timestamp"],
        }

        for key in sorted(numeric_keys):
            values = [r[key] for r in self.records if key in r and r[key] >= 0]
            if values:
                summary[key] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "last": values[-1],
                }

        return summary

    def log_current(self, step: int, phase: str):
        """Log current hardware usage (can be called at logging intervals)."""
        if not self.records:
            return

        record = self.records[-1]
        gpu_mem = record.get(f"gpu0_memory_used_gb", "N/A")
        gpu_util = record.get(f"gpu0_gpu_util_pct", "N/A")
        gpu_temp = record.get(f"gpu0_temperature_c", "N/A")

        log_msg = (
            f"[Hardware] step={step}, phase={phase}, "
            f"GPU mem={gpu_mem:.1f}GB, "
            f"GPU util={gpu_util}%, "
            f"GPU temp={gpu_temp}°C"
        )
        logger.info(log_msg)


def get_model_param_count(model) -> Dict[str, int]:
    """
    Count model parameters in different categories.

    Returns:
        Dict with:
        - total: Total parameters
        - trainable: Trainable parameters
        - frozen: Frozen parameters
        - by_module: Dict mapping module name to param count
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    by_module = {}
    for name, param in model.named_parameters():
        module_name = name.split(".")[0]
        if module_name not in by_module:
            by_module[module_name] = {"total": 0, "trainable": 0}
        by_module[module_name]["total"] += param.numel()
        if param.requires_grad:
            by_module[module_name]["trainable"] += param.numel()

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "by_module": by_module,
    }


def estimate_gpu_memory(model, img_size=518, batch_size=1, seq_len=12,
                        dtype_size=2, optimizer_factor=3):
    """
    Estimate GPU memory usage for training.

    Args:
        model: The PyTorch model
        img_size: Image resolution
        batch_size: Batch size
        seq_len: Number of frames per sequence
        dtype_size: Bytes per parameter (2 for bf16, 4 for fp32)
        optimizer_factor: AdamW memory factor (~3x parameters)

    Returns:
        Dict with memory estimates in GB
    """
    params = get_model_param_count(model)
    trainable_params = params["trainable"]

    # Model parameters (bf16/fp32)
    model_mem = params["total"] * dtype_size / (1024 ** 3)

    # Optimizer states (AdamW: param + exp_avg + exp_avg_sq = 3x)
    optimizer_mem = trainable_params * dtype_size * optimizer_factor / (1024 ** 3)

    # Gradients
    grad_mem = trainable_params * dtype_size / (1024 ** 3)

    # Activations (rough estimate based on img size)
    # ViT activations scale with O(num_tokens * embed_dim * num_layers)
    num_patches = (img_size // 14) ** 2
    activations_est = batch_size * seq_len * num_patches * 1024 * 48 * 2 / (1024 ** 3)

    total_est = model_mem + optimizer_mem + grad_mem + activations_est

    return {
        "model_params_gb": model_mem,
        "optimizer_states_gb": optimizer_mem,
        "gradients_gb": grad_mem,
        "activations_est_gb": activations_est,
        "total_est_gb": total_est,
        "param_stats": params,
    }
