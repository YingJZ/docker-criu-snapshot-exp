"""
PyTorch CPU 推理进程 — 用于裸机 CRIU dump/restore profiling
后台运行，写入 PID 文件，等待 CRIU checkpoint
"""

import torch
import torchvision.models as models
import time
import os
import signal
import sys
import json

RESULT_DIR = "/tmp/criu-pytorch-results"
PID_FILE = "/tmp/criu-pytorch.pid"
TRIGGER_FILE = "/tmp/criu-pytorch-trigger"

os.makedirs(RESULT_DIR, exist_ok=True)

print(f"[INIT] 进程启动! PID: {os.getpid()}", flush=True)

with open(PID_FILE, "w") as f:
    f.write(str(os.getpid()))
print(f"[INIT] PID 已写入 {PID_FILE}", flush=True)

phase1_start = time.time()
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()
phase1_time = time.time() - phase1_start
param_count = sum(p.numel() for p in model.parameters())
model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (
    1024 * 1024
)
print(
    f"[INIT] 模型加载完成: {param_count:,} params, {model_size_mb:.1f} MB, {phase1_time:.3f}s",
    flush=True,
)

torch.manual_seed(42)
dummy_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(dummy_input)
baseline_result = {
    "shape": list(output.shape),
    "sum": output.sum().item(),
    "mean": output.mean().item(),
}
with open(os.path.join(RESULT_DIR, "baseline.json"), "w") as f:
    json.dump(baseline_result, f, indent=2)
print(f"[INIT] 基线推理完成, output sum: {baseline_result['sum']:.6f}", flush=True)

timing = {
    "phase1_model_load_s": phase1_time,
    "param_count": param_count,
    "model_size_mb": model_size_mb,
}
with open(os.path.join(RESULT_DIR, "timing.json"), "w") as f:
    json.dump(timing, f, indent=2)

print(f"[READY] 模型已加载，进入等待状态。PID={os.getpid()}", flush=True)
print(f"[READY] 可执行 CRIU dump: sudo criu dump -t {os.getpid()} ...", flush=True)

counter = 0
while True:
    counter += 1
    if os.path.exists(TRIGGER_FILE):
        t0 = time.time()
        print(f"[TRIGGER] 执行推理验证...", flush=True)
        torch.manual_seed(42)
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        infer_time = time.time() - t0
        result = {
            "shape": list(out.shape),
            "sum": out.sum().item(),
            "mean": out.mean().item(),
        }
        match = abs(result["sum"] - baseline_result["sum"]) < 1e-5
        result["match"] = match
        result["infer_time_s"] = round(infer_time, 6)
        with open(os.path.join(RESULT_DIR, "verification.json"), "w") as f:
            json.dump(result, f, indent=2)
        os.remove(TRIGGER_FILE)
        print(f"[TRIGGER] 推理完成, match={match}, infer_time={infer_time:.3f}s, counter={counter}", flush=True)
    if counter % 5 == 0:
        print(f"[RUNNING] 计数: {counter}", flush=True)
    time.sleep(0.05)
