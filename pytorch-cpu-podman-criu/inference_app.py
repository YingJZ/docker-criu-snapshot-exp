"""
PyTorch CPU 推理服务 -- Podman 版：纯CPU推理 + CRIU快照加速冷启动

流程:
  1. 启动时执行 import torch + 模型加载 + 基线推理（耗时操作，被快照跳过）
  2. 进入稳定等待循环，等待 CRIU checkpoint
  3. restore 后进程从等待循环继续，通过文件触发器执行推理验证

触发推理验证方式:
  - 文件触发: podman exec <container> touch /app/trigger_inference
  - 信号触发: podman exec <container> kill -USR1 1
"""

import torch
import torch.nn as nn
import time
import os
import json
import signal
import sys

RESULT_DIR = "/app/results"
BASELINE_FILE = os.path.join(RESULT_DIR, "baseline.json")
VERIFICATION_FILE = os.path.join(RESULT_DIR, "verification.json")
TRIGGER_FILE = "/app/trigger_inference"
TIMING_FILE = os.path.join(RESULT_DIR, "timing.json")


class SmallCNN(nn.Module):
    """轻量级CNN，用于无 torchvision 环境下的基础验证"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_model():
    """创建并返回模型，优先使用 torchvision ResNet-50"""
    try:
        import torchvision.models as models

        print("[INIT] 使用 torchvision ResNet-50 模型", flush=True)
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.eval()
        input_shape = (1, 3, 224, 224)
        return model, input_shape
    except ImportError:
        print("[INIT] torchvision 未安装，使用自定义小型CNN", flush=True)
        model = SmallCNN()
        model.eval()
        input_shape = (1, 3, 32, 32)
        return model, input_shape
    except Exception as e:
        print(f"[INIT] ResNet-50 加载失败 ({e})，回退到自定义小型CNN", flush=True)
        print("[INIT] 提示: 可预先下载权重文件到 models/ 目录", flush=True)
        print("[INIT]   wget -O models/resnet50-11ad3fa6.pth https://download.pytorch.org/models/resnet50-11ad3fa6.pth", flush=True)
        model = SmallCNN()
        model.eval()
        input_shape = (1, 3, 32, 32)
        return model, input_shape


def run_inference(model, input_shape):
    torch.manual_seed(42)
    dummy_input = torch.randn(*input_shape)
    with torch.no_grad():
        output = model(dummy_input)
    return {
        "shape": list(output.shape),
        "sum": output.sum().item(),
        "mean": output.mean().item(),
        "std": output.std().item(),
        "first5": output.flatten()[:5].tolist(),
        "last5": output.flatten()[-5:].tolist(),
    }


def save_result(result, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def load_result(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def compare_results(baseline, restored):
    diffs = {
        "sum_diff": abs(baseline["sum"] - restored["sum"]),
        "mean_diff": abs(baseline["mean"] - restored["mean"]),
        "std_diff": abs(baseline["std"] - restored["std"]),
    }
    first5_diff = max(
        abs(a - b) for a, b in zip(baseline["first5"], restored["first5"])
    )
    last5_diff = max(abs(a - b) for a, b in zip(baseline["last5"], restored["last5"]))
    diffs["first5_max_diff"] = first5_diff
    diffs["last5_max_diff"] = last5_diff

    tolerance = 1e-5
    match = all(v < tolerance for v in diffs.values())

    return {
        "match": match,
        "tolerance": tolerance,
        "diffs": diffs,
        "baseline_shape": baseline["shape"],
        "restored_shape": restored["shape"],
        "shapes_match": baseline["shape"] == restored["shape"],
    }


verification_result = None


def sigusr1_handler(signum, frame):
    global verification_result
    print("[SIGNAL] 收到 SIGUSR1，执行推理验证...", flush=True)
    verification_result = do_verify()


signal.signal(signal.SIGUSR1, sigusr1_handler)


def do_verify():
    try:
        baseline = load_result(BASELINE_FILE)
        restored = run_inference(model, input_shape)
        report = compare_results(baseline, restored)
        report["baseline"] = baseline
        report["restored"] = restored
        save_result(report, VERIFICATION_FILE)

        if report["match"]:
            print(
                f"[VERIFY] 推理结果一致! 最大差异: {max(report['diffs'].values()):.2e}",
                flush=True,
            )
        else:
            print(
                f"[VERIFY] 推理结果不一致! 差异详情: {report['diffs']}", flush=True
            )

        return report
    except Exception as e:
        error_report = {"match": False, "error": str(e)}
        save_result(error_report, VERIFICATION_FILE)
        print(f"[VERIFY] 验证过程出错: {e}", flush=True)
        return error_report


# ============================================================
# 主流程
# ============================================================

print(f"[INIT] 进程启动! PID: {os.getpid()}", flush=True)

phase1_start = time.time()
model, input_shape = create_model()

param_count = sum(p.numel() for p in model.parameters())
model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (
    1024 * 1024
)

phase1_time = time.time() - phase1_start
print(f"[INIT] 模型加载完成", flush=True)
print(f"[INIT]   参数量: {param_count:,}", flush=True)
print(f"[INIT]   模型大小: {model_size_mb:.1f} MB", flush=True)
print(f"[INIT]   输入形状: {input_shape}", flush=True)
print(f"[INIT]   耗时: {phase1_time:.3f}s", flush=True)

phase2_start = time.time()

baseline_result = run_inference(model, input_shape)
save_result(baseline_result, BASELINE_FILE)

phase2_time = time.time() - phase2_start
print(f"[BASELINE] 基线推理完成，耗时: {phase2_time:.3f}s", flush=True)
print(f"[BASELINE]   output sum: {baseline_result['sum']:.6f}", flush=True)
print(f"[BASELINE]   output mean: {baseline_result['mean']:.6f}", flush=True)
print(f"[BASELINE]   output shape: {baseline_result['shape']}", flush=True)

save_result(
    {
        "phase1_model_load_s": phase1_time,
        "phase2_baseline_inference_s": phase2_time,
        "total_init_s": phase1_time + phase2_time,
        "param_count": param_count,
        "model_size_mb": model_size_mb,
    },
    TIMING_FILE,
)

print("", flush=True)
print("[READY] ========================================", flush=True)
print("[READY] 模型已加载，基线推理已完成", flush=True)
print("[READY] 进程进入等待状态，可执行 CRIU checkpoint", flush=True)
print("[READY] ========================================", flush=True)
print(f"[READY] 触发推理验证:", flush=True)
print(f"[READY]   方式1: podman exec <container> touch {TRIGGER_FILE}", flush=True)
print(f"[READY]   方式2: podman exec <container> kill -USR1 1", flush=True)
print("", flush=True)

counter = 0
while True:
    counter += 1

    if os.path.exists(TRIGGER_FILE):
        print(f"[TRIGGER] 检测到触发文件 {TRIGGER_FILE}，执行推理验证...", flush=True)
        if os.path.exists(VERIFICATION_FILE):
            os.remove(VERIFICATION_FILE)
        do_verify()
        os.remove(TRIGGER_FILE)
        print(f"[TRIGGER] 验证完成，触发文件已删除", flush=True)

    if counter % 5 == 0:
        print(f"[RUNNING] 计数: {counter}", flush=True)

    time.sleep(1)
