"""
PyTorch CPU 推理冷启动基准测试
精确测量从 Python 启动到首次推理可用的每个阶段耗时
使用延迟 import 来精确测量 import torch 的耗时
"""

import time
import os
import json
import sys
import subprocess

RESULT_DIR = "/tmp/criu-pytorch-results"
os.makedirs(RESULT_DIR, exist_ok=True)
pid = os.getpid()

t_python_ready = time.time()

t_import_start = time.time()
import torch

t_import_end = time.time()

t_tv_import_start = time.time()
import torchvision.models as models

t_tv_import_end = time.time()

t_model_load_start = time.time()
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()
t_model_load_end = time.time()

t_first_infer_start = time.time()
torch.manual_seed(42)
dummy_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(dummy_input)
t_first_infer_end = time.time()

t_second_infer_start = time.time()
torch.manual_seed(42)
dummy_input2 = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output2 = model(dummy_input2)
t_second_infer_end = time.time()


def read_proc_status():
    try:
        with open(f"/proc/{pid}/status") as f:
            status = {}
            for line in f:
                parts = line.strip().split(":", 1)
                if len(parts) == 2:
                    status[parts[0].strip()] = parts[1].strip()
            return status
    except:
        return {}


proc_status = read_proc_status()

import_torch_s = t_import_end - t_import_start
import_torchvision_s = t_tv_import_end - t_tv_import_start
model_load_s = t_model_load_end - t_model_load_start
first_infer_s = t_first_infer_end - t_first_infer_start
second_infer_s = t_second_infer_end - t_second_infer_start
total_from_script_s = t_first_infer_end - t_python_ready

results = {
    "pid": pid,
    "import_torch_s": round(import_torch_s, 6),
    "import_torchvision_s": round(import_torchvision_s, 6),
    "model_load_s": round(model_load_s, 6),
    "first_inference_s": round(first_infer_s, 6),
    "second_inference_s": round(second_infer_s, 6),
    "total_from_script_start_s": round(total_from_script_s, 6),
    "VmRSS_kb": proc_status.get("VmRSS", "N/A"),
    "VmSize_kb": proc_status.get("VmSize", "N/A"),
    "VmData_kb": proc_status.get("VmData", "N/A"),
    "Threads": proc_status.get("Threads", "N/A"),
}

print("=" * 60)
print("PyTorch CPU 冷启动基准测试结果")
print("=" * 60)
print(f"import torch:                {import_torch_s:.3f}s")
print(f"import torchvision:          {import_torchvision_s:.3f}s")
print(f"模型加载 (ResNet-50):        {model_load_s:.3f}s")
print(f"首次推理 (含 input 构造):    {first_infer_s:.3f}s")
print(f"第二次推理:                  {second_infer_s:.3f}s")
print(f"---")
print(f"脚本内总耗时 (到首次推理):   {total_from_script_s:.3f}s")
print(f"---")
print(f"VmRSS:    {results['VmRSS_kb']}")
print(f"VmSize:   {results['VmSize_kb']}")
print(f"Threads:  {results['Threads']}")
print("=" * 60)

output_path = os.path.join(RESULT_DIR, "coldstart_bench.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"结果已保存: {output_path}")
