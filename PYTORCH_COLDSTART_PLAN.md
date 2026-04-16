# Docker + CRIU 加速 PyTorch 推理冷启动技术方案

## 1. 问题分析：PyTorch 推理冷启动耗时拆解

### 1.1 冷启动全流程与耗时分布

PyTorch 推理服务从零启动到可服务请求，经历以下阶段。以下耗时数据基于社区公开基准测试和典型生产环境（GPU: NVIDIA A100/H100, 模型规模: 7B-70B 参数），按阶段量化：

| 阶段 | 操作内容 | 典型耗时 | 占比 |
|------|---------|----------|------|
| **T1: Python 解释器启动** | CPython 初始化、site-packages 扫描 | 0.1-0.3s | 1-2% |
| **T2: import torch** | 加载 torch._C (C 扩展 ~600MB)、注册算子、初始化 CUDA 运行时懒加载框架 | 1.5-5s | 10-25% |
| **T3: CUDA Context 初始化** | 首次 .to("cuda") 触发 CUDA context 创建、驱动初始化、cuDNN handle 创建 | 1-8s | 8-30% |
| **T4: 模型权重加载** | torch.load / safetensors 反序列化、CPU 内存分配 | 2-20s | 25-50% |
| **T5: 权重传输到 GPU** | model.to("cuda") 触发 PCIe/NVLink 数据传输 | 1-10s | 10-30% |
| **T6: Warmup / 编译** | 首次推理触发 kernel 编译、autotune、内存池初始化 | 0.5-5s | 5-15% |
| **合计** | | **6-48s** | 100% |

**关键观察**：

1. **import torch 的开销不可忽视**。社区实测 `import torch` 在 CPU 环境下需要 1.5-2s，在 GPU 环境下因为触发 CUDA 运行时懒加载可达 3-5s（来源：PyTorch GitHub Issue #97106, StackOverflow 实测数据）。瓶颈在 torch._C 的 C 扩展加载和 torch._meta_registrations 的算子注册。

2. **CUDA Context 初始化是隐含成本**。PyTorch 采用 CUDA lazy initialization，首次调用 `torch.cuda.init()` 或 `.to("cuda")` 时才创建 CUDA context，在 Jetson 等嵌入式平台上实测可达 4-8s（来源：NVIDIA Developer Forums）。在服务器 GPU 上通常 1-3s。

3. **模型权重加载是最大瓶颈**。对于 7B 参数模型（~14GB FP16），从 SSD 加载约需 5-8s；70B 模型可达 30-60s。使用 safetensors 的 mmap 模式和 `torch.device("meta")` 延迟初始化可将加载时间压缩约 40%（来源：PyTorch Serve Issue #2571）。

4. **Warmup 开销与 torch.compile 强相关**。如果使用 torch.compile，冷启动编译时间可达 10-67s（来源：PyTorch Issue #172547）。即使使用 compile cache，首次加载仍需 1-3s。

### 1.2 冷启动耗时的场景差异

| 场景 | 模型规模 | 典型总耗时 | 瓶颈阶段 |
|------|---------|-----------|----------|
| CPU 小模型推理 (ResNet-50) | ~25M params | 2-4s | import torch + model load |
| GPU 中等模型 (LLaMA-7B) | ~7B params | 15-25s | weight load + GPU transfer |
| GPU 大模型 (LLaMA-70B) | ~70B params | 60-120s | weight load (I/O bound) |
| torch.compile 场景 | 任意 | 额外 +10-60s | JIT compilation |
| Serverless / Lambda | 小模型量化 | 3-8s | import torch + CUDA init |

### 1.3 CRIU 方案的目标收益区间

CRIU checkpoint/restore 的核心价值是**跳过 T2-T6 的全部或部分阶段**。收益取决于方案能覆盖多少阶段：

- 仅跳过 import torch（T2）：节省 1.5-5s
- 跳过 import + model load 到 CPU（T2+T4）：节省 3.5-25s
- 跳过全部 CPU 侧初始化（T2+T4）：节省 3.5-25s，但 T3+T5 仍需执行
- 跳过全部阶段含 GPU（T2-T6）：节省 6-48s（理想情况）

---

## 2. CRIU 方案可行性论证

### 2.1 CRIU 对 Python 进程的 checkpoint/restore 可靠性

**结论：对纯 CPU Python 进程，CRIU checkpoint/restore 是可靠且经过验证的。**

论据：

1. **本项目实验已验证**：test_app.py 的 counter 变量在 checkpoint/restore 后精确恢复，证明 CRIU 能正确处理 Python 进程的堆、栈、全局变量状态。

2. **社区广泛验证**：CRIU 自 2011 年起用于 Linux 容器迁移，是 Podman、LXC、Kubernetes (cri-o) 的标准检查点方案。Python 进程作为标准 Linux 进程，其匿名内存、线程、文件描述符、管道等均可被 CRIU 正确捕获和恢复。

3. **Python 运行时的特殊性**：Python 进程的内存模型对 CRIU 而言并无特殊之处——CPython 的对象堆、GIL、帧栈均存储在进程的匿名内存页中，属于 CRIU 的标准处理范围。CRIU 不需要理解 Python 语义，它只关心内核可见的进程状态。

4. **已知限制**：CRIU 要求恢复时的 PID 一致（可通过 PID namespace 解决），且无法处理某些内核资源（如 timerfd、eventfd 在旧内核上的状态）。对于 Python 进程的典型工作负载，这些限制极少触发。

### 2.2 PyTorch Tensor 在 CRIU restore 后是否可用

**结论：CPU tensor 可用，GPU tensor 不可用（除非配合 cuda-checkpoint）。**

#### 2.2.1 CPU Tensor

PyTorch 的 CPU tensor 底层由 `torch.UntypedStorage` 管理，其数据存储在进程的匿名内存页中（对于小 tensor）或通过 mmap 映射的文件页中（对于使用 mmap 模式加载的 safetensors）。

- **匿名内存 tensor**：CRIU 完整捕获匿名内存页，restore 后 tensor 数据指针有效，tensor 完全可用。
- **mmap 映射的 tensor**：CRIU 会记录 mmap 的文件路径和偏移量。恢复前提是原文件仍存在于相同路径。对于容器内的模型文件，只要容器的文件系统层未变，这不会是问题。
- **共享内存 tensor**：PyTorch DataLoader 的共享内存（`/dev/shm/torch_*`）在 CRIU restore 时需要特殊处理。CRIU 默认会尝试恢复共享内存段，但需要 `--shm-size` 配置与 checkpoint 时一致。

#### 2.2.2 GPU Tensor

PyTorch 的 GPU tensor 数据存储在 GPU 显存（VRAM）中，CPU 侧只持有指针和元数据。CRIU 原生**完全无法感知 GPU 显存**的存在：

- GPU 显存不属于进程的虚拟地址空间中的任何可 checkpoint 的区域
- CUDA context 包含驱动侧状态（stream、event、handle），这些由 NVIDIA 内核驱动管理
- `cudaMalloc` 分配的内存对 CRIU 而言是不可见的

因此，**对包含 GPU tensor 的进程执行 CRIU restore，进程会恢复运行但所有 GPU 相关操作将崩溃或返回错误**——因为 GPU 侧的状态已完全丢失。

### 2.3 GPU 状态（CUDA Context、GPU Tensor）能否被保存

**结论：可以，但需要 NVIDIA cuda-checkpoint 工具配合，且有严格的驱动版本和硬件要求。**

这是 CRIU + PyTorch 方案最关键的可行性问题。详细分析如下：

#### 2.3.1 原生 CRIU：完全不支持 GPU

CRIU 只处理 CPU 侧的进程状态（内存页、线程、文件描述符等）。GPU 设备文件（`/dev/nvidia*`）和 GPU 内存映射不在 CRIU 的处理范围内。使用原生 CRIU 对含 GPU 操作的进程做 checkpoint，要么直接失败（无法 dump external device file），要么 restore 后进程因 GPU 状态丢失而崩溃。

#### 2.3.2 NVIDIA cuda-checkpoint：GPU 状态保存的唯一可行方案

NVIDIA 于 2024 年发布了 cuda-checkpoint 工具（https://github.com/NVIDIA/cuda-checkpoint），它通过驱动级 API 实现 CUDA 状态的透明保存和恢复：

**工作原理**：

1. **Lock**：锁定 CUDA driver API，阻止新的 GPU 操作，等待已有 GPU 工作完成
2. **Checkpoint**：将 GPU 显存数据拷贝到主机内存，释放所有 GPU 资源（进程不再直接引用 GPU 硬件）
3. **CRIU dump**：此时进程变成了"纯 CPU 进程"，CRIU 可以正常 checkpoint
4. **CRIU restore**：恢复 CPU 侧状态
5. **Restore**：重新获取 GPU 资源，将主机内存中的 GPU 数据拷贝回显存
6. **Unlock**：解锁 CUDA API，进程恢复正常运行

**关键前提条件**：

| 条件 | 要求 |
|------|------|
| NVIDIA 驱动版本 | >= 550（基础功能），>= 570（推荐，含 NVML 支持） |
| CRIU 版本 | >= 4.0（含 CUDA plugin） |
| GPU 架构 | 仅 x86_64 |
| CUDA IPC | 不支持（DataLoader num_workers > 1 会报错） |
| UVM (Unified Virtual Memory) | 不支持 |
| 多 GPU 通信 (NCCL) | 不支持（单进程单 GPU） |
| 部分设备透传 | 需 >= 575.57.08 驱动 |
| pinned memory | 存在已知问题 |

**已验证的 PyTorch 支持状态**：

根据 NVIDIA/cuda-checkpoint Issue #4 的讨论：
- 2024 年 4 月：d4l3k 首次尝试 PyTorch，CUDA 状态可保存但 CRIU dump 遇到问题
- 2024 年 10 月：NVIDIA 官方确认 PyTorch 支持在路线图上
- 2025 年 1 月：rst0git 确认 **PyTorch 的 checkpoint/restore 在驱动版本 570.86.10 下工作正常**
- 2025 年 7 月：**Baseten 公开报告**使用 CRIU + cuda-checkpoint 将 ML 推理冷启动从 32s 降至 6s

**生产环境验证**：

- CRIUgpu 论文（2025 年 2 月，arxiv 2502.16631）：LLaMA 3.1 (8B) on H100，checkpoint 77s，restore 39s
- Modal：GPU memory snapshots（基于 gVisor + cuda-checkpoint API），声称冷启动加速最高 12x，但目前仍为 Alpha 状态，用户反馈稳定性约 10% 失败率
- Baseten：基于 Podman + CRIU + cuda-checkpoint，冷启动从 32s 降至 6s

#### 2.3.3 可行性判断矩阵

| 场景 | CRIU 原生 | CRIU + cuda-checkpoint | 判断 |
|------|-----------|----------------------|------|
| CPU 推理 | 完全可行 | 不需要 | 可行，推荐 |
| GPU 推理（仅保存 CPU 侧权重） | 可行 | 不需要 | 可行，但 restore 后需 .to(cuda) |
| GPU 推理（保存全部 GPU 状态） | 不可行 | 条件可行 | 条件可行，需满足驱动和硬件要求 |

---

## 3. 架构设计

基于可行性分析，提出三个方案，按复杂度递增排列。

### 3.1 方案 A：纯 CPU 推理 -- CRIU 快照跳过 import + model load

#### 3.1.1 设计思路

将模型加载到 CPU 内存后执行 CRIU checkpoint，恢复时直接从已加载模型的内存状态继续，跳过 import torch 和 model.load() 的全部耗时。

#### 3.1.2 架构图

```
[初始化阶段]
  Docker 容器启动
    -> Python 解释器启动
    -> import torch (1.5-5s)
    -> model = load_model() (2-20s)
    -> model.eval()
    -> CRIU checkpoint (创建快照)

[运行阶段]
  CRIU restore (0.5-2s)
    -> 进程从已加载模型状态恢复
    -> 直接开始推理
```

#### 3.1.3 时序对比

| 步骤 | 无 CRIU | 有 CRIU |
|------|---------|---------|
| Python 启动 | 0.2s | - (已包含在快照中) |
| import torch | 1.5-5s | - (已包含在快照中) |
| model.load() | 2-20s | - (已包含在快照中) |
| model.eval() | 0.1s | - (已包含在快照中) |
| CRIU restore | - | 0.5-2s |
| **首次推理可用** | **4-26s** | **0.5-2s** |

#### 3.1.4 评估

| 维度 | 评价 |
|------|------|
| 可行性 | **高** -- 本项目实验已验证 CRIU 对简单 Python 进程的可靠性，CPU tensor 在 CRIU restore 后完全可用 |
| 复杂度 | **低** -- 无需特殊硬件或驱动，标准 CRIU 即可 |
| 收益 | **中等** -- 节省 4-26s，但仅限 CPU 推理场景 |
| 适用场景 | CPU 推理、边缘设备、无需 GPU 的模型（如小型 NLP 模型） |

### 3.2 方案 B：GPU 推理 + 延迟 GPU 加载 -- CRIU 快照保存 CPU 侧状态

#### 3.2.1 设计思路

Checkpoint 时只保存 CPU 侧状态（import torch + model weights 在 CPU），restore 后再将模型 `.to("cuda")`。这样跳过了 import torch 和 model.load 的耗时，但 CUDA context 初始化和 GPU 传输仍需在 restore 后执行。

此方案的灵感来自 Modal 的 CPU memory snapshot 方案：在 `snap=True` 阶段将模型加载到 CPU，在 `snap=False` 阶段将模型移到 GPU。

#### 3.2.2 架构图

```
[初始化阶段]
  Docker 容器启动
    -> import torch
    -> model = load_model()  # 加载到 CPU
    -> model.eval()
    -> CRIU checkpoint

[恢复阶段]
  CRIU restore (0.5-2s)
    -> 进程恢复，model 在 CPU 内存中
    -> model = model.to("cuda")  # 触发 CUDA init + 数据传输 (2-10s)
    -> 开始推理
```

#### 3.2.3 时序对比

| 步骤 | 无 CRIU | 方案 B |
|------|---------|--------|
| import torch | 1.5-5s | - (快照中) |
| model.load() | 2-20s | - (快照中) |
| CRIU restore | - | 0.5-2s |
| .to("cuda") | 1-10s | 1-10s |
| CUDA warmup | 0.5-1s | 0.5-1s |
| **首次推理可用** | **5-36s** | **2-13s** |

#### 3.2.4 关键实现细节

1. **checkpoint 时机**：必须在 model 加载到 CPU 后、调用 .to("cuda") 之前做 checkpoint。一旦调用 .to("cuda")，CUDA context 被创建，CRIU 原生无法正确处理。

2. **restore 后的 GPU 初始化**：需要在 restore 后显式调用 `.to("cuda")`。这要求在代码中检测"是否从 checkpoint 恢复"的状态，可通过环境变量或特定标志文件判断。

3. **CUDA lazy initialization 的风险**：PyTorch 2.x 默认使用 CUDA lazy loading。在某些版本中，`import torch` 可能不立即创建 CUDA context，但 `torch.cuda.is_available()` 的调用会触发。必须在 checkpoint 前确保 CUDA context 未被创建。

#### 3.2.5 评估

| 维度 | 评价 |
|------|------|
| 可行性 | **高** -- CPU 侧状态可靠保存，GPU 侧在 restore 后重新初始化是成熟的操作模式 |
| 复杂度 | **中** -- 需要管理 checkpoint/restore 生命周期，实现两阶段初始化 |
| 收益 | **中等** -- 跳过 import + load，但 CUDA init + GPU transfer 仍需执行 |
| 适用场景 | GPU 推理、NVIDIA 驱动 < 550 或无法安装 cuda-checkpoint 的环境 |

### 3.3 方案 C：GPU 推理 + CUDA 状态缓存 -- CRIU + cuda-checkpoint 全状态快照

#### 3.3.1 设计思路

使用 CRIU + NVIDIA cuda-checkpoint，将 CPU 状态和 GPU 状态（CUDA context、GPU tensor、kernel 编译缓存）统一保存为快照。restore 时一次性恢复全部状态，跳过所有初始化阶段。

这是最理想的方案，能最大化冷启动加速效果，但对环境要求最严格。

#### 3.3.2 架构图

```
[初始化阶段]
  Docker 容器启动
    -> import torch
    -> model = load_model()
    -> model = model.to("cuda")
    -> model.eval()
    -> warmup inference (可选)
    -> cuda-checkpoint --toggle --pid $PID  # 挂起 CUDA 状态
    -> CRIU checkpoint (含 CPU 状态 + CUDA plugin 处理 GPU 状态)

[恢复阶段]
  CRIU restore (含 CUDA plugin)
    -> CPU 状态恢复
    -> cuda-checkpoint --toggle --pid $PID  # 恢复 CUDA 状态
    -> 进程从完整状态恢复，可直接推理
```

#### 3.3.3 实际操作流程（基于 Podman + CRIU + cuda-checkpoint）

根据 Baseten 公开的技术方案和 NVIDIA 官方文档，完整流程如下：

```bash
# 1. 启动含 GPU 的容器
sudo podman run -d --device nvidia.com/gpu=all \
  --security-opt seccomp=unconfined \
  criu-pytorch-gpu:latest

# 2. 等待模型加载和 warmup 完成
# ... (模型加载代码运行完毕)

# 3. 创建包含 GPU 状态的检查点
sudo podman container checkpoint \
  --export /path/to/checkpoint.tar \
  --tcp-established \
  $CONTAINER_NAME

# 4. 恢复容器
sudo podman container restore \
  --import /path/to/checkpoint.tar \
  --tcp-established \
  $CONTAINER_NAME
```

CRIU 4.0+ 的 CUDA plugin 会自动调用 cuda-checkpoint 工具，对用户透明。

#### 3.3.4 时序对比

| 步骤 | 无 CRIU | 方案 C |
|------|---------|--------|
| import torch | 1.5-5s | - (快照中) |
| model.load() | 2-20s | - (快照中) |
| .to("cuda") | 1-10s | - (快照中) |
| CUDA warmup | 0.5-5s | - (快照中) |
| CRIU restore + CUDA restore | - | 2-15s (取决于 GPU 内存大小) |
| **首次推理可用** | **5-40s** | **2-15s** |

Baseten 的实测数据（Stable Diffusion 模型）：
- 冷启动（无快照）：32s（weight loading 18s + CUDA init 8s + kernel compilation 6s）
- 快照恢复：6s
- **加速比：5.3x**

#### 3.3.5 评估

| 维度 | 评价 |
|------|------|
| 可行性 | **条件可行** -- 需要 NVIDIA 驱动 >= 570、CRIU >= 4.0、cuda-checkpoint 工具可用，且不支持 IPC/UVM/NCCL |
| 复杂度 | **高** -- 环境依赖严格，调试困难，需 Podman 或特定 CRI 配置 |
| 收益 | **最高** -- 跳过全部冷启动阶段，加速比 3-12x |
| 适用场景 | GPU 推理服务、Serverless ML、对冷启动延迟敏感的生产环境 |

### 3.4 方案对比总结

| 维度 | 方案 A (纯 CPU) | 方案 B (CPU 快照 + GPU 延迟加载) | 方案 C (全状态快照) |
|------|----------------|------------------------------|-------------------|
| 跳过的阶段 | T2+T4 | T2+T4 | T2+T3+T4+T5+T6 |
| 典型加速比 | 2-10x (CPU) | 2-3x (GPU) | 3-12x (GPU) |
| 环境要求 | 标准 CRIU | 标准 CRIU | CRIU 4.0+ + cuda-checkpoint + 驱动 >= 570 |
| GPU 支持 | 无 | 部分 (restore 后 .to(cuda)) | 完整 |
| 实现复杂度 | 低 | 中 | 高 |
| 稳定性 | 高 | 高 | 中 (Alpha 阶段) |
| 多 GPU 支持 | N/A | 单 GPU | 单 GPU (不支持 NCCL/IPC) |
| 推荐优先级 | P0 (验证基础) | P1 (实用方案) | P2 (理想方案，待生态成熟) |

---

## 4. 实现细节

### 4.1 Docker 镜像设计

#### 4.1.1 方案 A/B 基础镜像

```dockerfile
# 基于 PyTorch 官方镜像，避免自行安装 CUDA 依赖
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# 预安装推理框架
RUN pip install --no-cache-dir \
    transformers \
    safetensors \
    flask

# 拷贝推理代码和模型文件
COPY inference_server.py .
COPY models/ /app/models/

# 推理服务入口
CMD ["python3", "inference_server.py"]
```

设计要点：
- 使用 PyTorch 官方镜像而非自建，确保 CUDA 运行时版本一致
- 模型文件打包在镜像中（适用于固定模型场景），或通过 Volume 挂载（适用于多模型场景）
- 不在镜像中安装 CRIU -- CRIU 运行在宿主机

#### 4.1.2 方案 C 镜像（含 GPU 支持）

```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# 方案 C 需要容器内可访问 cuda-checkpoint 工具
# cuda-checkpoint 需要从 NVIDIA 仓库获取
COPY --from=nvidia/cuda-checkpoint:latest /bin/cuda-checkpoint /usr/bin/

RUN pip install --no-cache-dir \
    transformers \
    safetensors \
    flask

COPY inference_server.py .
COPY models/ /app/models/

CMD ["python3", "inference_server.py"]
```

### 4.2 Checkpoint 时机选择

| 时机 | 保存内容 | 优势 | 劣势 |
|------|---------|------|------|
| import torch 完成后 | Python 运行时 + torch 模块 | 最小快照体积 | 仅节省 import 时间，收益有限 |
| model load 完成后 (CPU) | Python + torch + model weights (CPU) | 跳过 import + load | 快照体积 = 模型大小 + 运行时 |
| model.to("cuda") 完成后 | Python + torch + model weights (CPU+GPU) | 跳过全部初始化 | 需 cuda-checkpoint，快照巨大 |
| warmup 完成后 | 上述 + kernel cache + autotune 结果 | 跳过 warmup | 快照体积最大 |

**推荐策略**：
- 方案 A/B：在 model load 完成后（CPU 侧）做 checkpoint
- 方案 C：在 warmup 完成后做 checkpoint，最大化收益

### 4.3 推理服务集成方式

#### 4.3.1 方案一：Flask/FastAPI + CRIU restore

```
请求流程：
1. 容器从 CRIU 快照恢复
2. Flask/FastAPI 服务已在快照中就绪，直接监听端口
3. 请求到达，直接推理
```

优势：最简单，restore 后服务立即可用。
劣势：快照中保存了 HTTP 服务的 socket 状态，恢复时需要原端口可用；TCP 连接状态需要 `--tcp-established` 参数。

#### 4.3.2 方案二：preload + fork 模式

```
1. 主进程完成 import torch + model load
2. CRIU checkpoint 保存主进程状态
3. 需要新推理实例时：CRIU restore -> fork -> 子进程处理请求
4. 主进程继续待命
```

优势：避免重复 restore，多个推理请求可共享同一个 restore 实例。
劣势：实现复杂，fork 后 GPU 资源共享存在冲突。

#### 4.3.3 方案三：生命周期钩子模式（参考 Modal）

```python
# 参考 Modal 的 snap=True/snap=False 生命周期设计
class InferenceServer:
    @checkpoint_hook(phase="pre_checkpoint")  # 对应 snap=True
    def load_model(self):
        """在 checkpoint 前执行，状态会被保存"""
        import torch
        self.model = torch.load("model.pt", map_location="cpu")
        self.model.eval()

    @checkpoint_hook(phase="post_restore")  # 对应 snap=False
    def move_to_gpu(self):
        """在 restore 后执行，初始化 GPU 侧状态"""
        self.model = self.model.to("cuda")
        # warmup
        dummy = torch.randn(1, 3, 224, 224, device="cuda")
        self.model(dummy)
```

**推荐方案三**，因为它最灵活，可以适配方案 A/B/C 的所有场景。

### 4.4 多模型场景下的快照管理策略

#### 4.4.1 快照命名与版本管理

```
/var/lib/criu-snapshots/
  model-resnet50-v2.5-cpu/         # 方案 A/B
    checkpoint-20240101/
      inventory.img
      stats.img
      pages-1.img
      ...
  model-llama7b-v2.5-gpu/          # 方案 C
    checkpoint-20240101/
      inventory.img
      stats.img
      pages-1.img
      cuda-*.img                    # GPU 状态文件
      ...
```

#### 4.4.2 快照体积预估

| 模型 | 参数量 | CPU 快照体积 | GPU 快照体积 (含 CUDA) |
|------|--------|-------------|----------------------|
| ResNet-50 | 25M | ~100MB | ~300MB |
| BERT-base | 110M | ~440MB | ~1.2GB |
| LLaMA-7B | 7B | ~14GB | ~28GB |
| LLaMA-70B | 70B | ~140GB | ~280GB |

注：CPU 快照体积约等于模型权重 + Python 运行时开销；GPU 快照体积约等于 CPU 快照 + GPU 显存数据（模型权重在 CPU 和 GPU 各一份）。

#### 4.4.3 多模型调度策略

1. **按需恢复**：根据请求动态选择对应的快照进行 restore，适合模型种类多但并发低的场景
2. **预恢复池**：维护一个预热实例池，restore 后的实例保持待命，适合高并发场景
3. **分层快照**：公共基础快照（import torch）+ 模型增量快照，减少存储开销

---

## 5. 潜在问题与解决方案

### 5.1 Python GIL 状态恢复

**问题**：CRIU checkpoint 时 Python GIL 的状态（哪个线程持有、等待队列）能否正确恢复？

**分析**：Python GIL 是一个 pthread mutex，存储在进程的匿名内存中。CRIU 恢复进程的内存映射和线程状态时，mutex 的内部状态（locked/unlocked、owner）会被一并恢复。

**风险**：如果 checkpoint 时某线程持有 GIL 且正在执行 Python 字节码，restore 后该线程继续从断点执行，GIL 状态一致。但如果 checkpoint 发生在 C 扩展释放 GIL 的窗口期（如 CUDA 操作期间），restore 后可能出现 GIL 状态与实际线程状态不一致。

**解决方案**：
1. 在 checkpoint 前确保 Python 代码处于"稳定状态"——例如主线程在一个 `time.sleep()` 或 `select()` 调用中，GIL 已释放
2. 对于方案 A/B，在 `input()` 或 `while True: time.sleep(1)` 循环中做 checkpoint，GIL 状态可预测
3. 对于方案 C，cuda-checkpoint 的 lock 操作会先等待 CUDA 操作完成，此时线程通常处于 GIL 释放状态

**可行性判断**：在可控的 checkpoint 时机下，GIL 状态恢复**可靠**。

### 5.2 随机数生成器状态

**问题**：PyTorch 使用 CPU 和 GPU 两套随机数生成器（RNG）。CRIU restore 后 RNG 状态是否一致？

**分析**：
- CPU RNG：状态存储在进程内存中，CRIU 可正确恢复
- GPU RNG：状态存储在 GPU 显存中，原生 CRIU 无法恢复；cuda-checkpoint 可恢复

**影响**：
- 推理场景：随机数状态不影响推理结果（deterministic 模式下），影响较小
- 训练场景：随机数状态不一致会导致梯度计算错误，**不可用**

**解决方案**：
1. 推理模式下，在 restore 后显式设置 `torch.manual_seed(SEED)` 和 `torch.cuda.manual_seed(SEED)` 重置 RNG
2. 训练场景不建议使用 CRIU 快照（应使用 PyTorch 原生的 checkpoint 机制）

### 5.3 文件描述符和 Socket 状态

**问题**：CRIU restore 后，文件描述符和 socket 能否正确恢复？

**分析**：

| 资源类型 | CRIU 行为 | 注意事项 |
|---------|----------|---------|
| 普通文件 FD | 恢复文件路径和偏移量 | 文件必须在相同路径存在，内容不变 |
| /dev/shm 共享内存 | 尝试恢复 | 需 `--shm-size` 一致 |
| TCP socket | 默认断开 | 需 `--tcp-established` 保留 |
| Unix socket | 恢复 | 路径必须存在 |
| Pipe | 恢复 | 缓冲区数据保留 |

**解决方案**：
1. 推理服务在 checkpoint 前关闭所有非必要的 TCP 连接，避免 `--tcp-established` 的复杂性
2. 确保模型文件在容器文件系统中的路径不变（镜像层不变）
3. checkpoint 前清空日志文件的缓冲区（`flush()`）

### 5.4 内存占用（每个 checkpoint 的存储开销）

**问题**：每个 CRIU checkpoint 的磁盘开销是多少？

**分析**：

CRIU checkpoint 的体积由以下部分组成：
1. **内存页**：进程的 RSS（Resident Set Size），包括匿名页和文件映射页
2. **元数据**：进程树、FD 表、信号状态等（通常 < 10MB）
3. **GPU 状态**（方案 C）：GPU 显存数据拷贝到主机内存后的体积

典型内存占用：

| 组件 | 内存占用 |
|------|---------|
| Python 解释器 | ~30-50MB |
| torch._C 模块 | ~600MB |
| 模型权重 (7B FP16) | ~14GB |
| CUDA context + cuDNN | ~500MB-2GB |
| GPU 显存 (7B FP16) | ~14GB |

**方案 A/B 的快照体积**：约 Python 运行时 + 模型权重 = 15-20GB (7B 模型)
**方案 C 的快照体积**：约 30-40GB (7B 模型，含 CPU 和 GPU 数据)

**解决方案**：
1. 使用 `--checkpoint-dir` 指定快照存储到高吞吐 SSD
2. 对于方案 C，关注 NVIDIA Issue #33 的进展——cuda-checkpoint 计划支持直接将 GPU 数据写入文件描述符而非主机内存，避免内存翻倍
3. 考虑快照压缩（CRIU 本身不压缩，但可以对快照目录做 tar + zstd）

### 5.5 容器重启后 CUDA 重新初始化的问题

**问题**：方案 C 中，restore 后 CUDA 状态被恢复，但如果容器被分配到不同的 GPU 或 GPU 驱动版本不同，会发生什么？

**分析**：
- cuda-checkpoint 的 restore 要求 GPU 型号一致（至少计算能力一致）
- NVIDIA 驱动版本必须一致
- GPU 的 UUID 不需要一致，但 PCI 总线拓扑可能影响恢复

**解决方案**：
1. 确保同一快照只在相同 GPU 型号的节点上恢复
2. 在 Kubernetes 环境中使用 node selector / node affinity 约束调度
3. 对于异构 GPU 集群，为每种 GPU 型号维护独立的快照集

### 5.6 进程 PID 依赖问题

**问题**：CRIU restore 要求恢复进程使用与 checkpoint 时相同的 PID，这在容器间可能冲突。

**解决方案**：
1. 使用 Docker/Podman 的 PID namespace 隔离，容器内 PID 空间独立
2. 使用 `criu-ns` 工具在独立的 PID namespace 中执行 restore
3. 在 Kubernetes 中，containerd/CRI-O 已集成 PID namespace 管理

### 5.7 PyTorch CUDA pinned memory 问题

**问题**：cuda-checkpoint 目前对 pinned memory（cudaMallocHost 分配的内存）支持存在问题（来源：NVIDIA/cuda-checkpoint Issue #4，NVIDIA 开发者 sgurfinkel 的评论）。

**影响**：PyTorch 的 DataLoader 使用 pinned memory 加速 CPU-GPU 数据传输。如果 DataLoader 配置了 `pin_memory=True`，checkpoint 可能失败。

**解决方案**：
1. 在 checkpoint 前确保没有活跃的 DataLoader 使用 pinned memory
2. 将 `pin_memory` 设为 False（牺牲少量传输性能换取 checkpoint 兼容性）
3. 等待 NVIDIA 修复此问题（已在路线图上）

---

## 6. 性能预期

### 6.1 各方案预估冷启动时间

以 LLaMA-7B (FP16, ~14GB) 在 A100 GPU 上为例：

| 阶段 | 无 CRIU | 方案 A (CPU) | 方案 B (CPU快照+GPU延迟) | 方案 C (全状态快照) |
|------|---------|-------------|------------------------|-------------------|
| import torch | 3s | 0 | 0 | 0 |
| model.load() | 8s | 0 | 0 | 0 |
| .to("cuda") | 3s | N/A | 3s | 0 |
| CUDA warmup | 1s | N/A | 1s | 0 |
| CRIU restore | 0 | 2s | 2s | 5s |
| CUDA state restore | 0 | 0 | 0 | 3s |
| **总计** | **15s** | **2s** | **6s** | **8s** |
| **加速比** | 1x | 7.5x | 2.5x | 1.9x |

> 注：方案 C 的 restore 时间包含 GPU 内存数据从主机内存拷贝回显存的时间。对于 14GB 的 GPU 内存，A100 的 PCIe 4.0 x16 理论带宽 32GB/s，实际约 12-15GB/s，拷贝约需 1-2s。加上 CRIU CPU 状态恢复（2-3s）和 CUDA state restore（1-2s），总计约 5-8s。

以 ResNet-50 (CPU 推理) 为例：

| 阶段 | 无 CRIU | 方案 A |
|------|---------|--------|
| import torch | 2s | 0 |
| model.load() | 0.5s | 0 |
| model.eval() | 0.1s | 0 |
| CRIU restore | 0 | 0.3s |
| **总计** | **2.6s** | **0.3s** |
| **加速比** | 1x | **8.7x** |

### 6.2 Checkpoint/Restore 操作本身的耗时

| 操作 | 方案 A/B (CPU) | 方案 C (CPU+GPU) |
|------|---------------|-----------------|
| Checkpoint 创建 | 1-5s (取决于 RSS) | 10-80s (取决于 GPU 内存大小) |
| Restore 执行 | 0.5-3s | 2-40s (取决于 GPU 内存大小) |
| 快照体积 | 模型大小 + 运行时 | 模型大小 * 2 + 运行时 + CUDA 状态 |

CRIUgpu 论文数据（H100 GPU）：

| 模型 | GPU 数 | Checkpoint 时间 | Restore 时间 | 快照体积 |
|------|--------|----------------|-------------|---------|
| LLaMA 3.1 8B | 1x H100 | 77s | 39s | 56GB |
| - | 2x A100 | 26s | 17s | - |
| - | 4x A100 | 55s | 35s | - |

### 6.3 与其他冷启动优化方案的对比

| 方案 | 冷启动加速 | 侵入性 | 适用场景 | 限制 |
|------|-----------|--------|---------|------|
| **CRIU 方案 A (CPU)** | 5-10x | 低 | CPU 推理 | 不支持 GPU |
| **CRIU 方案 B (CPU快照+GPU延迟)** | 2-3x | 中 | GPU 推理 | GPU 初始化仍需时间 |
| **CRIU 方案 C (全状态快照)** | 3-12x | 高 | GPU 推理 | 驱动/硬件要求严格 |
| **safetensors + mmap** | 1.5-2x | 低 | 通用 | 仅优化 load 阶段 |
| **torch.device("meta") + load_state_dict** | 2-3x | 低 | GPU 推理 | 仅优化 load 阶段 |
| **torch.compile + cache** | 减少 warmup 7x | 中 | 编译场景 | 首次编译仍慢 |
| **模型量化 (INT8/INT4)** | 2-4x (load) | 中 | 对精度不敏感 | 可能降低模型质量 |
| **NVIDIA Run:ai Model Streamer** | 2-3x (load) | 低 | GPU 推理 | 仅优化 I/O |
| **vLLM / TensorRT-LLM** | 整体优化 | 高 | LLM 推理 | 特定模型类型 |
| **AOTInductor (torch.export)** | 减少 compile 10x+ | 中 | 编译场景 | 仅消除编译开销 |
| **Modal GPU Snapshots** | 3-12x | 低 (平台集成) | Modal 平台 | 平台锁定 |
| **Baseten BDN** | 2-3x | 低 (平台集成) | Baseten 平台 | 平台锁定 |

**关键洞察**：CRIU 方案的核心优势在于**跳过 import torch 阶段**，这是其他优化方案无法覆盖的盲区。safetensors、量化、Model Streamer 等方案都只优化模型加载阶段，但 import torch 的 1.5-5s 开销始终存在。CRIU 是唯一能消除此开销的方案。

---

## 7. 实施路线图

### 7.1 第一阶段：方案 A 验证（预计 1-2 周）

目标：验证 CRIU 对 PyTorch CPU 推理进程的 checkpoint/restore 可靠性。

关键任务：
1. 构建 PyTorch CPU 推理 Docker 镜像
2. 在容器内运行 import torch + model.load() + CPU 推理
3. 执行 CRIU checkpoint（通过 docker checkpoint create）
4. 从 checkpoint restore 并验证：
   - model 对象是否可用
   - tensor 数据是否完整
   - 推理结果是否与 checkpoint 前一致

验证标准：
- restore 后推理结果与正常启动完全一致
- restore 时间 < 3s
- 多次 restore 无内存泄漏

### 7.2 第二阶段：方案 B 验证（预计 1-2 周）

目标：验证 CPU 快照 + GPU 延迟加载方案的可行性。

关键任务：
1. 在方案 A 的基础上，实现两阶段初始化
2. checkpoint 前确保 CUDA context 未创建
3. restore 后执行 .to("cuda") 并验证推理正确性

验证标准：
- restore 后 .to("cuda") 正常执行
- GPU 推理结果正确
- restore + .to("cuda") 总时间 < 无 CRIU 的冷启动时间

### 7.3 第三阶段：方案 C 验证（预计 2-4 周）

目标：验证 CRIU + cuda-checkpoint 全状态快照方案。

前置条件：
- NVIDIA 驱动 >= 570
- CRIU >= 4.0 并安装 CUDA plugin
- cuda-checkpoint 工具可用

关键任务：
1. 在 GPU 节点上安装 CRIU + cuda-checkpoint
2. 运行 GPU 推理进程（import + load + .to("cuda") + warmup）
3. 通过 CRIU CUDA plugin 执行 checkpoint
4. restore 并验证 GPU 状态完整性

验证标准：
- restore 后无需 .to("cuda")，直接可推理
- GPU tensor 数据完整
- restore 时间 < 冷启动时间的 50%

### 7.4 第四阶段：推理服务集成（预计 2-3 周）

目标：将 CRIU 方案集成到实际推理服务中。

关键任务：
1. 实现 Flask/FastAPI 推理服务
2. 实现生命周期钩子（pre_checkpoint / post_restore）
3. 实现快照管理脚本（创建/恢复/清理/版本管理）
4. 端到端性能测试

---

## 8. 风险评估

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|---------|
| cuda-checkpoint 与特定 PyTorch 版本不兼容 | 中 | 高 | 锁定 PyTorch + 驱动版本组合，提前测试 |
| CRIU restore 后 Python 状态异常 | 低 | 高 | 可控的 checkpoint 时机，增加 post_restore 校验 |
| GPU 快照体积过大导致 I/O 瓶颈 | 高 | 中 | 使用高速 NVMe SSD，关注 FD-based dump 特性进展 |
| 多 GPU / NCCL 不支持 | 确定 | 高（大模型场景） | 限制单 GPU 场景，或等待 NVIDIA 后续支持 |
| 驱动升级导致快照失效 | 中 | 中 | 驱动升级后重建快照，快照版本与驱动版本绑定 |
| 快照恢复到不同 GPU 型号失败 | 确定 | 中 | 快照标注 GPU 型号，调度时匹配 |
| checkpoint 过程中的服务中断 | 确定 | 低 | 使用 `--leave-running` 或在低峰期执行 |
| 安全风险（快照包含内存数据） | 确定 | 中 | 快照文件权限控制，加密存储 |

---

## 9. 结论

### 9.1 核心结论

1. **CRIU + PyTorch CPU 推理是完全可行的**。方案 A 的技术风险极低，收益明确（5-10x 冷启动加速），建议作为首选实施目标。

2. **CRIU + PyTorch GPU 推理（方案 B）是实用且可靠的**。通过 CPU 快照 + restore 后 .to("cuda") 的两阶段模式，可实现 2-3x 加速，且无需特殊驱动或工具。

3. **CRIU + cuda-checkpoint 全状态快照（方案 C）已从实验阶段进入早期生产阶段**。Baseten、Modal 等平台已在生产环境中使用，但仍有显著限制（单 GPU、无 IPC/UVM/NCCL、驱动版本严格），且稳定性仍需提升。建议在 6-12 个月后、NVIDIA 驱动 580+ 发布后再大规模采用。

4. **CRIU 方案的最大独特价值是跳过 import torch**。这是其他优化方案无法覆盖的。对于需要频繁创建/销毁推理实例的 Serverless 场景，CRIU 方案具有不可替代的优势。

### 9.2 建议的执行策略

**短期（1-3 个月）**：实施方案 A + B，建立 CRIU 加速 PyTorch 冷启动的基础能力。

**中期（3-6 个月）**：在 GPU 驱动 >= 570 的环境中试点方案 C，验证全状态快照的稳定性和性能。

**长期（6-12 个月）**：跟随 NVIDIA cuda-checkpoint 的功能演进（IPC/UVM/多 GPU 支持），逐步扩展到更复杂的模型场景。

### 9.3 与现有实验的关系

本项目已有的 CRIU + Docker 宿主机端方案验证，为 PyTorch 冷启动加速方案提供了坚实的基础：

1. **CRIU 安装和 Docker 实验特性配置已就绪**
2. **docker checkpoint create / start --checkpoint 操作流程已验证**
3. **seccomp=unconfined 等安全配置已测试**

下一步只需将 test_app.py 替换为 PyTorch 推理脚本，即可开始方案 A 的验证。
