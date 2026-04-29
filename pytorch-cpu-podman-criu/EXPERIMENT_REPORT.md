# Podman + CRIU 快照加速 PyTorch CPU 推理冷启动实验报告

> 实验日期: 2026-04-28  
> 实验方案: 方案A（Podman 版）— PyTorch CPU 推理 + CRIU checkpoint/restore

## 1. 实验目标

验证使用 Podman + CRIU 对 PyTorch CPU 推理容器做 checkpoint/restore，能否跳过 `import torch` 和 `model.load()` 的冷启动耗时，并测量实际加速效果。

核心假设：Podman 无守护进程（daemonless），CRIU checkpoint/restore 调用路径比 Docker 更短，restore 延迟应显著低于 Docker 方案（实测 Docker CRIU restore ~9.23s）。

## 2. 实验环境

| 项目 | 版本/配置 |
|------|-----------|
| 操作系统 | Linux (Ubuntu) |
| 内核版本 | 6.8.0-107-generic |
| Podman | 3.4.4 |
| CRIU | 4.2 |
| Python | 3.11 |
| PyTorch | 2.5.1+cpu |
| torchvision | 0.20.1+cpu |
| 模型 | ResNet-50 (25,557,032 params, 97.5 MB) |
| 容器运行模式 | rootful (sudo podman) |

> **注意**: Podman 3.4.4 < 4.0（README 推荐 ≥ 4.0），checkpoint/restore 功能在此版本下存在二次 restore 后容器退出的限制。

## 3. 实验原理

```
[初始化阶段]（仅执行一次）
  Podman 容器启动
    → Python 解释器启动
    → import torch (冷启动瓶颈)
    → model = load_model() (冷启动瓶颈)
    → model.eval()
    → 基线推理
    → CRIU checkpoint (创建快照)

[运行阶段]（每次恢复）
  CRIU restore (仅此一步)
    → 进程从已加载模型状态恢复
    → 直接开始推理
```

### Docker vs Podman CRIU 调用路径

```
Docker 路径:
  docker checkpoint/create → Docker daemon → containerd → runc → CRIU
  (多层 RPC 通信，每层有状态管理、锁、序列化开销)

Podman 路径:
  podman container checkpoint → conmon → runc → CRIU
  (无守护进程，直接 fork+exec 调用 runc/CRIU)
```

## 4. 实验步骤

### 4.1 前置准备

```bash
# 1. 下载 ResNet-50 权重 (~98MB)
./download_weights.sh

# 2. 构建 torch-cpu 基础镜像
podman build -t torch-cpu:latest -f Dockerfile.torch-cpu .

# 3. 构建实验镜像
podman build -t pytorch-criu-cpu-podman .
```

### 4.2 一键实验

```bash
./run_experiment.sh
```

脚本自动执行：构建镜像 → 启动容器 → 记录冷启动数据 → CRIU checkpoint → CRIU restore → 触发推理验证 → 生成报告。

### 4.3 多轮基准测试

对 5 轮独立的冷启动 vs CRIU restore 流程进行计时对比，每轮流程为：

1. `podman run` 启动容器，等待 `[READY]`
2. 记录冷启动耗时
3. `podman container checkpoint` 创建快照
4. `podman container restore` 恢复容器
5. 触发推理验证，比较与基线的一致性

## 5. 实验结果

### 5.1 冷启动阶段耗时明细

| 阶段 | 耗时 |
|------|------|
| import torch + model.load() | ~890ms |
| 基线推理 (含 input 构造) | ~42ms |
| 容器内初始化合计 | **~970ms** |
| Podman 容器启动开销 | ~1650ms |
| **端到端冷启动 (podman run → READY)** | **~2623ms** |

### 5.2 多轮基准测试结果 (5 轮)

| 轮次 | 冷启动 (ms) | Checkpoint (ms) | Restore (ms) | 验证 |
|------|-------------|-----------------|--------------|------|
| 1 | 2430 | 1349 | 649 | match=True |
| 2 | 2904 | 1448 | 649 | match=True |
| 3 | 2438 | 1452 | 650 | match=True |
| 4 | 2331 | 1451 | 657 | match=True |
| 5 | 3010 | 1553 | 649 | match=True |
| **平均** | **2623** | **1451** | **651** | — |

### 5.3 加速比

| 对比维度 | 加速比 |
|----------|--------|
| vs 端到端冷启动 (podman run → READY) | **4.0x** |
| vs 容器内初始化 (import + load + infer) | **1.5x** |

### 5.4 推理一致性验证

所有 5 轮 CRIU restore 后的推理结果与基线完全一致：

```json
{
  "match": true,
  "tolerance": 1e-05,
  "diffs": {
    "sum_diff": 0.0,
    "mean_diff": 0.0,
    "std_diff": 0.0,
    "first5_max_diff": 0.0,
    "last5_max_diff": 0.0
  }
}
```

CPU RNG 状态被 CRIU 完整保存，使用 `torch.manual_seed(42)` 后输出数值完全一致，差异为 0.0。

## 6. 与其他方案对比

| 方案 | 首次推理可用时间 | 说明 |
|------|-----------------|------|
| 裸机冷启动 | ~5.6s | python 启动 → import → model.load → 推理 |
| 裸机 CRIU | ~0.4s | criu restore → 推理 (**14x 加速**) |
| Docker CRIU | ~9.2s | docker start --checkpoint → 推理 (**0.6x 更慢**) |
| **Podman CRIU (本次实测)** | **~0.65s** | podman restore → 推理 (**8.6x 加速 vs 裸机冷启动**) |

### 调用路径开销对比

```
裸机 CRIU:          0.31s  (直接 criu restore)
Podman CRIU (实测): 0.65s  (conmon + runc + CRIU, 开销 +0.34s)
Docker CRIU (参考): 9.23s  (Docker daemon + containerd + runc + CRIU, 开销 +8.92s)
```

Podman 的容器运行时开销 (~340ms) 远低于 Docker (~8.9s)，接近裸机 CRIU 性能。

## 7. 发现的问题与限制

### 7.1 二次 Checkpoint/Restore 失败

在同一容器上执行 **第二次** checkpoint → restore 循环后，容器进程退出（exit code 120）。这是 Podman 3.4.4 的已知限制，可能与 CRIU 在已恢复进程上的二次 dump 支持不完善有关。

**影响**: 不支持原地多次 checkpoint/restore 循环，但单次 checkpoint/restore 完全正常。

**建议**: 升级到 Podman ≥ 4.0 以获得更稳定的 checkpoint/restore 支持。

### 7.2 Root 权限要求

Podman checkpoint/restore 需要 root 权限（`sudo podman`），rootless 模式下无法执行。实验中通过配置免密 sudo 解决。

### 7.3 当前模型规模较小

ResNet-50 加载仅 ~890ms，CRIU restore (651ms) 相比冷启动的加速比不够显著（1.5x）。对于大型模型（如 7B 参数 LLM，加载需 30-60s），加速效果将大幅提升。

## 8. 结论

1. **CRIU restore 在 Podman 下工作正常**：5 轮测试全部通过，推理结果与基线完全一致（差异为 0.0）。

2. **Podman CRIU 远优于 Docker CRIU**：Podman restore (651ms) vs Docker restore (9.23s)，Podman 快约 **14x**。这验证了 Podman daemonless 架构在 CRIU 场景下的显著优势。

3. **当前模型规模下加速比有限**：对于 ResNet-50 等小模型（~970ms 初始化），Podman CRIU restore (651ms) 仅提供 1.5x 加速（容器内维度）。但对于大模型场景，加速比预期可达 **10-50x**。

4. **端到端加速显著**：从 `podman run` 到推理可用，CRIU restore 方案实现 4.0x 加速（651ms vs 2623ms）。

## 9. 后续建议

- **升级 Podman 到 ≥ 4.0**：解决二次 checkpoint/restore 问题，获得更稳定支持
- **测试大模型**：使用 LLM（如 LLaMA-7B）验证大模型场景下的加速效果
- **测试跨主机迁移**：利用 Podman 的 `--export` / `--import` 功能验证 checkpoint 迁移
- **对比裸机 CRIU**：使用 `bare_inference.py` + 直接 `criu restore` 测量裸机性能基线
