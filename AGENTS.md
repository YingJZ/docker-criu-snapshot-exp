# CRIU 容器快照恢复实验 — Agent 操作手册

## 项目概述

研究不同运行环境（原生 Linux / Podman / Docker）对 CRIU 快照启动性能的影响。核心问题：容器运行时的封装层（Docker daemon / Podman conmon）对 CRIU checkpoint/restore 延迟有多大开销？

## 目录结构

```
docker-criu-snapshot-exp/
├── AGENTS.md                    # 本文件 — Agent 操作手册
├── analyze_checkpoint.sh        # 检查点分解分析脚本（通用）
├── PYTORCH_COLDSTART_PLAN.md    # 实验规划文档
├── Dockerfile.torch-cpu         # 公共基础镜像 Dockerfile（PyTorch CPU）
│
├── pytorch-cpu-criu/            # 裸机 CRIU + Docker CRIU 实验
│   ├── bare_inference.py        # 裸机推理进程（供 CRIU 直接 dump/restore）
│   ├── bench_compare.sh         # 冷启动 vs CRIU restore 多轮对比基准测试
│   ├── coldstart_bench.py       # 冷启动基准测试（分阶段精确计时）
│   ├── inference_app.py         # 容器内推理脚本（Docker 版）
│   ├── Dockerfile               # Docker 实验镜像构建文件
│   ├── download_weights.sh      # 预下载 ResNet-50 权重
│   ├── run_experiment.sh        # Docker CRIU 一键实验脚本
│   ├── models/                  # 预下载的模型权重
│   ├── EXPERIMENT_RESULT.md     # 实验结果报告
│   ├── CRIU耗时长的原因分析.md  # Docker CRIU 9s 延迟根因分析
│   └── README.md
│
├── pytorch-cpu-podman-criu/     # Podman CRIU 实验
│   ├── bare_inference.py        # 裸机推理进程（同上）
│   ├── bench_compare.sh         # 冷启动 vs CRIU restore 对比测试（Podman 版）
│   ├── coldstart_bench.py       # 冷启动基准测试
│   ├── inference_app.py         # 容器内推理脚本（Podman 版）
│   ├── Dockerfile               # Podman 实验镜像构建文件
│   ├── download_weights.sh      # 预下载 ResNet-50 权重
│   ├── run_experiment.sh        # Podman CRIU 一键实验脚本
│   ├── models/                  # 预下载的模型权重
│   ├── EXPERIMENT_REPORT.md     # 实验报告
│   └── README.md
│
└── test-docker-criu/            # 早期 Docker CRIU 验证实验
    ├── Dockerfile
    ├── test_app.py
    ├── EXPERIMENT_RESULT.md
    └── README.md
```

## 环境准备

### CRIU 安装

```bash
sudo apt-get install -y criu
criu --version  # 建议 3.17+
```

### Docker 实验特性

```bash
# /etc/docker/daemon.json
{"experimental": true}
sudo systemctl restart docker
```

### Podman 安装

```bash
sudo apt-get install -y podman  # 建议 4.0+
podman --version
```

### Python 虚拟环境（裸机实验用）

```bash
cd pytorch-cpu-criu  # 或 pytorch-cpu-podman-criu
python3 -m venv .venv
.venv/bin/pip install torch torchvision
```

## 实验执行

### 跑裸机冷启动 + CRIU 对比 (pytorch-cpu-criu/)

此实验直接在宿主机运行 PyTorch 推理进程，用 CRIU 做 checkpoint/restore，绕过所有容器运行时。

```bash
cd pytorch-cpu-criu

# 1. 创建虚拟环境并安装依赖
python3 -m venv .venv
.venv/bin/pip install torch torchvision

# 2. 修改 bench_compare.sh 中的路径（VENV_PYTHON / SCRIPT_DIR）指向实际位置

# 3. 运行多轮对比基准测试
./bench_compare.sh
```

输出文件：
- `/tmp/criu-pytorch-results/comparison_summary.json` — 汇总对比数据（含 checkpoint_breakdown 字段）
- `/tmp/criu-pytorch-results/checkpoint_breakdown.json` — 检查点分解分析
- `/tmp/criu-pytorch-results/coldstart_bench_round*.json` — 各轮冷启动分阶段数据
- `/tmp/criu-pytorch-results/verification_round*.json` — 各轮 restore 后推理验证

### 跑 Docker CRIU (pytorch-cpu-criu/)

```bash
cd pytorch-cpu-criu

# 1. 预下载模型权重（国内网络推荐）
./download_weights.sh

# 2. 构建镜像
docker build -t pytorch-criu-cpu .

# 3. 运行一键实验
./run_experiment.sh
```

输出额外包含：
- `/tmp/criu-pytorch-results/checkpoint_breakdown_docker.json` — Docker 检查点分解分析

⚠️ **Docker 29 netns bug**：当前版本需要 `--network=host`，否则 checkpoint 后 restore 可能失败。

### 跑 Podman CRIU (pytorch-cpu-podman-criu/)

```bash
cd pytorch-cpu-podman-criu

# 1. 预下载模型权重
./download_weights.sh

# 2. 构建镜像
podman build -t pytorch-criu-cpu-podman .

# 3. 运行一键实验
./run_experiment.sh
```

Podman 无需开启实验特性，原生支持 checkpoint/restore。

输出额外包含：
- `/tmp/criu-pytorch-results/checkpoint_breakdown_podman.json` — Podman 检查点分解分析

## 结果解读

### comparison_summary.json 结构

```json
{
  "rounds": 5,
  "coldstart": {
    "avg_total_s": 5.632,
    "avg_import_torch_s": 1.823,
    "avg_import_torchvision_s": 0.412,
    "avg_model_load_s": 2.103,
    "avg_first_infer_s": 0.089,
    "per_round": [5.601, 5.632, ...]
  },
  "criu_restore": {
    "avg_restore_s": 0.312,
    "avg_infer_after_restore_s": 0.098,
    "avg_total_s": 0.410,
    "per_round_restore": [0.301, 0.312, ...],
    "per_round_infer": [0.102, 0.098, ...]
  },
  "speedup": "13.73x",
  "checkpoint_breakdown": { ... }
}
```

关键字段：
- `coldstart.avg_total_s` — 冷启动总耗时（从进程启动到首次推理可用）
- `criu_restore.avg_total_s` — CRIU restore + 首次推理总耗时
- `speedup` — 加速比 = coldstart / criu_total

### checkpoint_breakdown.json 结构

由 `analyze_checkpoint.sh` 生成，详细分解 CRIU 检查点目录的组成：

```json
{
  "checkpoint_dir": "/tmp/criu-dump-bare",
  "label": "bare-metal",
  "total_size_bytes": 307200000,
  "total_size_human": "292.97MB",
  "file_count": 47,
  "files": [
    {"name": "pages-1.img", "size_bytes": 268435456, "size_human": "256.00MB", "category": "pages"},
    {"name": "core-1.img", "size_bytes": 524288, "size_human": "512.00KB", "category": "core"},
    ...
  ],
  "category_sizes": {
    "pages": 268435456,
    "core": 524288,
    "mm": 32768,
    "fd": 8192,
    "net": 0,
    "cgroup": 4096,
    "proc": 2048,
    "inventory": 1024,
    "log": 65536,
    "other": 0
  },
  "crit_available": true,
  "crit_inventory": { ... },
  "process_info": {
    "VmRSS_kB": 423000,
    "VmSize_kB": 812345,
    "Threads": 4,
    "VMA_count": 67
  }
}
```

### 各分类含义

| 分类 | 匹配模式 | 含义 |
|------|----------|------|
| pages | `pages-*.img` | 进程内存页数据（最大组成部分，含模型权重 + Python 堆） |
| core | `core-*.img` | 进程核心状态（寄存器、信号、线程信息） |
| mm | `mm-*.img` | 内存映射描述（VMA 列表，不含实际数据） |
| fd | `files.img`, `fdinfo-*.img` | 文件描述符信息 |
| net | `netns-*.img` | 网络命名空间状态 |
| cgroup | `cgroup.img` | cgroup 配置 |
| proc | `pstree.img` | 进程树结构 |
| inventory | `inventory.img` | CRIU 元数据索引 |
| log | `*.log` | CRIU dump/restore 日志 |
| other | 其余文件 | 其他 CRIU 内部文件 |

### 如何判断 metadata vs pages 开销

- `pages` 占比 > 90% → 检查点体积主要由进程内存决定（正常），优化方向是减少内存占用
- `core` + `mm` + `fd` + `proc` 占比 > 20% → 元数据开销大，可能是多线程/多进程场景
- `log` 占比大 → 检查点目录中残留了多轮 CRIU 日志，可清理
- `net` / `cgroup` 占比大 → 容器环境额外开销（裸机实验通常为 0）

## 检查点分解分析

### 三种运行环境的分解输出

| 运行环境 | 输出文件 | 生成方式 |
|----------|----------|----------|
| 裸机 | `/tmp/criu-pytorch-results/checkpoint_breakdown.json` | `bench_compare.sh` 自动调用 `analyze_checkpoint.sh` |
| Docker | `/tmp/criu-pytorch-results/checkpoint_breakdown_docker.json` | `run_experiment.sh` 自动调用 `analyze_checkpoint.sh` |
| Podman | `/tmp/criu-pytorch-results/checkpoint_breakdown_podman.json` | `run_experiment.sh` 自动调用 `analyze_checkpoint.sh` |

三种输出的 JSON 结构完全一致（见下方 `checkpoint_breakdown.json 结构`），仅 `label` 字段不同（`bare-metal` / `docker` / `podman`），可直接对比各运行环境下的检查点体积和组成差异。

### analyze_checkpoint.sh 工作原理

```
输入: CRIU dump 目录
  ↓
枚举所有文件 → 按文件名模式分类 → 统计各类大小
  ↓
尝试 crit decode inventory.img → 获取 CRIU 元数据
  ↓
解析 dump.log → 提取进程信息 (VmRSS/VmSize/Threads/VMA)
  ↓
输出: 结构化 JSON
```

### 用法

```bash
# 基本用法
./analyze_checkpoint.sh /tmp/criu-dump-bare output.json bare-metal

# Docker checkpoint 目录
./analyze_checkpoint.sh /var/lib/docker/containers/<id>/checkpoints/cp1 output.json docker

# Podman checkpoint 目录
./analyze_checkpoint.sh /tmp/podman-checkpoint output.json podman
```

### 依赖

- `du`, `find`, `stat` — 核心（必需）
- `python3` — JSON 构建（必需）
- `crit` — CRIU 镜像解码（可选，缺失时 crit_inventory=null）
- `bc` — 人类可读大小计算（可选）

### 容错

- 缺失 `crit` 命令 → `crit_available=false`, `crit_inventory=null`
- 缺失 `dump.log` → `process_info=null`
- 空检查点目录 → `file_count=0`, `files=[]`, `category_sizes={}`
- 非目录参数 → 报错退出

## 已知问题

1. **Docker CRIU 延迟**：Docker daemon → containerd → runc 多层 RPC 导致 restore 耗时 ~9s，远超裸机 CRIU 的 ~0.3s
2. **Docker 29 netns bug**：checkpoint 后 restore 网络命名空间失败，需 `--network=host` 绕过
3. **ResNet-50 加速不足**：小模型冷启动仅 ~2.5s，CRIU restore I/O（307MB checkpoint）反而更慢；大模型（7B+参数）预期有 2-3x 加速
4. **bench_compare.sh 路径硬编码**：VENV_PYTHON 和 SCRIPT_DIR 需按实际环境修改
