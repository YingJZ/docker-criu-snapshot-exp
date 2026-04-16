# 方案A：PyTorch CPU推理 + CRIU 快照加速冷启动

使用 CRIU 对 PyTorch CPU 推理容器做 checkpoint/restore，跳过 `import torch` 和 `model.load()` 的冷启动耗时。

⚠️ **实测发现**：对于小模型（如 ResNet-50），CRIU restore 的 I/O 开销可能超过模型加载时间，不一定有加速效果。
但对于大模型（7B+参数，冷启动 >10s），CRIU 方案有显著优势。详见 [实验报告](EXPERIMENT_RESULT.md)。

## 原理

```
[初始化阶段]（仅执行一次）
  Docker 容器启动
    → Python 解释器启动
    → import torch (1.5-5s)      ← 冷启动瓶颈
    → model = load_model() (2-20s) ← 冷启动瓶颈
    → model.eval()
    → 基线推理
    → CRIU checkpoint (创建快照)

[运行阶段]（每次恢复）
  CRIU restore (0.5-2s)           ← 仅此一步!
    → 进程从已加载模型状态恢复
    → 直接开始推理
```

## 文件说明

| 文件 | 用途 |
|------|------|
| `inference_app.py` | PyTorch CPU 推理脚本，含模型加载、基线推理、CRIU 后验证逻辑 |
| `Dockerfile` | 基于本地 torch-cpu 镜像，预装 torchvision 并预下载 ResNet-50 权重 |
| `run_experiment.sh` | 一键自动化实验脚本（构建→启动→checkpoint→restore→验证→报告） |
| `download_weights.sh` | 预下载 ResNet-50 权重文件（国内网络环境推荐使用） |
| `models/` | 存放预下载的模型权重文件，构建时 COPY 进镜像 |
| `EXPERIMENT_RESULT.md` | 实验报告（含详细结果和性能分析） |
| `run_experiment.sh` | 一键自动化实验脚本（构建→启动→checkpoint→restore→验证→报告） |
| `download_weights.sh` | 预下载 ResNet-50 权重文件（国内网络环境推荐使用） |
| `models/` | 存放预下载的模型权重文件，构建时 COPY 进镜像 |
## 前置条件

1. **宿主机安装 CRIU**：`sudo apt-get install -y criu`（建议 3.17+）
2. **Docker 实验特性**：在 `/etc/docker/daemon.json` 中添加 `{"experimental": true}` 并重启 Docker
3. **磁盘空间**：镜像约 3-4GB，checkpoint 快照约 1-2GB

## 快速开始

### 一键运行（推荐）

```bash
cd pytorch-cpu-criu
./run_experiment.sh
```

### 手动步骤

#### 1. 构建镜像

**国内网络环境**（推荐）：先预下载模型权重，再构建镜像：

```bash
# 预下载 ResNet-50 权重（约 97MB）
./download_weights.sh

# 取消 Dockerfile 中 COPY models/ 行的注释
# 然后构建
docker build -t pytorch-criu-cpu .
```

> 如果权重下载失败，推理脚本会自动回退到内置 SmallCNN 模型（无需下载），
> 实验仍然可以验证 CRIU 方案的有效性，只是模型更小、加载更快。

**有代理环境**：构建时传入代理即可：

```bash
docker build --build-arg http_proxy=http://YOUR_PROXY:PORT \
             --build-arg https_proxy=http://YOUR_PROXY:PORT \
             -t pytorch-criu-cpu .
```

#### 2. 启动容器

```bash
docker run -d --name pytorch_criu_demo \
  --security-opt seccomp=unconfined \
  pytorch-criu-cpu
```

等待模型加载完成（日志中出现 `[READY]`）：

```bash
docker logs -f pytorch_criu_demo
```

#### 3. 创建快照

```bash
docker checkpoint create pytorch_criu_demo cp1
```

> 注意：Docker 默认将 checkpoint 存储在容器目录下，`--checkpoint-dir` 参数当前不被支持。

#### 4. 停止并从快照恢复

```bash
docker stop pytorch_criu_demo

docker start --checkpoint cp1 pytorch_criu_demo
```

#### 5. 验证推理结果

恢复后，通过文件触发器或信号触发推理验证：

```bash
# 方式1：文件触发器
docker exec pytorch_criu_demo touch /app/trigger_inference

# 方式2：SIGUSR1 信号
docker exec pytorch_criu_demo kill -USR1 1
```

查看验证结果：

```bash
docker exec pytorch_criu_demo cat /app/results/verification.json
```

**成功标准**：`verification.json` 中 `"match": true`，表示恢复后推理结果与基线完全一致。

#### 6. 查看冷启动数据

```bash
docker exec pytorch_criu_demo cat /app/results/timing.json
```

## 验证项

| 验证项 | 方法 | 成功标准 |
|--------|------|----------|
| 模型对象可用 | restore 后触发推理 | 推理正常执行，无崩溃 |
| Tensor 数据完整 | 比较推理输出 | sum/mean/std 差异 < 1e-5 |
| 推理结果一致 | 基线 vs restore 后 | `match: true` |
| 冷启动加速 | 计时对比 | restore 时间 < 正常启动时间 |
| 多次 restore | 反复 stop/start | 每次推理结果一致，无内存泄漏 |

## 结果文件

| 文件 | 内容 |
|------|------|
| `/app/results/timing.json` | 模型加载和推理的耗时数据 |
| `/app/results/baseline.json` | 基线推理结果（checkpoint 前） |
| `/app/results/verification.json` | restore 后推理验证结果和比较报告 |

## 实测性能（ResNet-50 CPU 推理）

| 指标 | 无 CRIU | 有 CRIU |
|------|---------|---------|
| import torch + model.load() | 2.36s | 0 (快照中) |
| model.eval() + 基线推理 | 0.12s | 0 (快照中) |
| CRIU restore | - | 9.23s |
| **首次推理可用** | **2.48s** | **9.23s** |

> ResNet-50 模型较小，CRIU restore 的磁盘 I/O（307MB checkpoint）比模型加载更慢。
> 大模型场景（7B+参数，冷启动 >10s）预期有 2-3x 加速。详见 [实验报告](EXPERIMENT_RESULT.md)。
## 注意事项

1. **不支持 TTY**：容器必须以 `-d` 后台模式运行
2. **seccomp 配置**：必须使用 `--security-opt seccomp=unconfined`
3. **模型文件路径**：预下载的模型权重打包在镜像中，checkpoint 后路径不变
4. **随机数状态**：CPU RNG 状态被 CRIU 完整保存，推理结果应完全一致
5. **快照体积**：ResNet-50 约 100MB，大型模型（7B参数）约 14GB
6. **checkpoint 时机**：在 `[READY]` 日志出现后执行 checkpoint，此时进程处于 `time.sleep()` 等待状态，GIL 已释放，CRIU checkpoint 安全
