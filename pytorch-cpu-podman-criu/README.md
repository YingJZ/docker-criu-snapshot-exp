# 方案A（Podman 版）：PyTorch CPU推理 + CRIU 快照加速冷启动

使用 Podman + CRIU 对 PyTorch CPU 推理容器做 checkpoint/restore，跳过 `import torch` 和 `model.load()` 的冷启动耗时。

**相比 Docker 版的优势**：Podman 无守护进程（daemonless），CRIU checkpoint/restore 调用路径更短，预期 restore 延迟显著低于 Docker 方案。

> Docker 方案实测 restore 耗时 9.23s，其中 ~9s 是 Docker daemon 多层封装开销（详见 [CRIU耗时分析](../pytorch-cpu-criu/CRIU耗时长的原因分析.md)）。
> Podman 直接通过 conmon+runc 调用 CRIU，绕过了 Docker daemon 的多层 RPC，理论上 restore 更接近裸机 CRIU 性能。

## 原理

```
[初始化阶段]（仅执行一次）
  Podman 容器启动
    → Python 解释器启动
    → import torch (1.5-5s)      ← 冷启动瓶颈
    → model = load_model() (2-20s) ← 冷启动瓶颈
    → model.eval()
    → 基线推理
    → CRIU checkpoint (创建快照)

[运行阶段]（每次恢复）
  CRIU restore (预期 1-3s)       ← 仅此一步!
    → 进程从已加载模型状态恢复
    → 直接开始推理
```

## Docker vs Podman CRIU 调用路径对比

```
Docker 路径 (9.23s):
  docker checkpoint create → Docker daemon → containerd → runc → CRIU
  docker start --checkpoint → Docker daemon → containerd → runc → CRIU
  （多层 RPC 通信，每层有状态管理、锁、序列化开销）

Podman 路径 (预期 1-3s):
  podman container checkpoint → conmon → runc → CRIU
  podman container restore → conmon → runc → CRIU
  （无守护进程，直接 fork+exec 调用 runc/CRIU）

裸机路径 (0.31s):
  sudo criu dump/restore
  （直接调用 CRIU，无容器运行时开销）
```

## 文件说明

| 文件 | 用途 |
|------|------|
| `inference_app.py` | PyTorch CPU 推理脚本，含模型加载、基线推理、CRIU 后验证逻辑 |
| `Dockerfile` | 基于本地 torch-cpu 镜像，预装 torchvision 并预下载 ResNet-50 权重 |
| `run_experiment.sh` | 一键自动化实验脚本（构建→启动→checkpoint→restore→验证→报告） |
| `download_weights.sh` | 预下载 ResNet-50 权重文件（国内网络环境推荐使用） |
| `bare_inference.py` | 裸机 CRIU profiling 脚本（绕过容器，直接测 CRIU 性能） |
| `coldstart_bench.py` | 冷启动基准测试，精确测量各阶段耗时 |
| `bench_compare.sh` | 冷启动 vs CRIU restore 多轮对比基准测试 |
| `models/` | 存放预下载的模型权重文件，构建时 COPY 进镜像 |

## 前置条件

1. **宿主机安装 Podman**：建议 Podman >= 4.0（checkpoint/restore 支持更完善）
   ```bash
   # Ubuntu/Debian
   sudo apt-get install -y podman

   # RHEL/CentOS/Fedora
   sudo dnf install -y podman
   ```

2. **宿主机安装 CRIU**：`sudo apt-get install -y criu`（建议 3.17+，Podman 依赖宿主机 CRIU）

3. **磁盘空间**：镜像约 3-4GB，checkpoint 快照约 1-2GB

4. **内核版本**：建议 Linux 5.9+（CRIU 对较新内核支持更好）

> ⚠️ **无需开启实验特性**：Podman 原生支持 container checkpoint/restore，不需要像 Docker 那样开启 `experimental: true`。

## 快速开始

### 一键运行（推荐）

```bash
cd pytorch-cpu-podman-criu
chmod +x run_experiment.sh download_weights.sh
./run_experiment.sh
```

### 手动步骤

#### 1. 构建镜像

**国内网络环境**（推荐）：先预下载模型权重，再构建镜像：

```bash
# 预下载 ResNet-50 权重（约 97MB）
./download_weights.sh

# 构建镜像
podman build -t pytorch-criu-cpu-podman .
```

**有代理环境**：构建时传入代理即可：

```bash
podman build --build-arg http_proxy=http://YOUR_PROXY:PORT \
             --build-arg https_proxy=http://YOUR_PROXY:PORT \
             -t pytorch-criu-cpu-podman .
```

> 如果权重下载失败，推理脚本会自动回退到内置 SmallCNN 模型（无需下载），
> 实验仍然可以验证 CRIU 方案的有效性，只是模型更小、加载更快。

#### 2. 启动容器

```bash
podman run -d --name pytorch_criu_podman_demo \
  --security-opt seccomp=unconfined \
  pytorch-criu-cpu-podman
```

等待模型加载完成（日志中出现 `[READY]`）：

```bash
podman logs -f pytorch_criu_podman_demo
```

#### 3. 创建快照

```bash
podman container checkpoint pytorch_criu_podman_demo
```

> 容器会自动停止。Checkpoint 默认保存在容器的存储目录中。

**导出 checkpoint 到指定目录**（可选，便于迁移）：

```bash
podman container checkpoint pytorch_criu_podman_demo \
  --export /tmp/pytorch-checkpoint.tar.gz
```

#### 4. 从快照恢复

```bash
# 从容器自身的 checkpoint 恢复
podman container restore pytorch_criu_podman_demo
```

**从导出的 checkpoint 文件恢复**（可选，用于跨主机迁移）：

```bash
# 先创建同名容器（已停止状态）
podman create --name pytorch_criu_podman_demo \
  --security-opt seccomp=unconfined \
  pytorch-criu-cpu-podman

# 从导出的 checkpoint 恢复
podman container restore pytorch_criu_podman_demo \
  --import /tmp/pytorch-checkpoint.tar.gz
```

#### 5. 验证推理结果

恢复后，通过文件触发器或信号触发推理验证：

```bash
# 方式1：文件触发器
podman exec pytorch_criu_podman_demo touch /app/trigger_inference

# 方式2：SIGUSR1 信号
podman exec pytorch_criu_podman_demo kill -USR1 1
```

查看验证结果：

```bash
podman exec pytorch_criu_podman_demo cat /app/results/verification.json
```

**成功标准**：`verification.json` 中 `"match": true`，表示恢复后推理结果与基线完全一致。

#### 6. 查看冷启动数据

```bash
podman exec pytorch_criu_podman_demo cat /app/results/timing.json
```

## 验证项

| 验证项 | 方法 | 成功标准 |
|--------|------|----------|
| 模型对象可用 | restore 后触发推理 | 推理正常执行，无崩溃 |
| Tensor 数据完整 | 比较推理输出 | sum/mean/std 差异 < 1e-5 |
| 推理结果一致 | 基线 vs restore 后 | `match: true` |
| 冷启动加速 | 计时对比 | restore 时间 < 正常启动时间 |
| 多次 restore | 反复 checkpoint/restore | 每次推理结果一致，无内存泄漏 |

## 结果文件

| 文件 | 内容 |
|------|------|
| `/app/results/timing.json` | 模型加载和推理的耗时数据 |
| `/app/results/baseline.json` | 基线推理结果（checkpoint 前） |
| `/app/results/verification.json` | restore 后推理验证结果和比较报告 |

## Podman 与 Docker CRIU 命令对照

| 操作 | Docker | Podman |
|------|--------|--------|
| 创建 checkpoint | `docker checkpoint create <container> <name>` | `podman container checkpoint <container>` |
| 恢复容器 | `docker start --checkpoint <name> <container>` | `podman container restore <container>` |
| 导出 checkpoint | 不支持 | `podman container checkpoint --export <file>` |
| 导入 checkpoint | 不支持 | `podman container restore --import <file>` |
| 前置条件 | 需要开启 `experimental: true` | 原生支持，无需额外配置 |

## 注意事项

1. **不支持 TTY**：容器必须以 `-d` 后台模式运行
2. **seccomp 配置**：必须使用 `--security-opt seccomp=unconfined`
3. **rootless 模式限制**：Podman rootless 模式下 checkpoint/restore 可能受限，建议使用 root 模式或配置 `--privileged`
4. **模型文件路径**：预下载的模型权重打包在镜像中，checkpoint 后路径不变
5. **随机数状态**：CPU RNG 状态被 CRIU 完整保存，推理结果应完全一致
6. **快照体积**：ResNet-50 约 100MB，大型模型（7B参数）约 14GB
7. **checkpoint 时机**：在 `[READY]` 日志出现后执行 checkpoint，此时进程处于 `time.sleep()` 等待状态，GIL 已释放，CRIU checkpoint 安全
8. **Podman 版本要求**：建议 Podman >= 4.0，3.x 版本的 checkpoint/restore 可能不稳定
9. **CRIU 版本要求**：建议 CRIU >= 3.17，可通过 `criu --version` 检查

## 常见问题

### Q: `podman container checkpoint` 报错 "checkpoint/restore not supported"

检查：
1. CRIU 是否安装：`criu --version`
2. 内核版本：`uname -r`（建议 5.9+）
3. Podman 版本：`podman --version`（建议 4.0+）
4. 是否有足够权限：rootless 模式下可能需要额外配置

### Q: restore 后容器立即退出

可能原因：
1. 容器内进程在 checkpoint 时不在稳定的等待状态 → 确保等待 `[READY]` 日志
2. seccomp 配置冲突 → 确保使用 `--security-opt seccomp=unconfined`
3. 文件描述符丢失 → 检查 Podman 日志：`podman logs <container>`

### Q: Podman restore 比 Docker 还慢

排查：
1. 检查 Podman 是否使用了正确的 CRIU 版本
2. 检查 checkpoint 存储位置是否在慢速磁盘上
3. 尝试将 checkpoint 目录放在 tmpfs：`--checkpoint-dir /dev/shm/criu-checkpoints`
4. 对比裸机 CRIU 性能：使用 `bare_inference.py` + 直接 `criu restore` 测试

## 性能预期

基于 [CRIU耗时分析](../pytorch-cpu-criu/CRIU耗时长的原因分析.md) 中的数据：

| 方案 | 首次推理可用 | 说明 |
|------|-------------|------|
| 裸机冷启动 | ~5.6s | python 启动 → import → model.load → 推理 |
| 裸机 CRIU | ~0.4s | criu restore → 推理 (**13.4x 加速**) |
| Docker CRIU | ~9.2s | docker start --checkpoint → 推理 (**1.7x 更慢**) |
| **Podman CRIU** | **预期 1-3s** | podman restore → 推理 (**预期 2-5x 加速**) |

Podman restore 的预期性能介于裸机 CRIU (0.3s) 和 Docker CRIU (9.2s) 之间，因为：
- 绕过了 Docker daemon → containerd 的多层 RPC 开销
- 但仍有 conmon + runc + namespace 重建等容器运行时开销
- 实际性能取决于 Podman 版本和 CRIU 集成质量，需实测确认
