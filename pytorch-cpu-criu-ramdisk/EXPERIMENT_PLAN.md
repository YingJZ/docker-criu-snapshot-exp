# 方案A-优化：CRIU Checkpoint 存放 RAM Disk 加速 Restore 实验

## 1. 问题背景

在[原始实验](../pytorch-cpu-criu/EXPERIMENT_RESULT.md)中，CRIU restore（9.23s）反而比常规冷启动（2.48s）更慢。分析认为瓶颈在于磁盘 I/O：

- Checkpoint 体积 307MB，restore 需要从磁盘读取全部内容
- 磁盘顺序读取速度（~500MB/s SATA SSD, ~3000MB/s NVMe）仍需 0.1-0.6s 的纯 I/O 时间
- 但 CRIU restore 还有内存页映射、页表重建、线程恢复等开销，I/O 延迟会被放大

**假设**：如果将 CRIU checkpoint 存放在 RAM disk（tmpfs）上，消除磁盘 I/O 瓶颈，restore 速度可能显著提升，甚至快于常规冷启动。

## 2. 实验假设

| 条件 | 预期效果 | 原因 |
|------|----------|------|
| Checkpoint 在普通磁盘 | restore ~9s（与原始实验一致） | 磁盘 I/O 是瓶颈 |
| Checkpoint 在 RAM disk | restore **显著降低** | 消除磁盘 I/O，内存→内存拷贝 |
| RAM disk restore vs 冷启动 | **可能更快** | 307MB 内存拷贝 (~0.05s) + CRIU 固定开销 (~1-3s) << 2.48s 冷启动 |

### 理论估算

```
冷启动 = import torch (~1.5s) + model.load() (~0.86s) + eval+baseline (~0.12s) = 2.48s

CRIU restore (RAM disk) = CRIU固定开销 + 内存页拷贝时间
  - CRIU 固定开销（线程恢复、FD重建、网络命名空间等）: ~1-3s
  - 307MB 内存→内存拷贝 (tmpfs → 进程地址空间): ~0.05-0.1s
  - 预期总计: ~1-3s

如果 CRIU 固定开销 < 2.4s，则 RAM disk restore 将快于冷启动 ✅
```

## 3. 实验设计

### 3.1 三组对照实验

| 组别 | 方案 | 存储位置 | 测量内容 |
|------|------|----------|----------|
| A | 常规冷启动 | 无 checkpoint | 容器启动 → 模型加载 → 推理就绪 |
| B | CRIU restore | 普通磁盘 | 从磁盘 checkpoint restore |
| C | CRIU restore | RAM disk (tmpfs) | 从内存 checkpoint restore |

每组运行 N 次（默认3次），取平均值和标准差。

### 3.2 RAM Disk 实现

使用 Linux tmpfs：

```bash
# 创建挂载点
sudo mkdir -p /tmp/criu-ramdisk

# 挂载 tmpfs（限制1GB，足够 ResNet-50 的307MB checkpoint）
sudo mount -t tmpfs -o size=1G tmpfs /tmp/criu-ramdisk

# 使用 --checkpoint-dir 参数指定 checkpoint 存放位置
docker checkpoint create --checkpoint-dir /tmp/criu-ramdisk <container> cp1
docker start --checkpoint cp1 --checkpoint-dir /tmp/criu-ramdisk <container>
```

### 3.3 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| Checkpoint 目录 (磁盘) | `/tmp/criu-disk-checkpoints` | 普通 ext4/xfs 文件系统 |
| Checkpoint 目录 (RAM) | `/tmp/criu-ramdisk` | tmpfs (内存文件系统) |
| tmpfs 大小限制 | 1GB | 远大于 ResNet-50 checkpoint (307MB) |
| 每组迭代次数 | 3（可调） | 通过脚本参数控制 |
| 模型 | ResNet-50 (25M params, 97.5MB) | 与原始实验一致 |

### 3.4 计时方法

- **Wall clock 时间**：`date +%s%N` 计算从命令发出到容器 start 返回的时间
- **容器内部时间**：`timing.json` 中记录的 `total_init_s`（仅冷启动有）
- **CRIU restore 不包含模型加载**：restore 后模型已在内存中，无需重新加载

## 4. 预期结果

### 4.1 乐观预期（CRIU 固定开销较小）

| 方案 | 预期耗时 | 加速比 |
|------|----------|--------|
| 冷启动 | ~2.5s | 1.0x (baseline) |
| 磁盘 restore | ~9.2s | 0.3x ❌ |
| RAM disk restore | ~1.5-3s | **0.8-1.7x** ⚡ |

如果 CRIU 固定开销 ~1.5s，RAM disk restore 可达 **1.7x 加速**。

### 4.2 保守预期（CRIU 固定开销较大）

| 方案 | 预期耗时 | 加速比 |
|------|----------|--------|
| 冷启动 | ~2.5s | 1.0x (baseline) |
| 磁盘 restore | ~9.2s | 0.3x ❌ |
| RAM disk restore | ~3-5s | **0.5-0.8x** |

即使 CRIU 固定开销较大，RAM disk restore 也应显著快于磁盘 restore。

### 4.3 关键变量

实际结果取决于：

1. **CRIU 进程恢复的固定开销**：线程状态重建、FD 恢复、命名空间设置等，这部分与 I/O 无关
2. **Docker 层面的开销**：容器 start 本身的延迟，与 CRIU 无关
3. **tmpfs 实际性能**：虽然是内存盘，但 Linux 仍需经过 VFS 层和内存拷贝
4. **CPU 缓存效果**：多次 restore 后，数据可能已在 CPU 缓存中

## 5. 文件说明

| 文件 | 用途 |
|------|------|
| `run_experiment_ramdisk.sh` | 主实验脚本：自动挂载 tmpfs、运行三组对照、输出统计报告 |
| `inference_app.py` | PyTorch 推理脚本（与原始实验相同） |
| `Dockerfile` | Docker 镜像构建（与原始实验相同） |
| `download_weights.sh` | 预下载 ResNet-50 权重 |
| `EXPERIMENT_PLAN.md` | 本文档：实验设计和预期 |

## 6. 运行方法

### 一键运行

```bash
cd pytorch-cpu-criu-ramdisk

# 默认3次迭代
./run_experiment_ramdisk.sh

# 自定义迭代次数（如5次）
./run_experiment_ramdisk.sh 5
```

### 前置条件

1. **sudo 权限**：脚本需要 `sudo` 来挂载/卸载 tmpfs
2. **可用内存 >= 1.5GB**：tmpfs 需要占用 ~307MB，容器运行也需要内存
3. Docker 实验特性已启用（同原始实验）
4. CRIU 已安装（同原始实验）

### 手动分步运行

如果需要更细粒度的控制：

```bash
# 1. 手动挂载 RAM disk
sudo mkdir -p /tmp/criu-ramdisk
sudo mount -t tmpfs -o size=1G tmpfs /tmp/criu-ramdisk

# 2. 构建镜像
docker build -t pytorch-criu-cpu-ramdisk .

# 3. 启动容器
docker run -d --name pytorch_criu_ramdisk \
  --security-opt seccomp=unconfined \
  pytorch-criu-cpu-ramdisk

# 等待就绪
docker logs -f pytorch_criu_ramdisk  # 看到 [READY] 后 Ctrl+C

# 4. 创建 checkpoint 到 RAM disk
time docker checkpoint create \
  --checkpoint-dir /tmp/criu-ramdisk \
  pytorch_criu_ramdisk cp1

# 5. 查看 checkpoint 大小
du -sh /tmp/criu-ramdisk/cp1

# 6. 停止容器
docker stop pytorch_criu_ramdisk

# 7. 从 RAM disk checkpoint 恢复（计时）
time docker start \
  --checkpoint cp1 \
  --checkpoint-dir /tmp/criu-ramdisk \
  pytorch_criu_ramdisk

# 8. 验证推理结果
sleep 2
docker exec pytorch_criu_ramdisk touch /app/trigger_inference
sleep 2
docker exec pytorch_criu_ramdisk cat /app/results/verification.json

# 9. 对比：从普通磁盘 checkpoint 恢复
mkdir -p /tmp/criu-disk-checkpoints
# 重复上述步骤，但 --checkpoint-dir 用 /tmp/criu-disk-checkpoints

# 10. 清理
docker rm -f pytorch_criu_ramdisk
sudo umount /tmp/criu-ramdisk
```

## 7. 结果分析指南

### 如果 RAM disk restore < 冷启动 ✅

**结论**：CRIU 方案在小模型场景也有效，前提是消除 I/O 瓶颈。

**后续方向**：
- 将 RAM disk 方案应用于大模型场景，预期加速效果更显著
- 探索预热策略：系统启动时自动将 checkpoint 加载到 tmpfs
- 评估 Serverless 场景下的实际收益

### 如果 RAM disk restore ≈ 冷启动 ⚠️

**结论**：CRIU 固定开销 ≈ 模型加载时间，RAM disk 消除了 I/O 瓶颈但收益被固定开销抵消。

**后续方向**：
- 对比 CRIU 固定开销的详细 profiling（`criu restore -vvv` 日志分析）
- 对更大模型测试（大模型冷启动 >> CRIU 固定开销）

### 如果 RAM disk restore > 冷启动 ❌

**结论**：CRIU 进程恢复的固定开销本身就是主要瓶颈（而非 I/O）。

**后续方向**：
- CRIU profiling：分析 restore 各阶段耗时
- 考虑替代方案（如进程预热池、model mmap 等）
- 仅在大模型场景（冷启动 >10s）使用 CRIU

## 8. 扩展实验建议

1. **NVMe SSD 对比**：在 NVMe 上重复实验，验证 RAM disk 相对 NVMe 的优势是否明显
2. **更大模型**：使用 BERT-base (110M) 或 LLaMA-7B，验证 RAM disk 在大模型下的收益
3. **多次连续 restore**：测试 CRIU restore 的时间稳定性（是否有热身效应）
4. **Checkpoint 压缩**：使用 `--compress` 参数减小 checkpoint 体积，测试压缩/解压的时间权衡
5. **页面预加载**：`madvise(MADV_WILLNEED)` 提前触发内存页读入
