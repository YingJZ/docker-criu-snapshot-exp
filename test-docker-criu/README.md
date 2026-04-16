# Docker + CRIU 容器快照与恢复实验（宿主机端方案）

在宿主机上使用 CRIU 对 Docker 容器执行 checkpoint（冻结）与 restore（恢复），验证容器内进程状态（内存中的变量）能够在恢复后精确延续。

> **方案说明**：CRIU 安装在**宿主机**上，通过 Docker 原生的 `docker checkpoint` 命令从外部对整个容器做快照，无需在容器内安装 CRIU，也无需 `--privileged` 特权模式。

## 文件说明

| 文件 | 用途 |
|------|------|
| `test_app.py` | 带状态计数器脚本，每秒向 `output.log` 写入递增数字 |
| `Dockerfile` | 基于 Ubuntu 22.04，仅安装 Python3（CRIU 在宿主机上运行） |

## 前置条件

### 1. 宿主机安装 CRIU

```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y criu

# CentOS / RHEL
sudo yum install -y criu

# 验证安装
criu --version
```

> 建议使用 CRIU 3.17+，内核建议 5.10+。

### 2. 启用 Docker 实验特性

CRIU 集成目前仍为 Docker 实验性功能，需显式启用：

```bash
# 创建或编辑 /etc/docker/daemon.json
sudo tee /etc/docker/daemon.json <<EOF
{
  "experimental": true
}
EOF

# 重启 Docker
sudo systemctl restart docker
```

验证实验特性已启用：

```bash
docker version -f '{{.Server.Experimental}}'
# 应输出 true
```

## 运行步骤

### 1. 构建镜像

```bash
docker build -t criu-test .
```

### 2. 启动容器（后台模式）

```bash
docker run -d --name criu_demo \
  --security-opt seccomp=unconfined \
  criu-test
```

> - **`-d`**：后台运行，CRIU 不支持对带有 TTY（`-it`）的容器做快照。
> - **`--security-opt seccomp=unconfined`**：放宽 seccomp 限制，允许 CRIU 所需的 ptrace 等系统调用。无需 `--privileged`。

### 3. 观察容器运行状态

```bash
# 查看实时日志输出
docker logs -f criu_demo
```

等待几秒，观察计数递增，然后 Ctrl+C 退出日志跟踪。

### 4. 创建快照（Checkpoint）

```bash
docker checkpoint create --leave-running criu_demo cp1
```

> - `--leave-running`：创建快照后**不停止**容器（可省略，省略则容器会被暂停）。
> - `cp1`：快照名称，可自定义。

也可以在不停止容器的情况下先确认快照已保存：

```bash
docker checkpoint ls criu_demo
```

### 5. 停止容器并从快照恢复

```bash
# 停止容器
docker stop criu_demo

# 从快照恢复
docker start --checkpoint cp1 criu_demo
```

### 6. 验证结果

```bash
docker logs -f criu_demo
```

**成功标准**：恢复后日志中的数字从 Checkpoint 时中断的值继续递增，而非从 `0` 重新开始。

也可以进入容器查看 `output.log`：

```bash
docker exec criu_demo cat /app/output.log
```

## 其他操作

### 仅冻结不保留运行（默认行为）

```bash
docker checkpoint create criu_demo cp1
# 容器自动停止，等价于先 checkpoint 再 stop
```

恢复方式同上：`docker start --checkpoint cp1 criu_demo`

### 指定快照存储目录

```bash
# 创建快照到自定义目录
docker checkpoint create --checkpoint-dir /tmp/checkpoints criu_demo cp1

# 从自定义目录恢复
docker start --checkpoint-dir /tmp/checkpoints --checkpoint cp1 criu_demo
```

### 删除快照

```bash
docker checkpoint rm criu_demo cp1
```

## 关键参数说明

| 参数 | 说明 |
|------|------|
| `--security-opt seccomp=unconfined` | 放宽 seccomp 限制，允许 CRIU 所需的 ptrace 等系统调用 |
| `docker checkpoint create` | 在宿主机端对容器执行 CRIU checkpoint |
| `docker start --checkpoint` | 从 CRIU 快照恢复容器 |
| `--leave-running` | checkpoint 后保持容器运行（默认会暂停容器） |
| `--checkpoint-dir` | 指定快照存储目录（默认在 `/var/lib/docker/containers/<ID>/checkpoints/`） |

## 注意事项

1. **不支持 TTY**：以 `-it`（交互式终端）启动的容器无法执行 checkpoint，必须使用 `-d` 后台模式。
2. **TCP 连接**：默认情况下，有已建立 TCP 连接的容器做 checkpoint 时连接会断开。如需保留 TCP 连接状态，需在宿主机配置 `/etc/criu/runc.conf` 并添加 `tcp-established`。
3. **内核版本**：建议 Linux 内核 4.3+（seccomp 挂起支持），5.10+ 更佳。
4. **GPU / 外设**：使用了 GPU 或特定硬件的容器通常无法被 CRIU 快照。
5. **快照存储**：默认存储在 Docker 的内部目录中，容器删除后快照也会丢失。如需持久化，请使用 `--checkpoint-dir` 指定外部目录。
