# Docker + CRIU 容器内进程快照与恢复实验

验证在 Docker 容器中使用 CRIU 对运行中进程进行 checkpoint（冻结）与 restore（恢复），确认进程状态（内存中的变量）能够在恢复后精确延续。

## 文件说明

| 文件 | 用途 |
|------|------|
| `test_app.py` | 带状态计数器脚本，每秒向 `output.log` 写入递增数字 |
| `Dockerfile` | 基于 Ubuntu 22.04，安装 CRIU 和 Python3 |

## 运行步骤

### 1. 构建镜像

```bash
docker build -t criu-test .
```

### 2. 启动容器（特权模式）

CRIU 需要操作 `/proc`、ptrace 等内核接口，Docker 默认安全策略会阻止这些操作，因此必须使用特权模式启动：

```bash
docker run -it --rm \
  --name criu_demo \
  --privileged \
  --security-opt seccomp=unconfined \
  criu-test
```

此时已进入容器内部的 bash 终端，后续命令均在容器内执行。

### 3. 准备快照目录

```bash
mkdir -p /app/img
```

### 4. 运行测试脚本

```bash
python3 test_app.py &
```

终端会打印进程 PID 和递增计数，记下第一行输出的 PID（如 `123`）。

### 5. 执行快照冻结（Dump）

将 `123` 替换为实际的 PID：

```bash
criu dump -t 123 --images-dir /app/img --shell-job
```

成功后无报错，Python 进程已终止，计数停止。`--shell-job` 告知 CRIU 此进程由 shell 启动，不需要恢复整个会话。

### 6. 验证快照文件

```bash
ls -l /app/img
```

目录下会生成大量进程内存与状态的镜像文件。

### 7. 恢复进程（Restore）

```bash
criu restore --images-dir /app/img --shell-job &
```

### 8. 验证结果

```bash
tail -f output.log
```

**成功标准**：恢复后日志中的数字从 Dump 时中断的值继续递增，而非从 `0` 重新开始。

## 关键参数说明

- `--privileged`：授予容器全部内核权限，CRIU 必需
- `--security-opt seccomp=unconfined`：禁用 Seccomp 限制，允许 ptrace 等系统调用
- `--shell-job`：标识进程由 shell 管理，避免 CRIU 尝试恢复控制终端
- `-t <PID>`：指定要冻结的目标进程 PID
