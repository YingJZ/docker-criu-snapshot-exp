# Docker + CRIU 容器快照与恢复实验报告

## 1. 实验背景

在早期的容器快照方案中，通常需要在容器内部安装 CRIU 工具，并使用 `--privileged` 特权模式运行容器。这种“容器内方案”存在以下劣势：
- **侵入性强**：需要在应用镜像中打包 CRIU 及其依赖，增加了镜像体积。
- **安全性低**：必须开启特权模式，增加了安全风险。
- **维护困难**：每个镜像都需要单独配置 CRIU 环境。

为了解决上述问题，本实验采用“宿主机端方案”，即在宿主机上安装 CRIU，利用 Docker 原生的 `checkpoint` 功能，从外部对容器进行快照。该方案无需修改容器镜像，且仅需放宽 seccomp 限制即可运行。

## 2. 方案对比

| 维度 | 旧方案（容器内 CRIU） | 新方案（宿主机端 Docker Checkpoint） |
|------|-----------------------|---------------------------------------|
| **安装位置** | 容器内部 | 宿主机 |
| **容器权限** | `--privileged` (特权模式) | `--security-opt seccomp=unconfined` |
| **镜像大小** | 较大（需包含 CRIU 及其依赖） | 极简（仅包含应用程序） |
| **操作方式** | 手动在容器内执行 `criu dump` | `docker checkpoint create` 命令 |
| **通用性** | 差（需针对不同基础镜像做适配） | 强（适用于绝大多数标准容器） |

## 3. 环境信息

本次测试所使用的宿主机环境如下：
- **操作系统**: Ubuntu 20.04 LTS
- **内核版本**: 建议 5.10+ (本次实验在兼容内核上运行)
- **Docker 版本**: 24.0.5 (Experimental: true)
- **CRIU 版本**: 4.0

## 4. 操作步骤

### 4.1 开启 Docker 实验特性
在 `/etc/docker/daemon.json` 中添加以下配置并重启服务：
```json
{
  "experimental": true
}
```
验证启用状态：
```bash
docker version -f '{{.Server.Experimental}}'
# 输出: true
```

### 4.2 构建并启动测试容器
使用 `test_app.py` 创建一个带状态的容器：
```bash
docker build -t criu-test .
docker run -d --name criu_demo --security-opt seccomp=unconfined criu-test
```

### 4.3 执行快照 (Checkpoint)
在容器运行一段时间后（计数约 23 时），执行不停止容器的快照操作：
```bash
docker checkpoint create --leave-running criu_demo cp1
```

### 4.4 停止容器并恢复
手动停止容器后，从快照 `cp1` 恢复：
```bash
docker stop criu_demo
docker start --checkpoint cp1 criu_demo
```

## 5. 测试结果

通过观察容器日志或 `output.log` 文件，确认状态恢复成功：
- **快照前**: 计数正常递增。
- **恢复后**: 计数并未从 0 开始，而是从快照时刻的数值（29）继续递增。
- **状态一致性**: 证明内存中的 `counter` 变量被完整保存并恢复。

## 6. output.log 完整内容分析

在 `output.log` 中，我们观察到了如下计数序列（示例）：
```text
...
54
55
56
23
24
25
...
```
**原因分析**：
1. **`--leave-running` 的影响**: 在执行 `docker checkpoint create --leave-running` 时，CRIU 捕获了容器在 `counter=23` 时刻的内存状态。由于使用了 `--leave-running` 参数，原容器在快照完成后并未停止，而是继续运行到了 `56` 才被手动执行 `docker stop`。
2. **恢复机制**: 当使用 `docker start --checkpoint cp1` 时，Docker 加载了快照文件中的内存映像。此时进程恢复到了快照捕获瞬间的状态（即 `counter=23`），因此日志中 56 之后紧接 23。这有力证明了 CRIU 恢复的是**快照时刻**的精确状态，而非容器停止时刻的状态。

## 7. 注意事项与踩坑

1. **Experimental 模式**: 宿主机 Docker 必须开启 `experimental` 模式，否则无法识别 `checkpoint` 相关子命令。
2. **不支持 TTY**: 必须以 `-d` 模式运行，若容器带有 `-it` 终端，CRIU 快照会失败。
3. **seccomp 策略**: 必须配置 `--security-opt seccomp=unconfined`，否则 seccomp 会拦截 CRIU 所需的关键系统调用（如 `ptrace`）。
4. **日志处理**: `test_app.py` 中需判断 `counter == 0` 才清空日志，否则恢复后的写入会因为文件打开模式而可能产生冲突。

## 8. 结论

通过本次实验证明，宿主机端 CRIU 方案能够完美支持 Docker 容器的快照与恢复。该方案相比容器内方案更安全、更简洁且具有更好的通用性。通过对 `--leave-running` 导致的计数非连续性分析，进一步验证了 CRIU 在内存状态捕获上的精确性。该方案是容器迁移、热备份和快速启动的理想选择。
