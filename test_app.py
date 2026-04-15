import time
import os
import sys

print(f"进程启动! PID: {os.getpid()}")
counter = 0
log_file = "output.log"

# 确保每次启动时清空旧日志
if os.path.exists(log_file) and counter == 0:
    open(log_file, "w").close()

while True:
    with open(log_file, "a") as f:
        f.write(f"当前计数: {counter}\n")
    print(f"正在运行... 计数: {counter}")
    counter += 1
    time.sleep(1)
