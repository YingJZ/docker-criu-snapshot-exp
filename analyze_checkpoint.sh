#!/bin/bash
# analyze_checkpoint.sh — CRIU 检查点目录分解分析
# 用法: ./analyze_checkpoint.sh <checkpoint_dir> <output_json> [label]
#   checkpoint_dir: CRIU dump 目录路径
#   output_json:    输出 JSON 文件路径
#   label:          可选标签 (bare-metal / docker / podman)

set -euo pipefail

# ---- 参数解析 ----
if [ $# -lt 2 ]; then
    echo "用法: $0 <checkpoint_dir> <output_json> [label]" >&2
    exit 1
fi

CHECKPOINT_DIR="$1"
OUTPUT_JSON="$2"
LABEL="${3:-}"

# ---- 前置检查 ----
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "[ERROR] 检查点目录不存在: $CHECKPOINT_DIR" >&2
    exit 1
fi

# ---- 辅助函数 ----
# 按文件名模式分类
categorize_file() {
    local name="$1"
    case "$name" in
        pages-*.img)    echo "pages" ;;
        core-*.img)     echo "core" ;;
        mm-*.img)       echo "mm" ;;
        files.img|fdinfo-*.img) echo "fd" ;;
        netns-*.img)    echo "net" ;;
        cgroup.img)     echo "cgroup" ;;
        pstree.img)     echo "proc" ;;
        inventory.img)  echo "inventory" ;;
        *.log)          echo "log" ;;
        *)              echo "other" ;;
    esac
}

# 人类可读大小
human_size() {
    local bytes="$1"
    if [ "$bytes" -ge 1073741824 ]; then
        echo "scale=2; $bytes / 1073741824" | bc | sed 's/^\./0./' | xargs -I{} echo "{}GB"
    elif [ "$bytes" -ge 1048576 ]; then
        echo "scale=2; $bytes / 1048576" | bc | sed 's/^\./0./' | xargs -I{} echo "{}MB"
    elif [ "$bytes" -ge 1024 ]; then
        echo "scale=2; $bytes / 1024" | bc | sed 's/^\./0./' | xargs -I{} echo "{}KB"
    else
        echo "${bytes}B"
    fi
}

# ---- 收集数据 ----

# 1. 目录总大小
total_size_bytes=0
if du -sb "$CHECKPOINT_DIR" &>/dev/null; then
    total_size_bytes=$(du -sb "$CHECKPOINT_DIR" | cut -f1)
fi
total_size_human=$(human_size "$total_size_bytes")

# 2. 文件数量和逐文件分解
file_count=0
files_json="[]"
category_sizes_json="{}"

# 临时文件存放中间数据
tmp_files=$(mktemp)
tmp_categories=$(mktemp)
trap 'rm -f "$tmp_files" "$tmp_categories"' EXIT

# 枚举所有文件并写入临时文件
while IFS= read -r -d '' filepath; do
    relpath="${filepath#$CHECKPOINT_DIR/}"
    size_bytes=$(stat -c%s "$filepath" 2>/dev/null || stat -f%z "$filepath" 2>/dev/null || echo 0)
    category=$(categorize_file "$relpath")
    size_human=$(human_size "$size_bytes")

    echo "${relpath}|${size_bytes}|${size_human}|${category}" >> "$tmp_files"
    echo "${category}|${size_bytes}" >> "$tmp_categories"

    file_count=$((file_count + 1))
done < <(find "$CHECKPOINT_DIR" -type f -print0 2>/dev/null)

# 3. crit 是否可用
crit_available=false
crit_inventory="null"
if command -v crit &>/dev/null; then
    crit_available=true
    if [ -f "$CHECKPOINT_DIR/inventory.img" ]; then
        crit_inventory=$(crit decode -i "$CHECKPOINT_DIR/inventory.img" --pretty 2>/dev/null || echo "null")
        # 转义 JSON 中的特殊字符
        crit_inventory=$(python3 -c "
import sys, json
raw = sys.stdin.read()
try:
    obj = json.loads(raw)
    print(json.dumps(obj))
except:
    print('null')
" <<< "$crit_inventory")
    fi
fi

# 4. 从 dump.log 提取进程信息
process_info="null"
if [ -f "$CHECKPOINT_DIR/dump.log" ]; then
    process_info=$(python3 -c "
import re, json, sys

log_path = '$CHECKPOINT_DIR/dump.log'
info = {}

try:
    with open(log_path, 'r', errors='replace') as f:
        content = f.read()

    # VmRSS
    m = re.search(r'VmRSS:\s*(\d+)\s*kB', content)
    if m:
        info['VmRSS_kB'] = int(m.group(1))

    # VmSize
    m = re.search(r'VmSize:\s*(\d+)\s*kB', content)
    if m:
        info['VmSize_kB'] = int(m.group(1))

    # Threads
    m = re.search(r'Threads:\s*(\d+)', content)
    if m:
        info['Threads'] = int(m.group(1))

    # VMA count
    m = re.search(r'(\d+) vma.s', content)
    if m:
        info['VMA_count'] = int(m.group(1))

    print(json.dumps(info if info else None))
except Exception as e:
    print('null')
")
fi

# ---- 用 python3 构建最终 JSON ----
python3 << PYEOF
import json, sys

# 读取逐文件数据
files = []
category_sizes = {}

with open("$tmp_files", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) != 4:
            continue
        name, size_bytes, size_human, category = parts
        files.append({
            "name": name,
            "size_bytes": int(size_bytes),
            "size_human": size_human,
            "category": category
        })
        category_sizes[category] = category_sizes.get(category, 0) + int(size_bytes)

# 读取分类汇总（验证用）
with open("$tmp_categories", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) != 2:
            continue
        cat, sz = parts
        # 已经在上面累加了，这里不做重复

# crit inventory
crit_available = True if "$crit_available" == "true" else False
crit_inv_raw = """$(echo "$crit_inventory")"""
try:
    crit_inv = json.loads(crit_inv_raw)
except:
    crit_inv = None

# process info
proc_info_raw = """$(echo "$process_info")"""
try:
    proc_info = json.loads(proc_info_raw)
except:
    proc_info = None

# 构建输出
result = {
    "checkpoint_dir": "$CHECKPOINT_DIR",
    "label": "$LABEL" if "$LABEL" else None,
    "total_size_bytes": $total_size_bytes,
    "total_size_human": "$total_size_human",
    "file_count": $file_count,
    "files": files,
    "category_sizes": category_sizes,
    "crit_available": crit_available,
    "crit_inventory": crit_inv,
    "process_info": proc_info
}

with open("$OUTPUT_JSON", "w") as out:
    json.dump(result, out, indent=2, ensure_ascii=False)

print(f"[analyze_checkpoint] 结果已写入: $OUTPUT_JSON")
PYEOF

echo "[analyze_checkpoint] 总大小: $total_size_human, 文件数: $file_count"
