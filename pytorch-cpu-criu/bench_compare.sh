#!/bin/bash
# 冷启动 vs CRIU restore 端到端对比基准测试
# 测量从零开始启动 PyTorch 推理 vs CRIU restore 恢复推理的完整耗时

set -euo pipefail

VENV_PYTHON="/home/yingjiaze/docker-criu-snapshot-exp/pytorch-cpu-criu/.venv/bin/python"
SCRIPT_DIR="/home/yingjiaze/docker-criu-snapshot-exp/pytorch-cpu-criu"
RESULT_DIR="/tmp/criu-pytorch-results"
PID_FILE="/tmp/criu-pytorch.pid"
TRIGGER_FILE="/tmp/criu-pytorch-trigger"
CHECKPOINT_DIR="/tmp/criu-dump-bare"
BARE_LOG="/tmp/pytorch-bare.log"

ROUNDS=5

mkdir -p "$RESULT_DIR"

echo "============================================"
echo "  PyTorch 推理冷启动 vs CRIU Restore 对比测试"
echo "============================================"
echo ""

# ============================================================
# Part 1: 冷启动基准测试 (多轮)
# ============================================================
echo "--- Part 1: 冷启动基准测试 (${ROUNDS} 轮) ---"

coldstart_times=()
import_times=()
import_tv_times=()
model_load_times=()
first_infer_times=()

for i in $(seq 1 $ROUNDS); do
    echo "[Cold Start] Round $i/$ROUNDS ..."
    
    # 清除之前的缓存影响 - 释放 page cache
    # (不 drop caches，因为这会影响所有进程；只记录结果)
    
    # 用 time 命令精确测量从进程启动到推理完成的总时间
    # 同时运行 coldstart_bench.py 获取分阶段数据
    start_ts=$(date +%s%N)
    
    # 运行冷启动基准
    output=$($VENV_PYTHON "$SCRIPT_DIR/coldstart_bench.py" 2>&1)
    end_ts=$(date +%s%N)
    
    wall_time_ms=$(( (end_ts - start_ts) / 1000000 ))
    wall_time_s=$(echo "scale=3; $wall_time_ms / 1000" | bc)
    
    # 从 JSON 中提取分阶段数据
    bench_json="$RESULT_DIR/coldstart_bench_round${i}.json"
    cp "$RESULT_DIR/coldstart_bench.json" "$bench_json"
    
    import_torch_t=$(python3 -c "import json; d=json.load(open('$bench_json')); print(d['import_torch_s'])")
    import_tv_t=$(python3 -c "import json; d=json.load(open('$bench_json')); print(d['import_torchvision_s'])")
    model_t=$(python3 -c "import json; d=json.load(open('$bench_json')); print(d['model_load_s'])")
    infer_t=$(python3 -c "import json; d=json.load(open('$bench_json')); print(d['first_inference_s'])")
    total_t=$(python3 -c "import json; d=json.load(open('$bench_json')); print(d['total_from_script_start_s'])")
    
    import_times+=("$import_torch_t")
    import_tv_times+=("$import_tv_t")
    model_load_times+=("$model_t")
    first_infer_times+=("$infer_t")
    coldstart_times+=("$total_t")
    
    echo "  wall_time=${wall_time_s}s  script_total=${total_t}s  import_torch=${import_torch_t}s  import_tv=${import_tv_t}s  model_load=${model_t}s  infer=${infer_t}s"
done

echo ""

# ============================================================
# Part 2: CRIU Restore 端到端基准测试 (多轮)
# ============================================================
echo "--- Part 2: CRIU Restore 端到端基准测试 (${ROUNDS} 轮) ---"

# 先启动进程并创建 checkpoint
echo "[CRIU] 启动推理进程..."
rm -f "$PID_FILE" "$TRIGGER_FILE"
$VENV_PYTHON "$SCRIPT_DIR/bare_inference.py" &> "$BARE_LOG" &
BARE_PID=$!

# 等待就绪
echo "[CRIU] 等待进程就绪..."
while ! grep -q '\[READY\]' "$BARE_LOG"; do sleep 0.5; done
CRUI_PID=$(cat "$PID_FILE")
echo "[CRIU] 进程 PID=$CRUI_PID 已就绪"

# 创建 checkpoint
echo "[CRIU] 创建 checkpoint..."
rm -rf "$CHECKPOINT_DIR"
mkdir -p "$CHECKPOINT_DIR"
sudo criu dump -vvv -o "$CHECKPOINT_DIR/dump.log" -t "$CRUI_PID" --shell-job -D "$CHECKPOINT_DIR"
echo "[CRIU] Checkpoint 完成, 大小: $(du -sh "$CHECKPOINT_DIR" | cut -f1)"

restore_times=()
restore_to_infer_times=()

for i in $(seq 1 $ROUNDS); do
    echo "[CRIU] Round $i/$ROUNDS: restore + 首次推理..."
    
    rm -f "$TRIGGER_FILE"
    
    # 测量 CRIU restore 时间
    restore_start=$(date +%s%N)
    sudo criu restore -d -vvv -o "$CHECKPOINT_DIR/restore_round${i}.log" -D "$CHECKPOINT_DIR" --shell-job
    restore_end=$(date +%s%N)
    
    restore_ms=$(( (restore_end - restore_start) / 1000000 ))
    restore_s=$(echo "scale=3; $restore_ms / 1000" | bc)
    restore_times+=("$restore_s")
    
    echo "  restore_time=${restore_s}s, 触发推理验证..."
    
    # 触发推理并测量从 restore 完成到推理可用的时间
    infer_start=$(date +%s%N)
    touch "$TRIGGER_FILE"
    
    # 等待推理完成（检查 verification.json 更新）
    verification_json="$RESULT_DIR/verification_round${i}.json"
    rm -f "$verification_json"
    
    # 等待触发文件被删除（表示推理完成）
    timeout=10
    while [ -f "$TRIGGER_FILE" ] && [ $timeout -gt 0 ]; do
        sleep 0.1
        timeout=$((timeout - 1))
    done
    
    infer_end=$(date +%s%N)
    infer_ms=$(( (infer_end - infer_start) / 1000000 ))
    infer_after_restore_s=$(echo "scale=3; $infer_ms / 1000" | bc)
    restore_to_infer_times+=("$infer_after_restore_s")
    
    # 验证推理结果
    cp "$RESULT_DIR/verification.json" "$verification_json" 2>/dev/null || true
    match=$(python3 -c "import json; d=json.load(open('$verification_json')); print(d.get('match', 'N/A'))" 2>/dev/null || echo "N/A")
    
    echo "  infer_after_restore=${infer_after_restore_s}s  match=$match"
    
    # 再次 dump 以便下一轮 restore
    NEW_PID=$(cat "$PID_FILE")
    if [ -f "$PID_FILE" ] && kill -0 "$NEW_PID" 2>/dev/null; then
        echo "[CRIU] 创建下一轮 checkpoint..."
        sudo criu dump -vvv -o "$CHECKPOINT_DIR/dump_round${i}.log" -t "$NEW_PID" --shell-job -D "$CHECKPOINT_DIR"
    fi
done

echo ""

# ============================================================
# Part 3: 汇总对比
# ============================================================
echo "============================================"
echo "  对比结果汇总"
echo "============================================"

# 计算平均值
avg_coldstart=$(python3 -c "times=[${coldstart_times[*]/%/,}0]; print(f'{sum(times)/len(times):.3f}')")
avg_import=$(python3 -c "times=[${import_times[*]/%/,}0]; print(f'{sum(times)/len(times):.3f}')")
avg_import_tv=$(python3 -c "times=[${import_tv_times[*]/%/,}0]; print(f'{sum(times)/len(times):.3f}')")
avg_model_load=$(python3 -c "times=[${model_load_times[*]/%/,}0]; print(f'{sum(times)/len(times):.3f}')")
avg_first_infer=$(python3 -c "times=[${first_infer_times[*]/%/,}0]; print(f'{sum(times)/len(times):.3f}')")
avg_restore=$(python3 -c "times=[${restore_times[*]/%/,}0]; print(f'{sum(times)/len(times):.3f}')")
avg_restore_infer=$(python3 -c "times=[${restore_to_infer_times[*]/%/,}0]; print(f'{sum(times)/len(times):.3f}')")

# CRIU restore 到首次推理可用的总时间
avg_criu_total=$(python3 -c "r=${avg_restore}; i=${avg_restore_infer}; print(f'{float(r)+float(i):.3f}')")

echo ""
echo "冷启动 (平均 ${ROUNDS} 轮):"
echo "  import torch:           ${avg_import}s"
echo "  import torchvision:     ${avg_import_tv}s"
echo "  模型加载 (ResNet-50):   ${avg_model_load}s"
echo "  首次推理:               ${avg_first_infer}s"
echo "  冷启动总耗时:           ${avg_coldstart}s"
echo ""
echo "CRIU Restore (平均 ${ROUNDS} 轮):"
echo "  CRIU restore:           ${avg_restore}s"
echo "  restore 后推理:         ${avg_restore_infer}s"
echo "  restore 到推理可用:     ${avg_criu_total}s"
echo ""

# 加速比
speedup=$(python3 -c "c=${avg_coldstart}; r=${avg_criu_total}; print(f'{float(c)/float(r):.2f}x')" 2>/dev/null || echo "N/A")

echo "加速比 (冷启动 / CRIU):   ${speedup}"
echo ""

# 详细数据
echo "冷启动各轮数据:"
for i in $(seq 1 $ROUNDS); do
    idx=$((i-1))
    echo "  Round $i: total=${coldstart_times[$idx]}s (import_torch=${import_times[$idx]}s, import_tv=${import_tv_times[$idx]}s, model=${model_load_times[$idx]}s, infer=${first_infer_times[$idx]}s)"
done

echo ""
echo "CRIU Restore 各轮数据:"
for i in $(seq 1 $ROUNDS); do
    idx=$((i-1))
    echo "  Round $i: restore=${restore_times[$idx]}s, infer=${restore_to_infer_times[$idx]}s"
done

# 保存汇总
cat > "$RESULT_DIR/comparison_summary.json" << EOF
{
  "rounds": $ROUNDS,
  "coldstart": {
    "avg_total_s": $avg_coldstart,
    "avg_import_torch_s": $avg_import,
    "avg_import_torchvision_s": $avg_import_tv,
    "avg_model_load_s": $avg_model_load,
    "avg_first_infer_s": $avg_first_infer,
    "per_round": [$(IFS=,; echo "${coldstart_times[*]}")],
  },
  "criu_restore": {
    "avg_restore_s": $avg_restore,
    "avg_infer_after_restore_s": $avg_restore_infer,
    "avg_total_s": $avg_criu_total,
    "per_round_restore": [$(IFS=,; echo "${restore_times[*]}")],
    "per_round_infer": [$(IFS=,; echo "${restore_to_infer_times[*]}")]
  },
  "speedup": "$speedup"
}
EOF

echo ""
echo "汇总已保存: $RESULT_DIR/comparison_summary.json"

# 清理 - 最后一轮的进程可能还在
if [ -f "$PID_FILE" ]; then
    LAST_PID=$(cat "$PID_FILE")
    kill "$LAST_PID" 2>/dev/null || true
fi
