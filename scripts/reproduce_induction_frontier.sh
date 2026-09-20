#!/usr/bin/env bash
# =============================================================================
# Induction Frontier reproduction bundle — training + evaluation entry point.
#
# Paper: the trained-variant rows of the induction-frontier table (F4 test-50 /
#        secret-30).  With a fixed seed this re-runs the variants end to end and
#        lets a reader check the reported floor (SR = 2.0%, CA = 0.28) instead
#        of taking it on faith.
#
# This is NOT a one-command full reproduction.  What is *not* in this repository:
#   * Base weights — download Qwen3.5-9B / Qwen3.5-4B (HF layout) and point
#     BASE9 / BASE4 at them.  Without them every training command exits.
#   * Checkpoints — this script produces them.  A default sweep is 5 trainings
#     plus evaluation; one CUDA device with room for 9B LoRA (bf16) or 4B full
#     FT is enough.  Nothing here sets CUDA_VISIBLE_DEVICES; select the device
#     yourself.
#   * Training data — the five F4-only trajectory files ARE included
#     (data/fmb_trajectories/, produced by the frozen environment).  The
#     evaluation splits (F4 test-50 / secret-30) ship as frozen suites under
#     data/frozen_suites/ and are loaded through scripts/run_eval.py.
#
# Read before quoting a number:
#   * ARCHIVAL variants (see the registry at the bottom).  Two training rows in
#     the paper carry implementation defects discovered after the original
#     runs: the multi-observation row's evaluation loop never called the model
#     (every action fell through to the scripted fallback), and the full-FT row
#     trained on an input that included the ground-truth schema line and was
#     evaluated against base weights.  Both are kept here as historical records
#     — the paper keeps them "as records ... contribut[ing] no evidence" — and
#     are OFF by default.  Run them only with ARCHIVE_ROWS=1, and do not report
#     their numbers as capability results.
#   * Statistics, not bits.  Hyperparameters are each script's defaults and
#     match the original runs, but those runs were not hash-pinned, so expect a
#     statistically equivalent result, not a bit-identical one.  Record the
#     commit you ran.
#
# Hyperparameters (defaults of the scripts below; no numbers are set here):
#   LoRA 9B:    epochs=10 batch=4 lr=1e-4 r=16  (f4_scale_{200,500,1000,2000})
#   CoT 9B:     epochs=30 batch=4 lr=1e-4       (f4_cot_2000)
#   FullFT 4B:  epochs=20 batch=2 lr=2e-5       (f4_scale_2000)   [ARCHIVAL]
#   Multi-obs:  epochs=8  batch=1 lr=1e-4       (f4_scale_2000)   [ARCHIVAL]
#   Seed: 42 (global + TrainingArguments seed/data_seed), overridable via $SEED
#
# Expected for the default rows (test-50 / secret-30):
#   SR = 2.0% / 3.3%  (sole successful env = test env_006 / secret secret_020,
#                       the degenerate column-aligned layout)
#   CA = 0.28
#
# Usage (from the repository root; this script lives in scripts/):
#   bash scripts/reproduce_induction_frontier.sh           # full: train + eval
#   bash scripts/reproduce_induction_frontier.sh train      # train only
#   bash scripts/reproduce_induction_frontier.sh eval       # eval only (after training)
#   ARCHIVE_ROWS=1 bash scripts/reproduce_induction_frontier.sh   # + archival rows
# Environment overrides: BASE9, BASE4, DATA, SEED, REPRO, ARCHIVE_ROWS.
# Outputs:
#   models/$REPRO/<variant>/final
#   results/$REPRO/<variant>_{test,secret}/
#   logs/$REPRO/<step>.log
# =============================================================================
set -u
cd "$(dirname "$0")/.."   # -> repository root
ROOT=$(pwd)

SEED=${SEED:-42}
REPRO=${REPRO:-repro_20260908}
BASE9=${BASE9:-models/Qwen3.5-9B}
BASE4=${BASE4:-models/Qwen3.5-4B}
DATA=${DATA:-data/fmb_trajectories}
ARCHIVE_ROWS=${ARCHIVE_ROWS:-0}
OUT=$ROOT/models/$REPRO
RES=$ROOT/results/$REPRO
LOGS=$ROOT/logs/$REPRO
mkdir -p "$OUT" "$RES" "$LOGS"

MODE="${1:-all}"

# ── preflight: fail with a usable message, not a stack trace in a log ────────
missing=0
for m in "$BASE9" "$BASE4"; do
    if [ ! -d "$m" ]; then
        echo "MISSING base weights: $m"
        echo "  -> download the model (HF layout) or set BASE9/BASE4 to a local path."
        missing=1
    fi
done
for d in f4_scale_200 f4_scale_500 f4_scale_1000 f4_scale_2000 f4_cot_2000; do
    if [ ! -f "$DATA/$d.jsonl" ]; then
        echo "MISSING training data: $DATA/$d.jsonl"
        missing=1
    fi
done
python - <<'PY' || missing=1
import importlib.util, sys
need = ["torch", "transformers", "peft", "datasets", "numpy"]
absent = [m for m in need if importlib.util.find_spec(m) is None]
if absent:
    print("MISSING python packages:", ", ".join(absent))
    print("  -> pip install -r requirements-train.txt")
    sys.exit(1)
PY
if [ "$missing" -ne 0 ]; then
    echo "Preflight failed; nothing was run."
    exit 2
fi
echo "Preflight OK. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} MODEL=$BASE9 DATA=$DATA"

# ── runners ─────────────────────────────────────────────────────────────────
# variant: name | training command | model base | data | output dir | prompt format
run_train() {
    local name="$1" model="$2" data="$3" out="$4" fmt="$5"
    local log="$LOGS/train_${name}.log"
    echo "=== [$(date +%F_%T)] TRAIN $name -> $out ===" | tee -a "$log"
    if [ -d "$out/final" ]; then
        echo "  (final exists, skip)" | tee -a "$log"; return 0
    fi
    python "scripts/$fmt" \
        --model "$model" --data "$data" --output "$out" --seed $SEED \
        >>"$log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then echo "  FAILED rc=$rc" | tee -a "$log"; return $rc; fi
    echo "  done" | tee -a "$log"
}

run_eval() {
    local name="$1" spec="$2" split="$3" n="$4"
    local log="$LOGS/eval_${name}_${split}.log"
    local outdir="$RES/${name}_${split}"
    echo "=== [$(date +%F_%T)] EVAL $name $split (n=$n) ===" | tee -a "$log"
    if [ -f "$outdir/.repro_done" ]; then
        echo "  (.repro_done exists, skip)" | tee -a "$log"; return 0
    fi
    # No marker = previous run incomplete; clear leftovers so environments are
    # not appended twice on re-entry.
    rm -rf "$outdir"
    python scripts/run_eval.py \
        --agent "$spec" --family 4 --split "$split" --modality text \
        --n-envs "$n" --output "$outdir" \
        >>"$log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then echo "  FAILED rc=$rc" | tee -a "$log"; return $rc; fi
    touch "$outdir/.repro_done"
    echo "  done" | tee -a "$log"
}

train_and_eval_variant() {
    local name="$1" model="$2" data="$3" out="$4" fmt="$5" spec="$6"
    [ "$MODE" != "eval" ] && run_train "$name" "$model" "$data" "$out" "$fmt" || true
    if [ "$MODE" != "train" ]; then
        run_eval "$name" "$spec" test 50 || true
        run_eval "$name" "$spec" secret 30 || true
    fi
}

# ── registry ────────────────────────────────────────────────────────────────
# Default rows (paper-consistent, evidence-bearing):

# 1) LoRA 9B x 4 data scales (F4-only)
train_and_eval_variant f4_scale_200  $BASE9 $DATA/f4_scale_200.jsonl  $OUT/f4_scale_200  train_fmb_stage1_text.py "fmb-text:$BASE9:$OUT/f4_scale_200/final"
train_and_eval_variant f4_scale_500  $BASE9 $DATA/f4_scale_500.jsonl  $OUT/f4_scale_500  train_fmb_stage1_text.py "fmb-text:$BASE9:$OUT/f4_scale_500/final"
train_and_eval_variant f4_scale_1000 $BASE9 $DATA/f4_scale_1000.jsonl $OUT/f4_scale_1000 train_fmb_stage1_text.py "fmb-text:$BASE9:$OUT/f4_scale_1000/final"
train_and_eval_variant f4_scale_2000 $BASE9 $DATA/f4_scale_2000.jsonl $OUT/f4_scale_2000 train_fmb_stage1_text.py "fmb-text:$BASE9:$OUT/f4_scale_2000/final"

# 2) CoT LoRA 9B (2000 F4-only)
train_and_eval_variant f4_cot_2000 $BASE9 $DATA/f4_cot_2000.jsonl $OUT/f4_cot_2000 train_fmb_cot.py "fmb-text:$BASE9:$OUT/f4_cot_2000/final:cot"

# ARCHIVAL rows — historical records of two defective runs; NOT evidence.
if [ "$ARCHIVE_ROWS" = "1" ]; then
    echo "=== ARCHIVAL rows enabled: outputs are records, not capability results ==="
    # 3) Full FT 4B (2000 F4-only).  Defect: training input carried the
    #    ground-truth schema line, and the evaluation loaded base weights
    #    (no adapter_config.json).  See the script's header note.
    train_and_eval_variant f4_fullft_2000 $BASE4 $DATA/f4_scale_2000.jsonl $OUT/f4_fullft_2000 train_fmb_fullft.py "fmb-text:$BASE4:$OUT/f4_fullft_2000/final:fullft"
    # 4) Multi-obs LoRA 9B (2000, grouped).  Defect: the evaluation loop never
    #    called the model, so every action fell through to the scripted
    #    fallback.
    train_and_eval_variant fmb_multiobs_9b $BASE9 $DATA/f4_scale_2000.jsonl $OUT/fmb_multiobs_9b train_fmb_multiobs.py "fmb-text:$BASE9:$OUT/fmb_multiobs_9b/final:multiobs"
else
    echo "ARCHIVAL rows skipped (run with ARCHIVE_ROWS=1 for the historical records)."
fi

echo "=== ALL DONE $(date +%F_%T) ==="
echo "Default rows should land on SR test=2.0% (sole success env_006), secret=3.3% (secret_020), CA=0.28."
echo "Per-variant SR/CA appear in the summary block at the end of logs/$REPRO/eval_*.log"
echo "(recompute from the jsonl for the authoritative numbers)."
