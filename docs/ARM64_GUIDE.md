# Arm64 setup and validation guide

## Prerequisites

- An Arm64 (`aarch64`) Linux VM.
- A local image directory and a Phase 3 checkpoint directory.

## Install

On the selected Arm64 Linux VM:

```bash
git clone https://github.com/nakulvyas21/suvi-arm-optimization.git
cd suvi-arm-optimization
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run SUVI

```bash
python scripts/benchmark_arm64.py \
  --adapter-path /secure/checkpoints/phase3-checkpoint \
  --image-dir /secure/permitted-demo-images \
  --artifact-dir artifacts/20260812_arm_cloud_seed-20260809 \
  --prompt 'View:' \
  --max-new-tokens 8 \
  --warmups 1 \
  --repetitions 2 \
  --threads 8 \
  --llm-dtype bfloat16
```

## Run the dual-tower comparator

```bash
python scripts/benchmark_dual_tower.py \
  --image-dir /secure/permitted-demo-images \
  --artifact-dir artifacts/20260812_arm_cloud_dual_tower \
  --prompt 'View:' \
  --max-new-tokens 8 \
  --warmups 1 \
  --repetitions 2 \
  --threads 8 \
  --dtype bfloat16
```

See [results](RESULTS.md).
