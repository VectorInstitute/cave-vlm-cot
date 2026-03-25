#!/bin/bash
#SBATCH --job-name=cite_verify_vlm_cot
#SBATCH --gres=gpu:a100:3
#SBATCH --time=3-00:00:00
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module purge
module load python/3.10

source /fs01/home/sneharao/cbm-vlms/fresh_env/bin/activate

# ── Quick sanity checks ────────────────────────────────────────────────────────
echo "==== PYTHON DIAGNOSTIC ===="
which python && python --version

python - <<'EOF'
import sys, torch
print("Executable:", sys.executable)

for name, mod in [("transformers", "transformers"), ("torch", "torch"), ("unsloth", "unsloth")]:
    try:
        __import__(mod)
        print(f"{name} OK")
    except Exception as e:
        print(f"{name} FAIL: {e}")

print(f"\nGPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    free, total = torch.cuda.mem_get_info(i)
    print(f"  cuda:{i}  {props.name}  {total/1e9:.0f}GB total  {free/1e9:.1f}GB free")

EOF

# ── Phoenix env check ─────────────────────────────────────────────────────────
echo "PHOENIX_HOST='$PHOENIX_HOST'"
echo "PHOENIX_PORT='$PHOENIX_PORT'"
echo "PHOENIX_COLLECTOR_ENDPOINT='$PHOENIX_COLLECTOR_ENDPOINT'"
echo "PHOENIX_CLIENT_HEADERS='$PHOENIX_CLIENT_HEADERS'"

pip show arize-phoenix-client 2>/dev/null || pip show arize-phoenix 2>/dev/null

python -c "import inspect; from phoenix.client import Client; print('Client signature:', inspect.signature(Client.__init__))"
python -c "from phoenix.client import Client; c = Client(); print('Phoenix base URL:', c._client.base_url)"

# ── Run experiment (single process, all 3 GPUs pinned inside experiments.py) ──
echo ""
echo "Starting experiment at $(date)"
python experiments.py
echo "Finished at $(date)"