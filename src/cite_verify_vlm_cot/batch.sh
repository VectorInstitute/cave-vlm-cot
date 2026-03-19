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

print("V50 Changes")
print("solver.py — REASONING hints rewritten (both templates)")
print("verifier.py — UNKNOWN verdict final fallback")
print("v51 changes")
print("SUPPLY-LIST EXPERIMENT QUESTIONS — scoped to exact phrase 'Using only these supplies' only (Ex:51/69), not the broader 'best answer' that misfired on Ex:3")
print("LITERAL YES/NO or TRUE/FALSE QUESTIONS — scoped to choices that are exactly yes/no/true/false (only 8 questions), not all 57 two-choice questions that broke Ex:38 in v50")
EOF

echo "=========================="

# print("\nChanges in this run (V45):")
# print("  Retriever — choice-augmented web searches cut from choices[:4] → choices[:2]. That removes up to 6 DuckDuckGo calls per question (~25–30s off mean latency). The first two choices are the most semantically useful variants anyway")
# print("  Citation Injector - evidence cap raised 10 → 15. Conclusion-supporting chunks that were being evicted from the pool will now survive into the NLI grounding check.")
# print("  Verifier: max_new_tokens=2048 → 1024. The 5-step CoT + structured output block rarely exceeds 800 tokens.")
# print("  Solver kwargs: do_sample=False (~3s saved) — greedy decoding is ~15% faster than sampling for Llama on structured-format tasks.")
# print("  Planner stays on GPU0, Solver pinned to cuda:1, Verifier pinned to cuda:2")
# print("  Retriever: Parallel web search, Replaced with ThreadPoolExecutor that fires all jobs simultaneously. Since DDG is I/O-bound, threads work well. All calls collapse to the time of the single slowest one (~5s instead of ~35s).")
# print("  Retriever: CrossEncoder loads on CPU by default. Forced to cuda so the predict() calls on 15-pair batches run ~10× faster.")
# print("_web_search_with_choices() — parallel ThreadPoolExecutor with 0.3s stagger between submissions, choices[:4] restored, 1. Staggered submission (0.3s between jobs) — instead of all 5 calls hitting DDG at the same instant, they're spread 0.3s apart. Total overhead: ~1.2s vs the ~30s saved by parallelism. DDG never sees a burst spike.")
# print("web_search() — exponential backoff (1s, 2s, 4s) on rate-limit errors, 2. Exponential backoff in web_search() — if DDG does rate-limit (returns 202/blocked/timeout), it retries after 1s, then 2s, then 4s before giving up. This means a rate-limit hit costs ~1-3s instead of silently returning empty results and killing grounding/recall.")

# print("\nChanges in this run (V44):")
# print("  Retriever query expansion changes")
# print("  Solver image prompt changes (Cite images in REASONING)")
# print("  Verifier prompt changes (Structured CoT hallucination detection)")

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