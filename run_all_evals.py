"""
Runs eval.py for all combinations of main model × judge model.

Main models:
  - Qwen/Qwen2.5-7B-Instruct
  - huihui-ai/Qwen2.5-7B-Instruct-abliterated

Judge models:
  - meta-llama/Llama-3.2-1B-Instruct
  - Qwen/Qwen2.5-7B-Instruct
  - huihui-ai/Qwen2.5-7B-Instruct-abliterated

6 total runs, executed sequentially.

Results land in results/ as:
  {mode}_results_{main_slug}_judge-{judge_slug}.csv

Usage:
    conda activate vllm_llama
    python run_all_evals.py [--backend vllm] [--judge-gpu 2] [--dry-run]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

MAIN_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "huihui-ai/Qwen2.5-7B-Instruct-abliterated",
]

JUDGE_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "huihui-ai/Qwen2.5-7B-Instruct-abliterated",
]

EVAL_SCRIPT = Path(__file__).parent / "eval.py"


def slug(model_id: str) -> str:
    return model_id.split("/")[-1]


def expected_outputs(main_model: str, judge_model: str) -> list[Path]:
    results_dir = Path(__file__).parent / "results"
    ms, js = slug(main_model), slug(judge_model)
    modes = ["baseline", "secure_prompt", "judge_only", "defended"]
    return [results_dir / f"{mode}_results_{ms}_judge-{js}.csv" for mode in modes]


def run_combo(main_model: str, judge_model: str, backend: str, judge_gpu: int, dry_run: bool) -> bool:
    ms, js = slug(main_model), slug(judge_model)
    print(f"\n{'='*70}")
    print(f"  main  : {ms}")
    print(f"  judge : {js}")
    print(f"{'='*70}")

    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--backend", backend,
        "--model", main_model,
        "--judge-model", judge_model,
        "--judge-gpu", str(judge_gpu),
    ]
    print(f"  cmd   : {' '.join(cmd)}\n")

    if dry_run:
        print("  [dry-run] skipping.")
        return True

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(EVAL_SCRIPT.parent))
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"\n  [FAILED] exit code {result.returncode} after {elapsed/60:.1f} min")
        return False

    missing = [p for p in expected_outputs(main_model, judge_model) if not p.exists()]
    if missing:
        print(f"\n  [WARNING] expected outputs not found:")
        for p in missing:
            print(f"    {p.name}")
    else:
        print(f"\n  [OK] finished in {elapsed/60:.1f} min — all 4 CSVs written")
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--judge-gpu", type=int, default=2,
                        help="Physical GPU index for the judge model")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running them")
    args = parser.parse_args()

    combos = [(m, j) for m in MAIN_MODELS for j in JUDGE_MODELS]
    total = len(combos)
    print(f"\nRunning {total} eval combinations  (backend={args.backend}, judge-gpu={args.judge_gpu})")

    failures = []
    wall_t0 = time.perf_counter()

    for i, (main_model, judge_model) in enumerate(combos, 1):
        print(f"\n[{i}/{total}]", end="")
        ok = run_combo(main_model, judge_model, args.backend, args.judge_gpu, args.dry_run)
        if not ok:
            failures.append((main_model, judge_model))

    total_min = (time.perf_counter() - wall_t0) / 60
    print(f"\n{'='*70}")
    print(f"Done — {total - len(failures)}/{total} runs succeeded in {total_min:.1f} min total")

    if failures:
        print("\nFailed combinations:")
        for m, j in failures:
            print(f"  main={slug(m)}  judge={slug(j)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
