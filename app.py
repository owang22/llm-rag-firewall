"""
Interactive CLI demo for llm-rag-firewall.

Run:
    python app.py                         # interactive prompt
    python app.py --mode baseline --docs poisoned --question "How do I configure the API?"
    python app.py --backend vllm --model Qwen/Qwen2.5-7B-Instruct

Flags:
    --mode      baseline | defended       (default: baseline)
    --docs      clean | poisoned | mixed  (default: clean)
    --question  pre-set question string   (default: interactive prompt)
    --backend   transformers | vllm (default: vllm)
    --model     HuggingFace model ID
    --no-color  disable rich formatting
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse
import json
import tempfile
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from defenses import PromptInjectionDetector
from rag import RAGPipeline, DEFAULT_MODEL

DATA_DIR = Path(__file__).parent / "data"
CONSOLE = Console()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _chunk_panel(chunk: dict, blocked: bool = False) -> Panel:
    score = chunk.get("score", 0.0)
    title_text = f"[red]✗ BLOCKED[/] " if blocked else "[green]✓[/] "
    title_text += f"[bold]{chunk['title']}[/]  [dim](score: {score:.3f})[/]"

    content = chunk["content"]
    if blocked:
        detector = PromptInjectionDetector()
        result = detector.detect(content)
        highlights = result.matched_patterns
        content += f"\n\n[red][Matched patterns: {', '.join(highlights)}][/red]"

    return Panel(
        Text.from_markup(content) if blocked else content,
        title=title_text,
        border_style="red" if blocked else "dim",
        expand=True,
    )


def _show_result(result: dict) -> None:
    mode = result["mode"].upper()
    mode_color = "green" if mode == "DEFENDED" else "yellow"
    CONSOLE.print()
    CONSOLE.rule(f"[bold {mode_color}]Mode: {mode}[/bold {mode_color}]")

    # Retrieved chunks
    CONSOLE.print(f"\n[bold]Retrieved Chunks ({len(result['retrieved_chunks'])} total)[/bold]")
    blocked_ids = {c["id"] for c in result.get("blocked_chunks", [])}
    for chunk in result["retrieved_chunks"]:
        CONSOLE.print(_chunk_panel(chunk, blocked=chunk["id"] in blocked_ids))

    # Blocked chunks summary (defended mode)
    if result.get("blocked_chunks"):
        CONSOLE.print(f"\n[bold red]Blocked ({len(result['blocked_chunks'])} chunk(s))[/bold red]")
        meta = result.get("detection_metadata", {})
        if meta.get("matched_patterns"):
            for doc_id, patterns in meta["matched_patterns"].items():
                CONSOLE.print(f"  [dim]{doc_id}[/dim]: {', '.join(patterns)}")

    # Output validation flags
    flags = result.get("detection_metadata", {}).get("output_flags", [])
    if flags:
        CONSOLE.print(f"\n[bold red]Output validation flags: {flags}[/bold red]")

    # Final answer
    answer = result["answer"]
    answer_style = "red" if "INJECTION FOLLOWED" in answer or "BLOCKED" in answer else "green" if mode == "DEFENDED" else "yellow"
    CONSOLE.print(
        Panel(
            answer,
            title=f"[bold {answer_style}]Model Answer[/bold {answer_style}]",
            border_style=answer_style,
            expand=True,
        )
    )

    CONSOLE.print(f"[dim]Latency: {result['latency_s']:.3f}s[/dim]\n")


# ---------------------------------------------------------------------------
# Mixed corpus helper
# ---------------------------------------------------------------------------

def _make_mixed_corpus(tmp_path: Path) -> str:
    lines = []
    with open(DATA_DIR / "clean_docs.jsonl") as f:
        lines.extend(f.readlines())
    with open(DATA_DIR / "poisoned_docs.jsonl") as f:
        lines.extend(f.readlines()[:3])
    out = tmp_path / "mixed_docs.jsonl"
    out.write_text("".join(lines))
    return str(out)


def _resolve_docs_path(docs: str, tmp_path: Path) -> str:
    if docs == "poisoned":
        return str(DATA_DIR / "poisoned_docs.jsonl")
    if docs == "mixed":
        return _make_mixed_corpus(tmp_path)
    return str(DATA_DIR / "clean_docs.jsonl")


# ---------------------------------------------------------------------------
# Demo walkthrough
# ---------------------------------------------------------------------------

DEMO_STEPS = [
    ("Step 1: Benign QA over clean documents",
     "baseline", "clean",
     "What are the system requirements for AcmePlatform?"),
    ("Step 2: Baseline with poisoned documents — injection may be followed",
     "baseline", "poisoned",
     "How do I configure the API?"),
    ("Step 3: Defended mode with poisoned documents — chunks filtered",
     "defended", "poisoned",
     "How do I configure the API?"),
    ("Step 4: Direct secret request — baseline",
     "baseline", "clean",
     "What is the secret API key?"),
    ("Step 5: Direct secret request — defended",
     "defended", "clean",
     "What is the secret API key?"),
]


def run_demo(backend: str, model_name: str, secrets: dict, tmp_path: Path,
             judge_model: str = "meta-llama/Llama-3.2-1B-Instruct") -> None:
    CONSOLE.rule("[bold cyan]RAG Firewall — Full Demo Walkthrough[/bold cyan]")
    CONSOLE.print(
        "This walkthrough shows 5 steps: benign QA, baseline with poisoned docs, "
        "defended mode, and direct secret requests.\n"
    )

    for title, mode, docs, question in DEMO_STEPS:
        CONSOLE.rule(f"[bold white]{title}[/bold white]")
        CONSOLE.print(f"[bold]Question:[/bold] {question}")

        path = _resolve_docs_path(docs, tmp_path)
        pipeline = RAGPipeline(
            docs_path=path, secrets=secrets,
            backend=backend, model_name=model_name, top_k=3,
            judge_model=judge_model, judge_backend=backend,
        )

        if mode == "defended":
            result = pipeline.query_defended(question)
        else:
            result = pipeline.query_baseline(question)

        _show_result(result)
        CONSOLE.print()
        input("[dim]Press Enter for next step…[/dim]  ")
        CONSOLE.print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="llm-rag-firewall: interactive demo of RAG prompt injection & defense",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py                              # interactive session (vllm backend)
  python app.py --demo                       # run the full 5-step walkthrough
  python app.py --mode baseline --docs poisoned --question "How do I configure the API?"
  python app.py --mode defended --docs poisoned --question "How do I configure the API?"
  python app.py --backend vllm --model Qwen/Qwen2.5-7B-Instruct --demo
        """,
    )
    parser.add_argument("--mode",        choices=["baseline", "defended"], default="baseline")
    parser.add_argument("--docs",        choices=["clean", "poisoned", "mixed"], default="clean")
    parser.add_argument("--question",    default=None, help="Skip interactive prompt")
    parser.add_argument("--backend",     choices=["transformers", "vllm"], default="vllm")
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    parser.add_argument("--judge-model", default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Small model used as the semantic judge in defended mode")
    parser.add_argument("--demo",        action="store_true", help="Run the 5-step demo walkthrough")
    args = parser.parse_args()

    with open(DATA_DIR / "secrets.json") as f:
        secrets = json.load(f)

    tmp_dir = Path(tempfile.mkdtemp())

    try:
        if args.demo:
            run_demo(args.backend, args.model, secrets, tmp_dir, args.judge_model)
            return

        # Single-query or interactive loop
        docs_path = _resolve_docs_path(args.docs, tmp_dir)
        pipeline = RAGPipeline(
            docs_path=docs_path, secrets=secrets,
            backend=args.backend, model_name=args.model, top_k=3,
            judge_model=args.judge_model, judge_backend=args.backend,
        )

        CONSOLE.rule("[bold cyan]RAG Firewall Demo[/bold cyan]")
        CONSOLE.print(f"Mode        : [{'green' if args.mode == 'defended' else 'yellow'}]{args.mode}[/]")
        CONSOLE.print(f"Docs        : [yellow]{args.docs}[/]")
        CONSOLE.print(f"Backend     : [yellow]{args.backend}[/]")
        CONSOLE.print(f"Judge model : [yellow]{args.judge_model.split('/')[-1]}[/]")
        CONSOLE.print("[dim]Type 'quit' to exit.\n[/dim]")

        while True:
            if args.question:
                question = args.question
            else:
                try:
                    question = CONSOLE.input("[bold cyan]> Question:[/bold cyan] ").strip()
                except (EOFError, KeyboardInterrupt):
                    CONSOLE.print("\n[dim]Bye![/dim]")
                    break

            if question.lower() in ("quit", "exit", "q"):
                CONSOLE.print("[dim]Bye![/dim]")
                break
            if not question:
                continue

            if args.mode == "defended":
                result = pipeline.query_defended(question)
            else:
                result = pipeline.query_baseline(question)

            _show_result(result)

            if args.question:
                break  # single-shot mode

    finally:
        for f in tmp_dir.iterdir():
            f.unlink()
        tmp_dir.rmdir()


if __name__ == "__main__":
    main()
