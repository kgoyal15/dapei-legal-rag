"""
Step 5 (repair): Re-attach CUAD QA annotations to already-saved contracts.

Run this if diagnose.py shows "Contracts WITH annotations: 0/511".

What it does:
  1. Downloads CUADv1.json QA annotations (cached after first run)
  2. Re-iterates the HuggingFace dataset to recover each contract's original filename
  3. Matches filenames against annotation titles and updates saved JSON files
  4. Re-creates the benchmark so evaluation can proceed

Run: python scripts/05_repair_annotations.py
"""
import sys
sys.path.insert(0, ".")

from config import RAW_DIR, BENCHMARK_DIR
from src.data.cuad_loader import repair_cuad_annotations
from rich.console import Console

console = Console()
console.rule("[bold cyan]Step 5: Repair CUAD annotations[/bold cyan]")

cuad_dir = RAW_DIR / "cuad"
if not cuad_dir.exists() or not list(cuad_dir.glob("cuad_*.json")):
    console.print("[red]No saved CUAD contracts found. Run scripts/01_download_data.py first.[/red]")
    sys.exit(1)

updated = repair_cuad_annotations(cuad_dir)

if updated == 0:
    console.print("[yellow]No contracts were updated. Possible causes:[/yellow]")
    console.print("  - All contracts already have annotations (run diagnose.py to confirm)")
    console.print("  - Annotation title format does not match dataset filenames")
    console.print("  - QA annotations download failed (check network access)")
else:
    console.print(f"[green]Updated {updated} contracts.[/green]")
    console.print("\nRe-creating benchmark with repaired annotations...")
    try:
        import runpy
        runpy.run_path("scripts/03_create_benchmark.py", run_name="__main__")
    except Exception as e:
        console.print(f"[yellow]Could not auto-run benchmark script: {e}[/yellow]")
        console.print("Run manually: python scripts/03_create_benchmark.py")

console.rule("[bold]Done — run diagnose.py to verify[/bold]")
