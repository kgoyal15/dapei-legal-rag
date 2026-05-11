"""
Step 1: Download contracts.

Run: python scripts/01_download_data.py

Downloads:
  - CUAD dataset (510 annotated contracts) from HuggingFace
  - 200 material contracts from SEC EDGAR
"""

import sys
sys.path.insert(0, ".")

from config import RAW_DIR
from src.data.cuad_loader import load_cuad
from src.data.edgar_downloader import bulk_download
from rich.console import Console

console = Console()


def main():
    console.rule("[bold]Step 1: Downloading contract data[/bold]")

    # CUAD — best starting point: already annotated, 510 contracts
    console.print("\n[cyan]1/2 Downloading CUAD (510 annotated contracts)...[/cyan]")
    cuad_dir = RAW_DIR / "cuad"
    contracts = load_cuad(cuad_dir, max_contracts=511)
    console.print(f"[green]CUAD: {len(contracts)} contracts saved to {cuad_dir}[/green]")

    # EDGAR — real commercial contracts for diversity
    console.print("\n[cyan]2/2 Downloading EDGAR material contracts...[/cyan]")
    edgar_dir = RAW_DIR / "edgar"
    paths = bulk_download(edgar_dir, max_contracts=200)
    console.print(f"[green]EDGAR: {len(paths)} contracts saved to {edgar_dir}[/green]")

    console.rule("[bold green]Done. Run scripts/02_build_pipeline.py next.[/bold green]")


if __name__ == "__main__":
    main()
