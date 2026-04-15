from __future__ import annotations
"""Download pre-computed data from GitHub releases."""

import gzip
import shutil
import subprocess
import urllib.request
from pathlib import Path

from smt.config import REPO_ROOT
from smt.display import console

# Configure these before creating the GitHub release
GITHUB_REPO = "thebassil/smt-reproduce"
RELEASE_TAG = "v0.1.0"

DB_ASSET = "results.sqlite.gz"
BENCHMARKS_ASSET = "benchmarks.tar.gz"


def _gh_available() -> bool:
    try:
        subprocess.run(["gh", "--version"], capture_output=True)
        return True
    except FileNotFoundError:
        return False


def _download_asset(asset_name: str, dest: Path) -> bool:
    """Try gh CLI first, fall back to urllib."""
    if _gh_available():
        console.print(f"[dim]Downloading {asset_name} via gh...[/dim]")
        proc = subprocess.run(
            [
                "gh", "release", "download", RELEASE_TAG,
                "--repo", GITHUB_REPO,
                "--pattern", asset_name,
                "--dir", str(dest.parent),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return True
        console.print(f"[yellow]gh download failed, trying direct URL...[/yellow]")

    url = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/{asset_name}"
    console.print(f"[dim]Downloading {url}...[/dim]")
    try:
        urllib.request.urlretrieve(url, str(dest.parent / asset_name))
        return True
    except Exception as e:
        console.print(f"[red]Download failed:[/red] {e}")
        console.print(
            f"\nManual download: go to https://github.com/{GITHUB_REPO}/releases"
            f"\nand place {asset_name} in {dest.parent}/"
        )
        return False


def download_data(db_only: bool = False, benchmarks: bool = False) -> None:
    data_dir = REPO_ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    db_path = data_dir / "results.sqlite"
    if db_path.exists() and db_path.stat().st_size > 1000:
        size_mb = round(db_path.stat().st_size / 1e6, 1)
        console.print(f"[green]Database already present:[/green] {db_path} ({size_mb} MB)")
    else:
        gz_path = data_dir / DB_ASSET
        if _download_asset(DB_ASSET, gz_path):
            if gz_path.exists():
                console.print("[dim]Decompressing...[/dim]")
                with gzip.open(str(gz_path), "rb") as f_in:
                    with open(str(db_path), "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                gz_path.unlink()
                size_mb = round(db_path.stat().st_size / 1e6, 1)
                console.print(f"[green]Database ready:[/green] {db_path} ({size_mb} MB)")

    if db_only:
        return

    if benchmarks:
        bench_dir = data_dir / "benchmarks"
        if bench_dir.exists() and any(bench_dir.iterdir()):
            console.print(f"[green]Benchmarks already present:[/green] {bench_dir}")
        else:
            tar_path = data_dir / BENCHMARKS_ASSET
            if _download_asset(BENCHMARKS_ASSET, tar_path):
                if tar_path.exists():
                    console.print("[dim]Extracting benchmarks...[/dim]")
                    shutil.unpack_archive(str(tar_path), str(data_dir))
                    tar_path.unlink()
                    console.print(f"[green]Benchmarks ready:[/green] {bench_dir}")

    console.print("\n[bold]Done.[/bold] Run `smt status` to verify.")
