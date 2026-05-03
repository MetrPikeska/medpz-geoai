"""Spustí celý pipeline: detekce → prostorová analýza → parking analýza.

Použití:
    python src/run_all.py                # celý pipeline
    python src/run_all.py --skip-detect  # přeskočí detekci (použije existující vehicles.gpkg)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    {
        "name": "Detekce vozidel",
        "cmd": ["python", "src/detect.py", "--conf", "0.25", "--slice-size", "640", "--overlap", "0.2"],
        "skip_flag": "skip_detect",
        "output": Path("data/vectors/vehicles.gpkg"),
    },
    {
        "name": "Prostorová analýza (Voronoi + korelace)",
        "cmd": ["python", "src/analyze.py"],
        "skip_flag": None,
        "output": Path("data/vectors/voronoi.gpkg"),
    },
    {
        "name": "Analýza parkování (clustering, ulice, vzdálenost)",
        "cmd": ["python", "src/parking_analysis.py"],
        "skip_flag": None,
        "output": Path("outputs/parking_clusters.gpkg"),
    },
]


def run_step(step: dict) -> bool:
    print(f"\n{'='*60}")
    print(f"  {step['name']}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(step["cmd"], text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n✗ Krok selhal (exit {result.returncode}) po {elapsed:.0f}s")
        return False
    print(f"\n✓ Hotovo za {elapsed:.0f}s")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Kompletní pipeline detekce a analýzy vozidel")
    parser.add_argument("--skip-detect", action="store_true",
                        help="Přeskočí detekci, použije existující vehicles.gpkg")
    args = parser.parse_args()

    print("Pipeline: Detekce vozidel z ortofota – Olomouc 2016")
    print(f"Python: {sys.executable}")

    t_total = time.time()
    for step in STEPS:
        if args.skip_detect and step["skip_flag"] == "skip_detect":
            out = step["output"]
            if out.exists():
                print(f"\n[přeskočeno] {step['name']} – {out} existuje")
            else:
                print(f"\n✗ --skip-detect: {out} neexistuje, spusť nejprve bez --skip-detect")
                sys.exit(1)
            continue

        if not run_step(step):
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Pipeline dokončen za {(time.time()-t_total)/60:.1f} min")
    print(f"{'='*60}")
    print("\nVýstupy pro QGIS:")
    print("  data/vectors/voronoi.gpkg      – Voronoi zóny + statistiky")
    print("  outputs/parking_clusters.gpkg  – DBSCAN parkoviště")
    print("\nGrafy:")
    for p in sorted(Path("outputs").glob("*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
