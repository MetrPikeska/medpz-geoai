"""Spustí celý pipeline pro všechna 4 ortofota: clip → detekce → analýza → parking.

Použití:
    python src/run_all.py                        # celý pipeline pro všechna ortofota
    python src/run_all.py --imagery 1 2          # jen vybraná ortofota
    python src/run_all.py --skip-detect          # přeskočí detekci (použije existující gpkg)
    python src/run_all.py --imagery 1 --skip-detect
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def make_steps(idx: int) -> list[dict]:
    tif = Path(f"data/raw/Olomouc_imagery_{idx}.tif")
    adr = Path(f"data/adr_imagery_{idx}.geojson")
    vehicles = Path(f"data/vectors/vehicles_{idx}.gpkg")
    voronoi = Path(f"data/vectors/voronoi_{idx}.gpkg")
    parking = Path(f"outputs/parking_clusters_{idx}.gpkg")
    suffix = f"_{idx}"

    return [
        {
            "name": f"[{idx}] Ořez adresních bodů",
            "cmd": ["python", "src/clip_addresses.py"],
            "skip_flag": "skip_clip",
            "output": adr,
        },
        {
            "name": f"[{idx}] Detekce vozidel – {tif.name}",
            "cmd": [
                "python", "src/detect.py",
                "--input", str(tif),
                "--output-vectors", str(vehicles),
                "--conf", "0.25", "--slice-size", "640", "--overlap", "0.2",
            ],
            "skip_flag": "skip_detect",
            "output": vehicles,
        },
        {
            "name": f"[{idx}] Prostorová analýza (Voronoi + korelace)",
            "cmd": [
                "python", "src/analyze.py",
                "--vehicles", str(vehicles),
                "--addresses", str(adr),
                "--output-voronoi", str(voronoi),
                "--output-map", f"outputs/analysis_map{suffix}.png",
                "--output-stats", f"outputs/statistics{suffix}.csv",
            ],
            "skip_flag": None,
            "output": voronoi,
        },
        {
            "name": f"[{idx}] Analýza parkování (clustering, ulice)",
            "cmd": [
                "python", "src/parking_analysis.py",
                "--vehicles", str(vehicles),
                "--voronoi", str(voronoi),
                "--suffix", suffix,
            ],
            "skip_flag": None,
            "output": parking,
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
    parser = argparse.ArgumentParser(description="Pipeline detekce a analýzy vozidel – všechna ortofota")
    parser.add_argument("--imagery", type=int, nargs="+", choices=[1, 2, 3, 4], default=[1, 2, 3, 4],
                        help="Která ortofota zpracovat (default: všechna)")
    parser.add_argument("--skip-detect", action="store_true",
                        help="Přeskočí detekci, použije existující vehicles_N.gpkg")
    args = parser.parse_args()

    print("Pipeline: Detekce vozidel z ortofota – Olomouc 2016")
    print(f"Ortofota: {args.imagery}")
    print(f"Python: {sys.executable}")

    t_total = time.time()

    # Clip addresses once before per-imagery steps
    clip_done = False

    for idx in args.imagery:
        print(f"\n{'#'*60}")
        print(f"#  ORTOFOTO {idx}")
        print(f"{'#'*60}")

        steps = make_steps(idx)

        for step in steps:
            out = step["output"]

            if step["skip_flag"] == "skip_clip":
                if not clip_done:
                    if out.exists():
                        print(f"\n[přeskočeno] Ořez adres – {out} existuje")
                    else:
                        if not run_step(step):
                            sys.exit(1)
                    clip_done = True
                else:
                    print(f"\n[přeskočeno] Ořez adres – již provedeno")
                continue

            if args.skip_detect and step["skip_flag"] == "skip_detect":
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
    print("\nVýstupy (QGIS):")
    for idx in args.imagery:
        v = Path(f"data/vectors/voronoi_{idx}.gpkg")
        p = Path(f"outputs/parking_clusters_{idx}.gpkg")
        if v.exists():
            print(f"  {v}")
        if p.exists():
            print(f"  {p}")
    print("\nGrafy:")
    for p in sorted(Path("outputs").glob("*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
