"""Evaluace kvality detekce vozidel bez ground truth.

Měřené metriky (bez GT):
  duplicate_rate  – podíl boxů, které mají IOS > 0.3 s jiným boxem stejné třídy
  size_ratio      – medián plochy velkých vozidel / medián plochy malých vozidel
                    (vyšší = lepší separace tříd → lepší klasifikace)
  mw_p            – Mann-Whitney p-value pro oddělení distribucí ploch (nižší = lepší)

Sweepované parametry:
  conf            – confidence threshold YOLOv8
  overlap         – poměr překryvu dlaždic (ovlivňuje duplicity)
  iou_match       – postprocess_match_threshold v SAHI (nižší = agresivnější sloučení)
"""

import argparse
import itertools
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from shapely.geometry import box as sbox

warnings.filterwarnings("ignore")

VEHICLES_GPKG = Path("data/vectors/vehicles.gpkg")
TEST_INPUT = Path("data/raw/Olomouc_imagery_1.tif")
TEST_CROP = Path("data/processed/eval_crop.tif")
MODEL = "yolov8n-obb.pt"
OUT_DIR = Path("outputs")
VEHICLE_CLASSES = {"small vehicle", "large vehicle"}

# Parametrický grid pro sweep
SWEEP_CONF = [0.15, 0.25, 0.35]
SWEEP_OVERLAP = [0.1, 0.2, 0.3]
SWEEP_IOU_MATCH = [0.3, 0.5, 0.7]
SWEEP_SLICE = 512  # fixní pro rychlost


# ---------------------------------------------------------------------------
# Metriky bez ground truth
# ---------------------------------------------------------------------------

def box_ios(a, b) -> float:
    """Intersection over Smaller area."""
    inter = a.intersection(b).area
    smaller = min(a.area, b.area)
    return inter / smaller if smaller > 0 else 0.0


def compute_duplicate_rate(gdf: gpd.GeoDataFrame, ios_thresh: float = 0.3) -> dict:
    """Podíl boxů s IOS > ios_thresh vůči jinému boxu stejné třídy."""
    results = {"total": 0.0, "small vehicle": 0.0, "large vehicle": 0.0}
    for cls in ("small vehicle", "large vehicle"):
        sub = gdf[gdf["class"] == cls].reset_index(drop=True)
        if len(sub) < 2:
            continue
        geoms = list(sub.geometry)
        flagged = set()
        for i in range(len(geoms)):
            for j in range(i + 1, len(geoms)):
                if box_ios(geoms[i], geoms[j]) > ios_thresh:
                    flagged.add(i)
                    flagged.add(j)
        results[cls] = len(flagged) / len(sub)
    n = len(gdf)
    n_s = (gdf["class"] == "small vehicle").sum()
    n_l = (gdf["class"] == "large vehicle").sum()
    results["total"] = (
        results["small vehicle"] * n_s + results["large vehicle"] * n_l
    ) / n if n > 0 else 0.0
    return results


def compute_size_metrics(gdf: gpd.GeoDataFrame) -> dict:
    """Separace distribucí ploch malých a velkých vozidel."""
    small = gdf[gdf["class"] == "small vehicle"]["geometry"].area
    large = gdf[gdf["class"] == "large vehicle"]["geometry"].area
    if len(small) < 2 or len(large) < 2:
        return {"size_ratio": np.nan, "mw_p": np.nan,
                "small_med_m2": np.nan, "large_med_m2": np.nan}
    _, p = mannwhitneyu(large, small, alternative="greater")
    return {
        "size_ratio": round(large.median() / small.median(), 2) if small.median() > 0 else np.nan,
        "mw_p": round(p, 4),
        "small_med_m2": round(small.median(), 2),
        "large_med_m2": round(large.median(), 2),
    }


def metrics_from_gdf(gdf: gpd.GeoDataFrame) -> dict:
    dup = compute_duplicate_rate(gdf)
    size = compute_size_metrics(gdf)
    return {
        "n_total": len(gdf),
        "n_small": int((gdf["class"] == "small vehicle").sum()),
        "n_large": int((gdf["class"] == "large vehicle").sum()),
        "dup_rate": round(dup["total"], 3),
        "dup_small": round(dup["small vehicle"], 3),
        "dup_large": round(dup["large vehicle"], 3),
        **size,
    }


# ---------------------------------------------------------------------------
# Analýza existujícího vehicles.gpkg
# ---------------------------------------------------------------------------

def analyze_existing(vehicles_path: Path) -> None:
    print("=== Analýza existujících detekcí ===")
    gdf = gpd.read_file(vehicles_path)
    print(f"Načteno {len(gdf)} vozidel ({(gdf['class']=='small vehicle').sum()} malých, "
          f"{(gdf['class']=='large vehicle').sum()} velkých)")

    m = metrics_from_gdf(gdf)
    print(f"\nDuplicity (IOS > 0.3):")
    print(f"  celkový podíl:  {m['dup_rate']:.1%}")
    print(f"  malá vozidla:   {m['dup_small']:.1%}")
    print(f"  velká vozidla:  {m['dup_large']:.1%}")
    print(f"\nSeparace tříd:")
    print(f"  medián plochy  small:  {m['small_med_m2']:.1f} m²")
    print(f"  medián plochy  large:  {m['large_med_m2']:.1f} m²")
    print(f"  poměr (large/small):   {m['size_ratio']:.2f}×")
    print(f"  Mann-Whitney p:        {m['mw_p']:.4f}"
          f"  {'✓ třídy dobře odděleny' if m['mw_p'] < 0.05 else '✗ třídy se překrývají'}")

    _plot_size_distributions(gdf, OUT_DIR / "eval_size_distribution.png",
                             title="Distribuce ploch boxů – aktuální nastavení (conf=0.25)")
    _plot_duplicate_map(gdf, OUT_DIR / "eval_duplicate_map.png")


def _plot_size_distributions(gdf: gpd.GeoDataFrame, path: Path, title: str) -> None:
    small_area = gdf[gdf["class"] == "small vehicle"].geometry.area
    large_area = gdf[gdf["class"] == "large vehicle"].geometry.area

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # Boxplot ploch
    ax = axes[0]
    ax.boxplot([small_area, large_area], labels=["small vehicle", "large vehicle"],
               patch_artist=True,
               boxprops=dict(facecolor="#4575b4", alpha=0.6),
               medianprops=dict(color="red", linewidth=2))
    ax.set_ylabel("Plocha boxu [m²]")
    ax.set_title("Distribuce ploch boxů")
    ax.grid(axis="y", alpha=0.3)

    # Histogram ploch
    ax = axes[1]
    bins = np.linspace(0, max(small_area.quantile(0.99), large_area.quantile(0.99)), 50)
    ax.hist(small_area, bins=bins, alpha=0.6, label="small vehicle", color="#4575b4", density=True)
    ax.hist(large_area, bins=bins, alpha=0.6, label="large vehicle", color="#d73027", density=True)
    ax.axvline(small_area.median(), color="#4575b4", linestyle="--", linewidth=1.5,
               label=f"median small = {small_area.median():.1f} m²")
    ax.axvline(large_area.median(), color="#d73027", linestyle="--", linewidth=1.5,
               label=f"median large = {large_area.median():.1f} m²")
    ax.set_xlabel("Plocha boxu [m²]")
    ax.set_ylabel("Hustota")
    ax.set_title("Histogram ploch (normalizovaný)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def _plot_duplicate_map(gdf: gpd.GeoDataFrame, path: Path) -> None:
    """Zvýraznění boxů s IOS > 0.3 jako duplicity."""
    ios_thresh = 0.3
    is_dup = pd.Series(False, index=gdf.index)
    for cls in ("small vehicle", "large vehicle"):
        sub = gdf[gdf["class"] == cls]
        idxs = list(sub.index)
        geoms = list(sub.geometry)
        for i in range(len(geoms)):
            for j in range(i + 1, len(geoms)):
                if box_ios(geoms[i], geoms[j]) > ios_thresh:
                    is_dup[idxs[i]] = True
                    is_dup[idxs[j]] = True

    gdf = gdf.copy()
    gdf["is_dup"] = is_dup

    fig, ax = plt.subplots(figsize=(12, 9))
    gdf[~gdf["is_dup"]].assign(geometry=gdf[~gdf["is_dup"]].geometry.centroid).plot(
        ax=ax, color="steelblue", markersize=3, alpha=0.5, label="unikátní")
    gdf[gdf["is_dup"]].assign(geometry=gdf[gdf["is_dup"]].geometry.centroid).plot(
        ax=ax, color="red", markersize=6, alpha=0.8, label="duplicita")
    ax.set_title(f"Mapa duplicitních detekcí (IOS > {ios_thresh})\n"
                 f"{is_dup.sum()} z {len(gdf)} boxů označeno jako duplicita "
                 f"({is_dup.mean():.1%})")
    ax.legend(fontsize=9)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ---------------------------------------------------------------------------
# Parametrický sweep
# ---------------------------------------------------------------------------

def _make_test_crop(src_path: Path, crop_path: Path, size: int = 2000) -> None:
    """Vyřízne čtverec size×size px centrovaný na hustotu detekovaných vozidel."""
    import rasterio
    from rasterio.windows import Window

    crop_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        # Střed na centroidu existujících detekcí, jinak střed snímku
        if VEHICLES_GPKG.exists():
            veh = gpd.read_file(VEHICLES_GPKG)
            cx = veh.geometry.centroid.x.mean()
            cy = veh.geometry.centroid.y.mean()
            center_row, center_col = src.index(cx, cy)
        else:
            center_col, center_row = src.width // 2, src.height // 2

        col_off = max(0, min(center_col - size // 2, src.width - size))
        row_off = max(0, min(center_row - size // 2, src.height - size))
        window = Window(col_off, row_off, min(size, src.width), min(size, src.height))
        transform = src.window_transform(window)
        data = src.read(window=window)
        profile = src.profile.copy()
        profile.update(width=window.width, height=window.height, transform=transform)

    with rasterio.open(crop_path, "w", **profile) as dst:
        dst.write(data)
    print(f"  Test crop: {crop_path} ({window.width}×{window.height} px, "
          f"offset col={col_off} row={row_off})")


def _detect_once(
    input_path: Path,
    conf: float,
    slice_size: int,
    overlap: float,
    iou_match: float,
) -> gpd.GeoDataFrame:
    import rasterio
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from shapely.geometry import box as sbox

    with rasterio.open(input_path) as src:
        crs = src.crs
        transform = src.transform

    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=MODEL,
        confidence_threshold=conf,
        device="cpu",
    )
    result = get_sliced_prediction(
        image=str(input_path),
        detection_model=model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        postprocess_match_threshold=iou_match,
        verbose=0,
    )
    records = []
    for pred in result.object_prediction_list:
        if pred.category.name not in VEHICLE_CLASSES:
            continue
        b = pred.bbox
        wx1, wy1 = transform * (b.minx, b.miny)
        wx2, wy2 = transform * (b.maxx, b.maxy)
        records.append({
            "class": pred.category.name,
            "confidence": round(pred.score.value, 3),
            "geometry": sbox(wx1, wy1, wx2, wy2),
        })
    return gpd.GeoDataFrame(records, crs=crs) if records else gpd.GeoDataFrame(
        columns=["class", "confidence", "geometry"])


def run_sweep(input_path: Path) -> pd.DataFrame:
    print("=== Parametrický sweep ===")
    if not TEST_CROP.exists():
        print("Vytvářím testovací výřez...")
        _make_test_crop(input_path, TEST_CROP, size=2000)

    configs = list(itertools.product(SWEEP_CONF, SWEEP_OVERLAP, SWEEP_IOU_MATCH))
    print(f"Celkem konfigurací: {len(configs)}  (slice={SWEEP_SLICE}px)\n")

    rows = []
    for i, (conf, overlap, iou_match) in enumerate(configs, 1):
        tag = f"conf={conf} ovlp={overlap} iou={iou_match}"
        print(f"[{i:2d}/{len(configs)}] {tag}", end=" ... ", flush=True)
        gdf = _detect_once(TEST_CROP, conf, SWEEP_SLICE, overlap, iou_match)
        if len(gdf) == 0:
            print("0 detekcí, přeskočeno")
            continue
        m = metrics_from_gdf(gdf)
        m.update({"conf": conf, "overlap": overlap, "iou_match": iou_match})
        rows.append(m)
        print(f"n={m['n_total']:4d}  dup={m['dup_rate']:.1%}  "
              f"size_ratio={m['size_ratio']:.2f}×  mw_p={m['mw_p']:.3f}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("Žádné výsledky.")
        return df

    df = df.sort_values("dup_rate")
    _save_sweep_results(df)
    return df


def _save_sweep_results(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "sweep_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nVýsledky uloženy: {csv_path}")

    print("\n=== TOP 5 konfigurací (nejnižší dup_rate) ===")
    cols = ["conf", "overlap", "iou_match", "n_total", "dup_rate", "size_ratio", "mw_p"]
    print(df[cols].head(5).to_string(index=False))

    # Doporučená konfigurace: nejnižší duplicity + dobrá separace (size_ratio > 2)
    good = df[df["size_ratio"] >= 2.0] if (df["size_ratio"] >= 2.0).any() else df
    best = good.sort_values("dup_rate").iloc[0]
    print(f"\n★ DOPORUČENÁ KONFIGURACE:")
    print(f"  conf={best['conf']}  overlap={best['overlap']}  "
          f"postprocess_match_threshold={best['iou_match']}")
    print(f"  dup_rate={best['dup_rate']:.1%}  "
          f"size_ratio={best['size_ratio']:.2f}×  n={int(best['n_total'])}")

    # Heatmap: overlap × iou_match, barevně dup_rate (průměr přes conf)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Sweep výsledky: míra duplicit (dup_rate)", fontsize=13, fontweight="bold")

    for ax, conf_val in zip(axes, SWEEP_CONF):
        sub = df[df["conf"] == conf_val]
        if sub.empty:
            ax.set_visible(False)
            continue
        pivot = sub.pivot_table(index="overlap", columns="iou_match",
                                values="dup_rate", aggfunc="mean")
        im = ax.imshow(pivot.values, cmap="RdYlGn_r", vmin=0, vmax=df["dup_rate"].max(),
                       aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v}" for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v}" for v in pivot.index])
        ax.set_xlabel("iou_match threshold")
        ax.set_ylabel("overlap ratio")
        ax.set_title(f"conf = {conf_val}")
        for r in range(len(pivot.index)):
            for c in range(len(pivot.columns)):
                val = pivot.values[r, c]
                if not np.isnan(val):
                    ax.text(c, r, f"{val:.0%}", ha="center", va="center",
                            fontsize=8, color="black")
        plt.colorbar(im, ax=ax, shrink=0.8, label="dup_rate")

    fig.tight_layout()
    heatmap_path = OUT_DIR / "sweep_heatmap.png"
    fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {heatmap_path}")

    # Scatter: dup_rate vs size_ratio (trade-off)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sc = ax2.scatter(df["dup_rate"], df["size_ratio"],
                     c=df["conf"], cmap="viridis", s=60,
                     edgecolors="black", linewidths=0.4)
    plt.colorbar(sc, ax=ax2, label="conf threshold")
    # Zvýrazni best config
    ax2.scatter(best["dup_rate"], best["size_ratio"],
                s=200, color="red", zorder=5, marker="*", label="★ doporučeno")
    ax2.set_xlabel("Míra duplicit (dup_rate) — nižší = lepší")
    ax2.set_ylabel("Separace tříd (size_ratio large/small) — vyšší = lepší")
    ax2.set_title("Trade-off: duplicity vs. separace tříd\n"
                  "(ideální = levý horní roh)")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    tradeoff_path = OUT_DIR / "sweep_tradeoff.png"
    fig2.savefig(tradeoff_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  → {tradeoff_path}")


# ---------------------------------------------------------------------------
# Hlavní vstupní bod
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluace kvality detekce vozidel")
    sub = parser.add_subparsers(dest="mode")

    p_analyze = sub.add_parser("analyze", help="Analyzuj existující vehicles.gpkg")
    p_analyze.add_argument("--vehicles", type=Path, default=VEHICLES_GPKG)

    p_sweep = sub.add_parser("sweep", help="Parametrický sweep přes conf/overlap/iou_match")
    p_sweep.add_argument("--input", type=Path, default=TEST_INPUT)

    args = parser.parse_args()

    if args.mode == "sweep":
        run_sweep(args.input)
    else:
        # výchozí: analýza existujícího souboru
        vehicles = getattr(args, "vehicles", VEHICLES_GPKG)
        analyze_existing(vehicles)
