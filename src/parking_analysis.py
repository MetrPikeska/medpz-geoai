"""Rozšířená analýza parkování: velikostní klasifikace, vzdálenost od centra,
ulice, DBSCAN clustering parkovišť, senioři vs. velká vozidla."""

import argparse
import re
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from shapely.geometry import MultiPoint
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore")

VORONOI_GPKG = Path("data/vectors/voronoi.gpkg")
VEHICLES_GPKG = Path("data/vectors/vehicles.gpkg")
OUT_DIR = Path("outputs")
OUT_CLUSTERS = Path("outputs/parking_clusters.gpkg")

AGE_COLS = ["sum_0_14", "sum_15_29", "sum_30_44", "sum_45_64", "sum_65_"]
AGE_LABELS = ["0–14", "15–29", "30–44", "45–64", "65+"]

# Přibližný střed Olomouce v EPSG:5514
CENTER_X, CENTER_Y = -548_500, -1_120_500


# ---------------------------------------------------------------------------
# Pomocné funkce
# ---------------------------------------------------------------------------

def extract_street(adresa: str) -> str:
    """'Stiborova 598/29, Neředín...' → 'Stiborova'"""
    street_part = adresa.split(",")[0]
    return re.sub(r"\s+\d+.*$", "", street_part).strip()


def pearson_line(x: pd.Series, y: pd.Series) -> str:
    r, p = pearsonr(x, y)
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "(n.s.)"
    return f"r = {r:.3f}, p = {p:.4f} {stars}"


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ---------------------------------------------------------------------------
# 1. Načtení dat
# ---------------------------------------------------------------------------

def load_data() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Načte Voronoi zóny (s počty aut), surová vozidla a filtrovanou podmnožinu."""
    voronoi = gpd.read_file(VORONOI_GPKG, layer="voronoi_analyza")
    vehicles = gpd.read_file(VEHICLES_GPKG)

    # Filtr pro korelační analýzy: ≥ 10 obyvatel + known density
    valid = voronoi.dropna(subset=["density_per_resident", "celkem"])
    valid = valid[valid["celkem"] >= 10].copy()

    print(f"Voronoi zón celkem: {len(voronoi)}")
    print(f"  z toho ≥ 10 obyvatel: {len(valid)}")
    print(f"Detekovaná vozidla: {len(vehicles)}")
    return voronoi, vehicles, valid


# ---------------------------------------------------------------------------
# 2. Velikostní klasifikace
# ---------------------------------------------------------------------------

def analyze_size_ratio(voronoi: gpd.GeoDataFrame, valid: gpd.GeoDataFrame) -> None:
    """Poměr velkých vozidel, korelace s věkovými skupinami."""
    print("\n[1/5] Velikostní klasifikace")

    # Poměr velkých vozidel (NaN pro adresy bez detekovaných aut)
    voronoi["large_ratio"] = np.where(
        voronoi["vehicles_total"] > 0,
        (voronoi["vehicles_large"] / voronoi["vehicles_total"]).round(3),
        np.nan,
    )

    # Pracovní podmnožina: ≥ 10 obyvatel + alespoň 1 vozidlo
    df = valid[valid["vehicles_total"] > 0].copy()
    df["large_ratio"] = np.where(
        df["vehicles_total"] > 0,
        df["vehicles_large"] / df["vehicles_total"],
        np.nan,
    )
    df = df.dropna(subset=["large_ratio"])

    print(f"  Průměrný podíl velkých vozidel: {df['large_ratio'].mean():.3f}")
    print(f"  Medián: {df['large_ratio'].median():.3f}")

    # Korelace podílu velkých vozidel vs. věkové skupiny
    print(f"\n  Korelace large_ratio × věk. skupiny (n = {len(df)})")
    corr_rows = []
    for col, label in zip(AGE_COLS, AGE_LABELS):
        share = df[col] / df["celkem"]
        r, p = pearsonr(share, df["large_ratio"])
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {label:<12} r={r:+.3f}  p={p:.4f} {stars}")
        corr_rows.append({"vekova_skupina": label, "r": round(r, 3), "p": round(p, 4)})

    # Graf – bar chart korelací
    fig, ax = plt.subplots(figsize=(8, 5))
    rs = [row["r"] for row in corr_rows]
    colors = ["#d73027" if r < 0 else "#4575b4" for r in rs]
    bars = ax.bar(AGE_LABELS, rs, color=colors, edgecolor="black", linewidth=0.5)
    for bar, row in zip(bars, corr_rows):
        offset = 0.012 if bar.get_height() >= 0 else -0.028
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"r={row['r']:+.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Věková skupina")
    ax.set_ylabel("Pearsonův r")
    ax.set_title("Korelace: podíl velkých vozidel × věková skupina\n"
                 f"(filtr: celkem ≥ 10 obyvatel, vehicles_total ≥ 1, n = {len(df)})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, OUT_DIR / "size_ratio_correlations.png")


# ---------------------------------------------------------------------------
# 3. Vzdálenost od centra
# ---------------------------------------------------------------------------

def analyze_distance(valid: gpd.GeoDataFrame) -> None:
    """Vzdálenost adresního bodu od středu Olomouce vs. hustota vozidel."""
    print("\n[2/5] Vzdálenost od centra")

    # Centroidy Voronoi zón jako proxy pozice adresy
    df = valid.copy()
    cx = df.geometry.centroid.x
    cy = df.geometry.centroid.y
    df["dist_m"] = np.sqrt((cx - CENTER_X) ** 2 + (cy - CENTER_Y) ** 2)

    r, p = pearsonr(df["dist_m"], df["density_per_resident"])
    print(f"  Rozsah vzdáleností: {df['dist_m'].min():.0f}–{df['dist_m'].max():.0f} m")
    print(f"  Korelace: {pearson_line(df['dist_m'], df['density_per_resident'])}")

    m, b = np.polyfit(df["dist_m"], df["density_per_resident"], 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["dist_m"], df["density_per_resident"],
               alpha=0.6, edgecolors="black", linewidths=0.3, s=45, color="#4575b4")
    xs = np.linspace(df["dist_m"].min(), df["dist_m"].max(), 200)
    ax.plot(xs, m * xs + b, color="#d73027", linewidth=1.5,
            label=f"regrese (slope = {m:.5f})")
    ax.set_xlabel("Vzdálenost od středu Olomouce [m]", fontsize=11)
    ax.set_ylabel("Vozidla na obyvatele", fontsize=11)
    ax.set_title(f"Vzdálenost od centra vs. hustota vozidel\n"
                 f"({pearson_line(df['dist_m'], df['density_per_resident'])}, n = {len(df)})",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, OUT_DIR / "distance_vs_density.png")


# ---------------------------------------------------------------------------
# 4. Analýza ulic
# ---------------------------------------------------------------------------

def analyze_streets(voronoi: gpd.GeoDataFrame) -> None:
    """Seskupení adres podle ulice, top 10 ulic s nejvyšší hustotou."""
    print("\n[3/5] Analýza ulic")

    df = voronoi.dropna(subset=["Adresa", "celkem"]).copy()
    df = df[df["celkem"] >= 10]
    df["ulice"] = df["Adresa"].apply(extract_street)

    grouped = df.groupby("ulice").agg(
        pocet_adres=("Adresa", "count"),
        celkem_obyvatel=("celkem", "sum"),
        celkem_vozidel=("vehicles_total", "sum"),
        prumer_density=("density_per_resident", "mean"),
    ).reset_index()

    # Filtr: min. 3 adresy v ulici
    grouped = grouped[grouped["pocet_adres"] >= 3].sort_values("prumer_density", ascending=False)
    grouped["prumer_density"] = grouped["prumer_density"].round(4)

    print(f"  Ulic s ≥ 3 adresami (celkem ≥ 10 obyv.): {len(grouped)}")
    print(f"\n  Top 10 ulic (průměr aut/os):")
    print(grouped.head(10).to_string(index=False))

    top10 = grouped.head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top10["ulice"][::-1], top10["prumer_density"][::-1],
                   color="#4575b4", edgecolor="black", linewidth=0.5)
    for bar, (_, row) in zip(bars, top10[::-1].iterrows()):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{row['prumer_density']:.3f}  (n={int(row['pocet_adres'])})",
                va="center", fontsize=8.5)
    ax.set_xlabel("Průměr vozidel na obyvatele")
    ax.set_title("Top 10 ulic – průměrná hustota vozidel na obyvatele\n"
                 "(filtr: ≥ 3 adresy v ulici, celkem ≥ 10 obyvatel)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, top10["prumer_density"].max() * 1.25)
    fig.tight_layout()
    save_fig(fig, OUT_DIR / "street_density_top10.png")

    # Export CSV
    grouped.to_csv(OUT_DIR / "street_stats.csv", index=False)
    print(f"  → {OUT_DIR / 'street_stats.csv'}")


# ---------------------------------------------------------------------------
# 5. DBSCAN clustering parkovišť
# ---------------------------------------------------------------------------

def analyze_parking_clusters(vehicles: gpd.GeoDataFrame) -> None:
    """DBSCAN na centroidech vozidel → polygony parkovišť."""
    print("\n[4/5] DBSCAN clustering parkovišť")

    centroids = vehicles.copy()
    centroids["geometry"] = vehicles.geometry.centroid
    coords = np.column_stack([centroids.geometry.x, centroids.geometry.y])

    # eps=25 m, min 5 vozidel = parkoviště
    labels = DBSCAN(eps=25, min_samples=5).fit_predict(coords)
    centroids["cluster"] = labels

    n_clusters = (labels >= 0).sum()
    n_noise = (labels == -1).sum()
    print(f"  Clusterů (parkovišť): {labels.max() + 1}")
    print(f"  Vozidel v clusterech: {n_clusters} / {len(labels)}")
    print(f"  Noise (izolovaná vozidla): {n_noise}")

    # Convex hull každého clusteru
    records = []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        pts = centroids[centroids["cluster"] == cid]
        hull = MultiPoint(list(pts.geometry)).convex_hull
        records.append({
            "cluster_id": cid,
            "vehicle_count": len(pts),
            "area_m2": round(hull.area, 1),
            "geometry": hull,
        })

    clusters_gdf = gpd.GeoDataFrame(records, crs=vehicles.crs)
    clusters_gdf = clusters_gdf.sort_values("vehicle_count", ascending=False)

    print(f"\n  Průměr vozidel na cluster: {clusters_gdf['vehicle_count'].mean():.1f}")
    print(f"  Průměrná plocha clusteru: {clusters_gdf['area_m2'].mean():.0f} m²")
    biggest = clusters_gdf.iloc[0]
    print(f"  Největší parkoviště: cluster #{biggest['cluster_id']} "
          f"({biggest['vehicle_count']} vozidel, {biggest['area_m2']:.0f} m²)")

    OUT_CLUSTERS.parent.mkdir(parents=True, exist_ok=True)
    clusters_gdf.to_file(OUT_CLUSTERS, driver="GPKG")
    print(f"  → {OUT_CLUSTERS}")


# ---------------------------------------------------------------------------
# 6. Podíl seniorů vs. podíl velkých vozidel
# ---------------------------------------------------------------------------

def analyze_seniors_vs_large(valid: gpd.GeoDataFrame) -> None:
    """Scatter: podíl seniorů vs. podíl velkých vozidel (filtr vehicles_total ≥ 3)."""
    print("\n[5/5] Senioři vs. podíl velkých vozidel")

    df = valid[valid["vehicles_total"] >= 3].copy()
    df["large_ratio"] = df["vehicles_large"] / df["vehicles_total"]
    df["senior_share"] = df["sum_65_"] / df["celkem"]
    df = df.dropna(subset=["large_ratio", "senior_share"])

    print(f"  Zón pro scatter: {len(df)} (filtr: vehicles_total ≥ 3, celkem ≥ 10)")

    if len(df) < 5:
        print("  Příliš málo dat, scatter vynechán.")
        return

    r, p = pearsonr(df["senior_share"], df["large_ratio"])
    m, b = np.polyfit(df["senior_share"], df["large_ratio"], 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["senior_share"], df["large_ratio"],
               alpha=0.65, edgecolors="black", linewidths=0.3, s=50, color="#4575b4")
    xs = np.linspace(df["senior_share"].min(), df["senior_share"].max(), 200)
    ax.plot(xs, m * xs + b, color="#d73027", linewidth=1.5,
            label=f"regrese (slope = {m:.3f})")
    ax.set_xlabel("Podíl seniorů (65+) z obyvatel adresy", fontsize=11)
    ax.set_ylabel("Podíl velkých vozidel (large_ratio)", fontsize=11)
    ax.set_title(f"Podíl seniorů vs. podíl velkých vozidel\n"
                 f"(r = {r:.3f}, p = {p:.4f}, n = {len(df)})",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, OUT_DIR / "seniors_vs_large_ratio.png")


# ---------------------------------------------------------------------------
# Hlavní vstupní bod
# ---------------------------------------------------------------------------

def run_parking_analysis(
    voronoi_gpkg: Path = VORONOI_GPKG,
    vehicles_gpkg: Path = VEHICLES_GPKG,
) -> None:
    global VORONOI_GPKG, VEHICLES_GPKG
    VORONOI_GPKG = voronoi_gpkg
    VEHICLES_GPKG = vehicles_gpkg

    print("=" * 56)
    print("  Rozšířená analýza parkování")
    print("=" * 56)

    voronoi, vehicles, valid = load_data()

    analyze_size_ratio(voronoi, valid)
    analyze_distance(valid)
    analyze_streets(voronoi)
    analyze_parking_clusters(vehicles)
    analyze_seniors_vs_large(valid)

    print("\nHotovo. Výstupy v outputs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rozšířená analýza parkování")
    parser.add_argument("--voronoi", type=Path, default=VORONOI_GPKG)
    parser.add_argument("--vehicles", type=Path, default=VEHICLES_GPKG)
    args = parser.parse_args()
    run_parking_analysis(args.voronoi, args.vehicles)
