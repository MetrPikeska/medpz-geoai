"""Spatial analysis: vehicle density per address Voronoi zone + demographic correlation."""

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from shapely.geometry import MultiPoint, box
from shapely.ops import voronoi_diagram

warnings.filterwarnings("ignore")

VEHICLES = Path("data/vectors/vehicles.gpkg")
ADDRESSES = Path("data/adr_ol.geojson")
OUTPUT_VORONOI = Path("data/vectors/voronoi.gpkg")
OUTPUT_MAP = Path("outputs/analysis_map.png")
OUTPUT_STATS = Path("outputs/statistics.csv")

AGE_COLS = ["sum_0_14", "sum_15_29", "sum_30_44", "sum_45_64", "sum_65_"]
AGE_LABELS = ["0–14", "15–29", "30–44", "45–64", "65+"]


def build_voronoi(addresses: gpd.GeoDataFrame, clip_geom) -> gpd.GeoDataFrame:
    pts = MultiPoint(list(addresses.geometry))
    regions = voronoi_diagram(pts, envelope=clip_geom)
    polys = list(regions.geoms)
    voronoi_gdf = gpd.GeoDataFrame(geometry=polys, crs=addresses.crs)

    # Associate each Voronoi polygon with the address point it contains
    addr_reset = addresses.reset_index(drop=True).reset_index().rename(columns={"index": "addr_idx"})
    joined = gpd.sjoin(addr_reset, voronoi_gdf.reset_index().rename(columns={"index": "vor_idx"}),
                       how="left", predicate="within")

    attr_cols = ["addr_idx", "vor_idx", "Adresa", "celkem"] + AGE_COLS
    voronoi_gdf = voronoi_gdf.reset_index().rename(columns={"index": "vor_idx"})
    voronoi_gdf = voronoi_gdf.merge(
        joined[[c for c in attr_cols if c in joined.columns]],
        on="vor_idx",
        how="left",
    )

    # Clip to study area
    voronoi_gdf["geometry"] = voronoi_gdf.geometry.intersection(clip_geom)
    return voronoi_gdf.drop(columns=["addr_idx"], errors="ignore")


def run_analysis(
    vehicles_path: Path,
    addresses_path: Path,
    output_voronoi: Path,
    output_map: Path,
    output_stats: Path,
) -> gpd.GeoDataFrame:
    print("Loading data...")
    vehicles = gpd.read_file(vehicles_path)
    addresses = gpd.read_file(addresses_path)

    if vehicles.crs != addresses.crs:
        vehicles = vehicles.to_crs(addresses.crs)

    n_small = (vehicles["class"] == "small vehicle").sum()
    n_large = (vehicles["class"] == "large vehicle").sum()
    print(f"  Vehicles: {len(vehicles)} ({n_small} small, {n_large} large)")
    print(f"  Addresses: {len(addresses)}")

    # Study area bounding box with 100 m buffer
    all_geoms = vehicles.union_all().union(addresses.union_all())
    study_area = box(*all_geoms.bounds).buffer(100)

    print("Creating Voronoi polygons...")
    voronoi = build_voronoi(addresses, study_area)
    print(f"  {len(voronoi)} zones")

    print("Counting vehicles per zone...")
    veh = vehicles[["class", "geometry"]].copy()
    veh_centroids = veh.copy()
    veh_centroids["geometry"] = veh.geometry.centroid

    vor_indexed = voronoi[["vor_idx", "geometry"]].set_index("vor_idx")
    veh_in_zones = gpd.sjoin(
        veh_centroids.reset_index().rename(columns={"index": "veh_idx"}),
        vor_indexed.reset_index(),
        how="left",
        predicate="within",
    )

    counts = (
        veh_in_zones.groupby("vor_idx")
        .agg(
            vehicles_total=("class", "count"),
            vehicles_small=("class", lambda x: (x == "small vehicle").sum()),
            vehicles_large=("class", lambda x: (x == "large vehicle").sum()),
        )
        .reset_index()
    )

    voronoi = voronoi.merge(counts, on="vor_idx", how="left")
    for col in ["vehicles_total", "vehicles_small", "vehicles_large"]:
        voronoi[col] = voronoi[col].fillna(0).astype(int)

    voronoi["density_per_resident"] = np.where(
        voronoi["celkem"] > 0,
        (voronoi["vehicles_total"] / voronoi["celkem"]).round(3),
        np.nan,
    )

    # --- Summary statistics ---
    print("\n=== SOUHRNNÉ STATISTIKY ===")
    print(f"Celkem detekovaných vozidel: {voronoi['vehicles_total'].sum()}")
    print(f"  - malá vozidla: {voronoi['vehicles_small'].sum()}")
    print(f"  - velká vozidla: {voronoi['vehicles_large'].sum()}")
    print(f"Průměr vozidel / zóna: {voronoi['vehicles_total'].mean():.1f}")
    print(f"Medián vozidel / zóna: {voronoi['vehicles_total'].median():.1f}")

    idx_max = voronoi["vehicles_total"].idxmax()
    print(f"Zóna s nejvíce vozidly: {voronoi.loc[idx_max, 'Adresa']} "
          f"({voronoi.loc[idx_max, 'vehicles_total']} ks)")

    valid = voronoi.dropna(subset=["density_per_resident", "celkem"])
    valid = valid[valid["celkem"] >= 10]

    print(f"\nKorelační analýza: {len(valid)} zón (filtr celkem ≥ 10 obyvatel)")
    print("\n=== KORELACE PEARSON (r) — density_per_resident × věk. skupiny ===")
    print(f"{'Věk. skupina':<12} {'r':>7} {'p':>8} {'':>4}")
    print("-" * 48)
    corr_rows = []
    for col, label in zip(AGE_COLS, AGE_LABELS):
        if col not in valid.columns:
            continue
        # Podíl věkové skupiny z celkového počtu obyvatel zóny
        share = valid[col] / valid["celkem"]
        r, p = pearsonr(share, valid["density_per_resident"])
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{label:<12} {r:>7.3f} {p:>8.4f} {stars}")
        corr_rows.append({"vekova_skupina": label, "r": round(r, 3), "p": round(p, 4)})

    # Derive suffix from output_map stem (e.g. analysis_map_1 → _1)
    stem = output_map.stem
    suffix = stem[len("analysis_map"):]

    # --- Exports ---
    output_voronoi.parent.mkdir(parents=True, exist_ok=True)
    output_map.parent.mkdir(parents=True, exist_ok=True)

    # GeoPackage with three layers for QGIS
    voronoi.to_file(output_voronoi, driver="GPKG", layer="voronoi_analyza")
    vehicles.to_file(output_voronoi, driver="GPKG", layer="vozidla")
    addresses.to_file(output_voronoi, driver="GPKG", layer="adresy")
    print(f"\nGeoPackage uložen: {output_voronoi}")
    print("  Vrstvy: voronoi_analyza, vozidla, adresy")

    # CSV statistics
    stats_cols = ["Adresa", "celkem"] + AGE_COLS + [
        "vehicles_total", "vehicles_small", "vehicles_large", "density_per_resident"
    ]
    voronoi[[c for c in stats_cols if c in voronoi.columns]].to_csv(output_stats, index=False)
    pd.DataFrame(corr_rows).to_csv(output_stats.parent / f"correlations{suffix}.csv", index=False)
    print(f"Statistiky: {output_stats}")
    print(f"Korelace: {output_stats.parent / 'correlations.csv'}")

    # --- Map ---
    print(f"\nGeneruji mapu → {output_map}")
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle("Analýza vozidel – Neředín, Olomouc", fontsize=14, fontweight="bold")

    # Left: vehicle count per Voronoi zone
    ax = axes[0]
    voronoi.plot(
        column="vehicles_total", ax=ax, cmap="YlOrRd", legend=True,
        legend_kwds={"label": "Počet vozidel", "shrink": 0.7},
        edgecolor="grey", linewidth=0.4,
    )
    vehicles.assign(geometry=vehicles.geometry.centroid).plot(
        ax=ax, markersize=1.5, color="steelblue", alpha=0.5, zorder=3
    )
    addresses.plot(ax=ax, markersize=18, color="black", zorder=5, marker="^")
    ax.set_title("Počet detekovaných vozidel\nna Voronoi zónu", fontsize=11)
    ax.set_axis_off()

    # Right: density per resident
    ax = axes[1]
    voronoi.plot(
        column="density_per_resident", ax=ax, cmap="RdYlGn_r", legend=True,
        legend_kwds={"label": "Vozidla / obyvatel", "shrink": 0.7},
        edgecolor="grey", linewidth=0.4,
        missing_kwds={"color": "lightgrey", "label": "bez dat"},
    )
    addresses.plot(ax=ax, markersize=18, color="black", zorder=5, marker="^")
    ax.set_title("Hustota vozidel na obyvatele\n(Voronoi zóny)", fontsize=11)
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(output_map, dpi=150, bbox_inches="tight")
    print(f"Mapa uložena: {output_map}")

    # Correlation bar chart
    corr_map = output_map.parent / f"correlations{suffix}.png"
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    colors_bar = ["#d73027" if r < 0 else "#4575b4" for r in [row["r"] for row in corr_rows]]
    bars = ax2.bar(AGE_LABELS[:len(corr_rows)], [row["r"] for row in corr_rows],
                   color=colors_bar, edgecolor="black", linewidth=0.5)
    for bar, row in zip(bars, corr_rows):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (0.01 if bar.get_height() >= 0 else -0.025),
                 f"r={row['r']:.3f}", ha="center", va="bottom", fontsize=9)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Věková skupina")
    ax2.set_ylabel("Pearsonův r (vozidla × počet obyvatel skupiny)")
    ax2.set_title("Korelace počtu vozidel s věkovými skupinami")
    ax2.set_ylim(-1, 1)
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(corr_map, dpi=150, bbox_inches="tight")
    print(f"Graf korelací: {corr_map}")

    # Scatter plot: podíl seniorů vs. auta na osobu
    scatter_path = output_map.parent / f"scatter_seniors{suffix}.png"
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    senior_share = valid["sum_65_"] / valid["celkem"]
    ax3.scatter(senior_share, valid["density_per_resident"],
                alpha=0.6, edgecolors="black", linewidths=0.4, s=50, color="#4575b4")
    # Regresní přímka
    m, b = np.polyfit(senior_share, valid["density_per_resident"], 1)
    xs = np.linspace(senior_share.min(), senior_share.max(), 100)
    ax3.plot(xs, m * xs + b, color="#d73027", linewidth=1.5, label=f"regrese (slope={m:.3f})")
    r_s, p_s = pearsonr(senior_share, valid["density_per_resident"])
    ax3.set_xlabel("Podíl seniorů (65+) z celkového počtu obyvatel", fontsize=11)
    ax3.set_ylabel("Vozidla na obyvatele (density_per_resident)", fontsize=11)
    ax3.set_title(f"Senioři vs. hustota vozidel\n(r = {r_s:.3f}, p = {p_s:.4f}, n = {len(valid)})", fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(scatter_path, dpi=150, bbox_inches="tight")
    print(f"Scatter plot: {scatter_path}")

    return voronoi


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spatial analysis of detected vehicles")
    parser.add_argument("--vehicles", type=Path, default=VEHICLES)
    parser.add_argument("--addresses", type=Path, default=ADDRESSES)
    parser.add_argument("--output-voronoi", type=Path, default=OUTPUT_VORONOI)
    parser.add_argument("--output-map", type=Path, default=OUTPUT_MAP)
    parser.add_argument("--output-stats", type=Path, default=OUTPUT_STATS)
    args = parser.parse_args()

    run_analysis(
        vehicles_path=args.vehicles,
        addresses_path=args.addresses,
        output_voronoi=args.output_voronoi,
        output_map=args.output_map,
        output_stats=args.output_stats,
    )
