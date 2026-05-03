"""Cross-TIF srovnání: korelace věk × hustota vozidel pro všechna 4 ortofota.

Výstupy:
  outputs/compare_correlation_heatmap.png  – r-hodnoty pro všechna ortofota × věk. skupiny
  outputs/compare_best_agegroup.png        – scatter strongest age group + regrese (per TIF)
  outputs/compare_vehicle_distribution.png – distribuce hustoty vozidel (boxploty)
  outputs/compare_summary.png              – souhrnný bar chart průměrných |r| per věk. skupinu
  docs/                                    – aktualizované PNG pro README
"""

import shutil
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

AGE_COLS   = ["sum_0_14", "sum_15_29", "sum_30_44", "sum_45_64", "sum_65_"]
AGE_LABELS = ["0–14", "15–29", "30–44", "45–64", "65+"]
TIF_LABELS = {
    1: "Neředín / Nová Ulice",
    2: "Nová Ulice (východ)",
    3: "Hodolany",
    4: "Bělidla",
}
OUT = Path("outputs")
DOCS = Path("docs")
DOCS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Načtení dat
# ---------------------------------------------------------------------------

def load_valid(idx: int) -> pd.DataFrame:
    gdf = gpd.read_file(f"data/vectors/voronoi_{idx}.gpkg", layer="voronoi_analyza")
    df = gdf.drop(columns="geometry").copy()
    df = df.dropna(subset=["density_per_resident", "celkem"])
    df = df[df["celkem"] >= 10]
    return df


def compute_correlations(df: pd.DataFrame) -> list[dict]:
    rows = []
    for col, label in zip(AGE_COLS, AGE_LABELS):
        if col not in df.columns:
            continue
        share = df[col] / df["celkem"]
        r, p = pearsonr(share, df["density_per_resident"])
        rows.append({"age": label, "col": col, "r": r, "p": p,
                     "sig": "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""})
    return rows


# ---------------------------------------------------------------------------
# 1. Heatmapa korelací (ortofoto × věk. skupina)
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(all_corr: dict[int, list[dict]]) -> None:
    r_mat = np.zeros((len(all_corr), len(AGE_LABELS)))
    p_mat = np.zeros_like(r_mat)
    sig_mat = [[""] * len(AGE_LABELS) for _ in range(len(all_corr))]

    for row_i, idx in enumerate(sorted(all_corr)):
        for col_j, row in enumerate(all_corr[idx]):
            r_mat[row_i, col_j] = row["r"]
            p_mat[row_i, col_j] = row["p"]
            sig_mat[row_i][col_j] = row["sig"]

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(r_mat, cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearsonův r", shrink=0.85)

    ax.set_xticks(range(len(AGE_LABELS)))
    ax.set_xticklabels(AGE_LABELS, fontsize=11)
    ax.set_yticks(range(len(all_corr)))
    ax.set_yticklabels([f"TIF {i}: {TIF_LABELS[i]}" for i in sorted(all_corr)], fontsize=10)
    ax.set_title("Korelace density_per_resident × věková skupina\n(všechna ortofota, filtr: celkem ≥ 10 obyvatel)",
                 fontsize=12, fontweight="bold")

    for row_i in range(len(all_corr)):
        for col_j in range(len(AGE_LABELS)):
            r_val = r_mat[row_i, col_j]
            sig = sig_mat[row_i][col_j]
            text_color = "white" if abs(r_val) > 0.22 else "black"
            ax.text(col_j, row_i, f"{r_val:+.3f}{sig}",
                    ha="center", va="center", fontsize=9.5, color=text_color, fontweight="bold")

    fig.tight_layout()
    path = OUT / "compare_correlation_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ---------------------------------------------------------------------------
# 2. Identifikace nejsilnější věkové skupiny + scatter per TIF
# ---------------------------------------------------------------------------

def find_best_agegroup(all_corr: dict[int, list[dict]]) -> tuple[str, str]:
    """Vrátí (label, col) věk. skupiny s nejvyšším průměrným |r| přes všechna TIF."""
    totals: dict[str, list[float]] = {label: [] for label in AGE_LABELS}
    for rows in all_corr.values():
        for row in rows:
            totals[row["age"]].append(abs(row["r"]))
    best_label = max(totals, key=lambda k: np.mean(totals[k]))
    best_col = AGE_COLS[AGE_LABELS.index(best_label)]
    mean_r = {k: np.mean(v) for k, v in totals.items()}
    print("\n  Průměrný |r| per věk. skupina:")
    for label in AGE_LABELS:
        marker = " ← NEJSILNĚJŠÍ" if label == best_label else ""
        print(f"    {label:<8} {mean_r[label]:.3f}{marker}")
    return best_label, best_col


def plot_best_agegroup_scatter(all_data: dict[int, pd.DataFrame],
                               best_label: str, best_col: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"Věková skupina {best_label} vs. hustota vozidel na obyvatele\n"
                 f"(nejvyšší průměrný |r| ze všech věk. skupin)",
                 fontsize=13, fontweight="bold")

    for ax, (idx, df) in zip(axes.flat, sorted(all_data.items())):
        share = df[best_col] / df["celkem"]
        dens  = df["density_per_resident"]
        r, p  = pearsonr(share, dens)
        sig   = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "(n.s.)"
        color = "#d73027" if r < 0 else "#4575b4"

        ax.scatter(share, dens, alpha=0.55, edgecolors="black", linewidths=0.3,
                   s=40, color=color)
        m, b = np.polyfit(share, dens, 1)
        xs = np.linspace(share.min(), share.max(), 200)
        ax.plot(xs, m * xs + b, color="black", linewidth=1.4,
                label=f"r = {r:+.3f} {sig}\np = {p:.4f}, n = {len(df)}")
        ax.set_xlabel(f"Podíl {best_label} let z obyvatel", fontsize=9)
        ax.set_ylabel("Vozidla / obyvatel", fontsize=9)
        ax.set_title(f"TIF {idx}: {TIF_LABELS[idx]}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    path = OUT / "compare_best_agegroup.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ---------------------------------------------------------------------------
# 3. Distribuce hustoty vozidel (boxploty)
# ---------------------------------------------------------------------------

def plot_vehicle_distribution(all_data: dict[int, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Distribuce hustoty vozidel a počtu vozidel – srovnání ortofot",
                 fontsize=12, fontweight="bold")

    labels = [f"TIF {i}\n{TIF_LABELS[i]}" for i in sorted(all_data)]

    # Boxplot density_per_resident
    ax = axes[0]
    data_density = [all_data[i]["density_per_resident"].dropna().values
                    for i in sorted(all_data)]
    bp = ax.boxplot(data_density, labels=labels, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 1.5},
                    flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
    colors = ["#4575b4", "#74add1", "#fdae61", "#d73027"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel("Vozidla na obyvatele")
    ax.set_title("Hustota vozidel na obyvatele")
    ax.grid(axis="y", alpha=0.3)

    # Boxplot vehicles_total
    ax = axes[1]
    data_total = [all_data[i]["vehicles_total"].values for i in sorted(all_data)]
    bp2 = ax.boxplot(data_total, labels=labels, patch_artist=True,
                     medianprops={"color": "black", "linewidth": 1.5},
                     flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
    for patch, c in zip(bp2["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel("Počet vozidel v zóně")
    ax.set_title("Absolutní počet vozidel per Voronoi zóna")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = OUT / "compare_vehicle_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ---------------------------------------------------------------------------
# 4. Souhrnný bar chart průměrného |r| per věk. skupina
# ---------------------------------------------------------------------------

def plot_summary_barchart(all_corr: dict[int, list[dict]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Souhrn korelací věková skupina × hustota vozidel",
                 fontsize=12, fontweight="bold")

    # Levý panel: grouped bar chart – r per TIF per věk. skupina
    ax = axes[0]
    x = np.arange(len(AGE_LABELS))
    width = 0.18
    bar_colors = ["#4575b4", "#74add1", "#fdae61", "#d73027"]
    for i, (idx, rows) in enumerate(sorted(all_corr.items())):
        rs = [row["r"] for row in rows]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, rs, width, label=f"TIF {idx}: {TIF_LABELS[idx]}",
                      color=bar_colors[i], edgecolor="black", linewidth=0.4, alpha=0.85)
        for bar, row in zip(bars, rows):
            if row["sig"]:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.008 if bar.get_height() >= 0 else -0.022),
                        row["sig"], ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(AGE_LABELS)
    ax.set_ylabel("Pearsonův r")
    ax.set_ylim(-0.45, 0.45)
    ax.set_title("Pearsonův r per ortofoto a věk. skupina\n(* p<0.05, ** p<0.01, *** p<0.001)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Pravý panel: průměrný |r| per věk. skupina (přes všechna TIF)
    ax = axes[1]
    mean_abs_r = []
    std_abs_r = []
    for col, label in zip(AGE_COLS, AGE_LABELS):
        vals = [abs(row["r"]) for rows in all_corr.values() for row in rows if row["age"] == label]
        mean_abs_r.append(np.mean(vals))
        std_abs_r.append(np.std(vals))

    best_idx = int(np.argmax(mean_abs_r))
    bar_colors2 = ["#d9d9d9"] * len(AGE_LABELS)
    bar_colors2[best_idx] = "#d73027"
    bars = ax.bar(AGE_LABELS, mean_abs_r, color=bar_colors2, edgecolor="black",
                  linewidth=0.5, yerr=std_abs_r, capsize=4)
    for bar, val in zip(bars, mean_abs_r):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Průměrný |r| (přes 4 ortofota)")
    ax.set_ylim(0, max(mean_abs_r) * 1.4)
    ax.set_title("Průměrná síla korelace per věk. skupina\n(červená = nejsilnější)")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = OUT / "compare_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ---------------------------------------------------------------------------
# 5. Aktualizace docs/
# ---------------------------------------------------------------------------

def update_docs() -> None:
    copies = {
        # Klíčové mapy z imagery_1 (referenční snímek)
        "outputs/analysis_map_1.png":          "docs/analysis_map.png",
        "outputs/street_density_top10_1.png":  "docs/street_density_top10.png",
        # Cross-TIF srovnání
        "outputs/compare_correlation_heatmap.png": "docs/compare_correlation_heatmap.png",
        "outputs/compare_best_agegroup.png":        "docs/compare_best_agegroup.png",
        "outputs/compare_summary.png":              "docs/compare_summary.png",
        "outputs/compare_vehicle_distribution.png": "docs/compare_vehicle_distribution.png",
        # Jednotlivé analýzy ortofot
        "outputs/analysis_map_2.png": "docs/analysis_map_2.png",
        "outputs/analysis_map_3.png": "docs/analysis_map_3.png",
        "outputs/analysis_map_4.png": "docs/analysis_map_4.png",
        # Eval (nemění se)
        "outputs/eval_duplicate_map.png": "docs/eval_duplicate_map.png",
        "outputs/sweep_heatmap.png":      "docs/sweep_heatmap.png",
    }
    print("\nAktualizuji docs/:")
    for src, dst in copies.items():
        if Path(src).exists():
            shutil.copy2(src, dst)
            print(f"  {src} → {dst}")
        else:
            print(f"  [chybí] {src}")


# ---------------------------------------------------------------------------
# Hlavní
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Cross-TIF analýza korelací ===\n")

    all_data: dict[int, pd.DataFrame] = {}
    all_corr: dict[int, list[dict]] = {}

    for idx in [1, 2, 3, 4]:
        df = load_valid(idx)
        all_data[idx] = df
        corr = compute_correlations(df)
        all_corr[idx] = corr
        print(f"TIF {idx} ({TIF_LABELS[idx]}): n={len(df)}")
        for row in corr:
            print(f"  {row['age']:<8} r={row['r']:+.3f}  p={row['p']:.4f} {row['sig']}")

    print("\n[1/4] Heatmapa korelací")
    plot_correlation_heatmap(all_corr)

    print("\n[2/4] Nejsilnější věková skupina")
    best_label, best_col = find_best_agegroup(all_corr)
    print(f"\n  → Nejsilnější: {best_label} (průměrný |r| přes 4 TIF)")
    plot_best_agegroup_scatter(all_data, best_label, best_col)

    print("\n[3/4] Distribuce hustoty vozidel")
    plot_vehicle_distribution(all_data)

    print("\n[4/4] Souhrnný bar chart")
    plot_summary_barchart(all_corr)

    update_docs()
    print("\nHotovo.")


if __name__ == "__main__":
    main()
