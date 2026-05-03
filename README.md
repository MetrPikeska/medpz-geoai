# Detekce a analýza vozidel z ortofota – Olomouc 2016

**Semestrální práce** | Pokročilé metody dálkového průzkumu Země (MEDPZ)  
**Autoři:** Petr Mikeska, Daniel Lee Číp  
**Oblast:** Neředín / Nová Ulice / Hodolany / Bělidla, Olomouc | 4× ortofoto 2016, ~10 cm/px, EPSG:5514

## Co projekt dělá

1. **Detekuje vozidla** ze 4 ortofot pomocí YOLOv8s-OBB + SAHI tiling
2. **Ořezává adresní body** RÚIAN (14 458 adres celé Olomouce) na extenty každého snímku
3. **Vytváří Voronoi zóny** a analyzuje hustotu vozidel na obyvatele
4. **Porovnává korelace** věkových skupin × hustota vozidel přes všechna ortofota
5. **Hledá parkoviště** pomocí DBSCAN clusteringu
6. **Evaluuje kvalitu detekce** parametrickým sweepem bez ground truth

---

## Výsledky – přehled

| Ortofoto | Oblast | Vozidel | Malá | Velká | Adres | Zón (≥10 ob.) | Největší parkoviště |
|---|---|---|---|---|---|---|---|
| 1 | Neředín / Nová Ulice | 1 022 | 838 | 184 | 512 | 233 | 67 voz., 1 060 m² |
| 2 | Nová Ulice (východ) | 639 | 444 | 195 | 389 | 206 | 22 voz., 578 m² |
| 3 | Hodolany | 678 | 431 | 247 | 445 | 136 | 51 voz., 635 m² |
| 4 | Bělidla | 829 | 499 | 330 | 224 | 145 | **45 voz., 1 444 m²** |

| Metrika | Hodnota |
|---|---|
| Nejsilnější věk. skupina | **0–14 let** (průměrný \|r\| = 0,141 přes 4 TIF) |
| Nejvýznamnější korelace | TIF 4: 0–14 r=−0,297\*\*\*, 45–64 r=+0,293\*\*\* |
| Optimální `iou_match` | 0,3 → 0 % duplicit |
| Model | YOLOv8s-OBB (DOTA, small) |

---

## Detekce vozidel

![Náhled detekce](docs/detection_preview.jpg)

*Zelené boxy = malá vozidla, oranžové = velká vozidla. YOLOv8s-OBB + SAHI tiling (640 px dlaždice, overlap 20 %, iou_match 0,3).*

---

## Prostorová analýza – Voronoi hustota

### TIF 1 – Neředín / Nová Ulice

![Analýza TIF 1](docs/analysis_map.png)

*Vlevo: absolutní počty vozidel per Voronoi zóna. Vpravo: hustota na obyvatele. 1 022 vozidel, 512 adres, 39 parkovišť (největší: 67 voz., 1 060 m²).*

### TIF 2 – Nová Ulice (východ)

![Analýza TIF 2](docs/analysis_map_2.png)

*639 vozidel, 389 adres, 8 parkovišť (největší: 22 voz., 578 m²). Zástavba s více průmyslovými objekty — vyšší podíl velkých vozidel (30 %).*

### TIF 3 – Hodolany

![Analýza TIF 3](docs/analysis_map_3.png)

*678 vozidel, 445 adres, 18 parkovišť (největší: 51 voz., 635 m²). Sídlištní bloky — větší Voronoi zóny s nižší průměrnou hustotou.*

### TIF 4 – Bělidla

![Analýza TIF 4](docs/analysis_map_4.png)

*829 vozidel, 224 adres, 24 parkovišť (největší: 45 voz., 1 444 m²). Nejmenší počet adres → největší Voronoi zóny. Statisticky nejsilnější korelace věk × vozidla.*

---

## Korelace věková struktura × hustota vozidel

### Srovnání přes všechna 4 ortofota

![Heatmapa korelací](docs/compare_correlation_heatmap.png)

*Pearsonův r pro každou kombinaci ortofoto × věková skupina. Nejsilnější a statisticky nejvýznamnější signály jsou v TIF 4 (Bělidla).*

### Souhrnný přehled korelací

![Souhrn korelací](docs/compare_summary.png)

*Vlevo: r per ortofoto a věk. skupinu (\* p<0,05, \*\* p<0,01, \*\*\* p<0,001). Vpravo: průměrný |r| přes 4 ortofota — věková skupina **0–14 let** je nejkonzistentnější prediktor hustoty vozidel.*

### Nejsilnější věková skupina: 0–14 let

![Nejsilnější věk. skupina](docs/compare_best_agegroup.png)

*Scatter: podíl dětí (0–14 let) vs. hustota vozidel na obyvatele pro každé ortofoto. Negativní korelace: adresy s více dětmi mají méně aut na osobu.*

### Per ortofoto – korelační bar charty

| TIF 1 – Neředín | TIF 2 – Nová Ulice (v.) |
|---|---|
| ![](docs/correlations_1.png) | ![](docs/correlations_2.png) |

| TIF 3 – Hodolany | TIF 4 – Bělidla |
|---|---|
| ![](docs/correlations_3.png) | ![](docs/correlations_4.png) |

*TIF 1: 0–14 let slabě signifikantní (r=−0,134, p=0,041). TIF 4: silné a obousměrné — děti a mladí negativně, střední věk pozitivně.*

---

## Distribuce vozidel

![Distribuce](docs/compare_vehicle_distribution.png)

*Boxploty hustoty vozidel na obyvatele a absolutního počtu vozidel per Voronoi zóna. TIF 4 má výrazně vyšší medián i rozptyl — odráží větší zóny a koncentrovaná parkoviště.*

---

## Hustota vozidel podle ulic

| TIF 1 – Neředín | TIF 2 – Nová Ulice (v.) |
|---|---|
| ![](docs/street_density_top10_1.png) | ![](docs/street_density_top10_2.png) |

| TIF 3 – Hodolany | TIF 4 – Bělidla |
|---|---|
| ![](docs/street_density_top10_3.png) | ![](docs/street_density_top10_4.png) |

*Top 10 ulic per oblast — průměrná hustota vozidel na obyvatele (filtr: ≥ 3 adresy, celkem ≥ 10 obyvatel).*

---

## Evaluace kvality detekce

### Duplicitní detekce

![Duplicity](docs/eval_duplicate_map.png)

*Modrá = unikátní detekce, červená = duplicitní box (IOS > 0,3). Po nastavení `iou_match=0,3` → 0 % duplicit.*

### Parametrický sweep

![Heatmap sweepů](docs/sweep_heatmap.png)

*`postprocess_match_threshold` je dominantní parametr. Při hodnotě 0,3 jsou duplicity eliminovány pro všechny kombinace conf × overlap.*

---

## Instalace

```bash
pip install geopandas rasterio scipy matplotlib scikit-learn sahi ultralytics
# GPU (CUDA 11.8, např. Pascal GTX 1060):
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

## Rychlý start

```bash
# Celý pipeline pro všechna 4 ortofota
python src/run_all.py

# Jen vybraná ortofota (přeskočí detekci pokud vehicles_N.gpkg existuje)
python src/run_all.py --imagery 3 4 --skip-detect

# Srovnávací analýza přes všechna ortofota
python src/compare_all.py
```

## Skripty

### `src/clip_addresses.py` – Ořez adresních bodů

```bash
python src/clip_addresses.py
```

Ořízne `data/adr_ol.geojson` (14 458 adres) na extenty každého TIF → `data/adr_imagery_{1–4}.geojson`.

### `src/detect.py` – Detekce vozidel

```bash
python src/detect.py --input data/raw/Olomouc_imagery_1.tif \
                     --output-vectors data/vectors/vehicles_1.gpkg \
                     --conf 0.25 --slice-size 640 --overlap 0.2
```

| Parametr | Default | Popis |
|---|---|---|
| `--conf` | 0.25 | Confidence threshold |
| `--slice-size` | 640 | Velikost dlaždic v px |
| `--overlap` | 0.2 | Překryv dlaždic |
| `--output-vectors` | `data/vectors/vehicles.gpkg` | Výstupní GeoPackage |

> `postprocess_match_threshold=0.3` je nastaven napevno — sweep ukázal 0 % duplicit.

### `src/analyze.py` – Prostorová analýza

```bash
python src/analyze.py --vehicles data/vectors/vehicles_1.gpkg \
                      --addresses data/adr_imagery_1.geojson \
                      --output-voronoi data/vectors/voronoi_1.gpkg \
                      --output-map outputs/analysis_map_1.png \
                      --output-stats outputs/statistics_1.csv
```

Výstupy: Voronoi GeoPackage (3 vrstvy pro QGIS), mapa hustoty, korelační bar chart, scatter senioři, CSV statistiky.

### `src/parking_analysis.py` – Rozšířená analýza

```bash
python src/parking_analysis.py --vehicles data/vectors/vehicles_1.gpkg \
                                --voronoi data/vectors/voronoi_1.gpkg \
                                --suffix _1
```

| Analýza | Výstup |
|---|---|
| Podíl velkých vozidel × věk | `outputs/size_ratio_correlations_{N}.png` |
| Vzdálenost od centra vs. hustota | `outputs/distance_vs_density_{N}.png` |
| Top 10 ulic | `outputs/street_density_top10_{N}.png` |
| DBSCAN parkoviště (eps=12 m, min=8, max=5 000 m²) | `outputs/parking_clusters_{N}.gpkg` |
| Senioři vs. podíl velkých vozidel | `outputs/seniors_vs_large_ratio_{N}.png` |

> **Poznámka k DBSCAN:** Původní `eps=25 m` spojoval auta přes celé sídlištní bloky (uličky ~20 m) → falešná parkoviště přes 40 000 m². Zpřísněno na `eps=12 m, min_samples=8` + filtr max 5 000 m².

### `src/compare_all.py` – Cross-TIF srovnání

```bash
python src/compare_all.py
```

Načte voronoi_{1–4}.gpkg, spočítá korelace pro všechna ortofota, identifikuje nejsilnější věkovou skupinu, vygeneruje heatmapu, scatter a souhrnné grafy. Aktualizuje `docs/`.

### `src/evaluate.py` – Evaluace kvality detekce

```bash
python src/evaluate.py analyze   # rychlá analýza existujících detekcí
python src/evaluate.py sweep     # parametrický sweep 27 konfigurací (~20 min)
```

### `src/run_all.py` – Kompletní pipeline

```bash
python src/run_all.py [--imagery 1 2 3 4] [--skip-detect]
```

Postupně spustí: ořez adres → detekce → analýza → parking analýza pro každé vybrané ortofoto.

---

## Struktura projektu

```
data/
  raw/                        # Ortofoto TIF (gitignorováno)
  adr_ol.geojson              # Adresní body RÚIAN – celá Olomouc (14 458 bodů)
  adr_imagery_{1-4}.geojson   # Oříznuté na extenty TIF (generuje clip_addresses.py)
  vectors/                    # GeoPackage výstupy (gitignorováno)
src/
  clip_addresses.py   # Ořez RÚIAN na extenty TIF
  detect.py           # Detekce YOLOv8s-OBB + SAHI
  analyze.py          # Voronoi, hustota, Pearsonovy korelace
  parking_analysis.py # DBSCAN clustering, ulice, vzdálenost od centra
  compare_all.py      # Cross-TIF srovnání a aktualizace docs/
  evaluate.py         # Evaluace kvality bez ground truth
  tune.py             # Batch tuning parametrů modelu
  run_all.py          # Orchestrace celého pipeline
docs/                 # PNG grafy pro README (sledováno gitem)
outputs/              # Všechny generované výstupy (gitignorováno)
poznatky.md           # Klíčové výsledky a interpretace
yolov8s-obb.pt        # Model (DOTA, small)
```

---

## Klíčové poznatky

**Nejsilnější věková skupina: 0–14 let (průměrný |r| = 0,141)**  
Konzistentně negativní korelace přes všechna ortofota — adresy s vyšším podílem dětí mají méně aut na osobu. Zrcadlový efekt ve skupině 45–64 let (|r| = 0,134, pozitivní). Nejsilnější signal v Bělidlách (TIF 4), kde jsou oba efekty statisticky vysoce signifikantní (p < 0,001).

**Prostorová variabilita korelací**  
TIF 1 a 2 (Nová Ulice): korelace slabé, většinou nevýznamné. TIF 3 (Hodolany): slabý efekt ve skupině 45–64. TIF 4 (Bělidla): statisticky silné korelace v obou směrech — nejhomogennější demografická oblast ze čtyř.

**Korelační artefakt absolutních počtů**  
Analýza absolutních počtů vozidel vs. absolutních počtů obyvatel ukazuje r ≈ 0,5 pro všechny skupiny — artefakt velikosti budovy. Po normalizaci na hustotu (vozidla/obyvatel) efekt mizí nebo se obrátí.

**Klasifikace small/large vehicle**  
Nespolehlivá — model YOLOv8s-OBB trénovaný na DOTA (satelitní snímky) špatně rozlišuje třídy na 10 cm/px ortofotu. Pro analýzu používáme pouze celkový počet vozidel.

**Optimální detekční nastavení**  
`conf=0.25, overlap=0.2, postprocess_match_threshold=0.3` → 0 % duplicit.

---

## Tech stack

- **YOLOv8s-OBB** (ultralytics) + **SAHI** – tiled inference na velkých snímcích
- **GeoPandas** + **Shapely** – vektorová GIS analýza, Voronoi diagram
- **Rasterio** – rastrové I/O (GeoTIFF → PIL pro SAHI)
- **Scipy** – Pearsonova korelace, Mann-Whitney U test
- **scikit-learn** – DBSCAN clustering parkovišť
- **Matplotlib** – vizualizace
- **PyTorch 2.6 + CUDA 11.8** – GPU inference (kompatibilní s Pascal CC 6.1)
