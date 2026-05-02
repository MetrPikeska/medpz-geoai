# Vehicle Detection from Orthophoto – Olomouc 2016

Detekce vozidel ze školních ortofot Olomouce pomocí YOLOv8-OBB (Oriented Bounding Box) trénovaného na DOTA datasetu. SAHI tiling pro detekci na vysokorozlišovacích snímcích.

## Tech stack

- **Python 3.12** + Conda
- **YOLOv8-OBB** (ultralytics) – detekce aut z leteckých snímků
- **SAHI** – tiled inference (dlaždičky pro velké snímky)
- **GeoPandas** – vektorové zpracování
- **Rasterio** – rastrové I/O
- **OpenCV** – visualizace

## Instalace

### Conda prostředí

```bash
conda activate geoai
# nebo první spuštění:
source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoai
```

## Skript

### `src/detect.py` – Detekce vozidel s ladihodnými parametry

Detekce na plném ortofotu s volnými parametry:

```bash
python src/detect.py --input data/raw/105000312_0000_00_0000_P00_01.tif \
                     --conf 0.3 \
                     --slice-size 512 \
                     --overlap 0.2
```

**Parametry:**
- `--input` – vstupní TIF soubor (default: `data/raw/...`)
- `--conf` – confidence threshold (default: 0.25) – vyšší = méně false positive
- `--slice-size` – velikost dlaždic v px (default: 640) – menší = lépe malá auta
- `--overlap` – překryv mezi dlaždicemi (default: 0.2)

**Výstup:**
- `data/vectors/vehicles.gpkg` – detekovaná vozidla jako polygony
- `outputs/detection_preview.jpg` – náhled s vykreslenými boxy

### `src/tune.py` – Batch tuning parametrů

Vyzkoušej více kombinací parametrů najednou:

```bash
# Na croped snímku (1000×1000 px, rychle)
python src/tune.py

# Na plném snímku
python src/tune.py --input data/raw/105000312_0000_00_0000_P00_01.tif
```

Vytvoří 27 kombinací (3 modely × 3 conf × 3 slice_size × 1 overlap):
- Modely: `yolov8n-obb.pt` (nano), `yolov8s-obb.pt` (small), `yolov8m-obb.pt` (medium)
- Confidence: 0.05, 0.1, 0.2
- Slice size: 128, 256, 512 px
- Overlap: 0.3

**Výstup:**
- `outputs/tune/preview_*.jpg` – previews pro každou kombinaci
- Tabulka v terminálu s počty detekcí a průměrnou velikostí boxů

## Struktura projektu

```
data/
  raw/        # Původní ortofota (.tif)
  processed/  # Oříznuté či upravené TIF soubory
  vectors/    # Vektorové výstupy (GeoPackage)
src/
  detect.py   # Detekce vozidel YOLOv8-OBB + SAHI
  tune.py     # Batch parameter tuning
  osm.py      # [TODO] Stažení OSM dat
  analysis.py # [TODO] Spatial join, statistiky
notebooks/    # Jupyter notebooky pro exploraci
outputs/
  detection_preview.jpg  # Náhled detekce
  tune/                  # Previews z tuning batch
```

## Poznámky

- Ortofoto nemá georeferencování – souřadnice jsou pixelové (0–10328 × 0–7760)
- YOLO detekuje třídy `"small vehicle"` (zelené boxy) a `"large vehicle"` (oranžové boxy)
- SAHI tiling je nezbytné – bez něj by se auta v 10MP snímku zmenšila pod rozpoznávatelnou velikost
