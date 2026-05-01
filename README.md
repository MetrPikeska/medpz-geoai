# Vehicle Detection from Orthophoto – Olomouc 2016

Detekce a klasifikace vozidel ze školních ortofot Olomouce pomocí SAM2. Klasifikace zaparkovaná vs. na cestě pomocí OSM silniční sítě. Spatial join s budovami.

## Tech stack

- Python 3.12
- SAM2 (segment-geospatial)
- GeoPandas
- OSMnx
- Rasterio

## Struktura projektu

```
data/
  raw/        # Původní ortofota (.tif) – ignorováno gitem
  processed/  # Zpracovaná rastrová data
  vectors/    # Vektorové výstupy (GeoJSON, GPKG)
src/
  detect.py   # Detekce vozidel pomocí SAM2
  osm.py      # Stažení OSM silniční sítě a budov
  analysis.py # Spatial join, statistiky, klasifikace
notebooks/    # Jupyter notebooky pro exploraci
outputs/      # Výsledky, vizualizace
```

## Instalace

```bash
pip install -r requirements.txt
```
