"""Clips address points to the bounding box of each orthophoto TIF.

Outputs: data/adr_imagery_{1..4}.geojson
"""

from pathlib import Path

import geopandas as gpd
import rasterio
from shapely.geometry import box

ADDRESSES = Path("data/adr_ol.geojson")
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data")


def clip_addresses(addresses_path: Path = ADDRESSES, raw_dir: Path = RAW_DIR) -> dict[str, Path]:
    adr = gpd.read_file(addresses_path)
    print(f"Načteno {len(adr)} adresních bodů (CRS: {adr.crs})")

    tifs = sorted(raw_dir.glob("Olomouc_imagery_*.tif"))
    if not tifs:
        raise FileNotFoundError(f"Žádné TIF soubory v {raw_dir}")

    outputs: dict[str, Path] = {}

    for tif in tifs:
        idx = tif.stem.split("_")[-1]
        with rasterio.open(tif) as src:
            b = src.bounds
            crs_epsg = src.crs.to_epsg() if (src.crs and src.crs.to_epsg()) else 5514

        bbox = box(b.left, b.bottom, b.right, b.top)
        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs=f"EPSG:{crs_epsg}")

        if adr.crs != bbox_gdf.crs:
            adr_reproj = adr.to_crs(bbox_gdf.crs)
        else:
            adr_reproj = adr

        clipped = adr_reproj[adr_reproj.geometry.within(bbox)].copy()

        out_path = OUT_DIR / f"adr_imagery_{idx}.geojson"
        clipped.to_file(out_path, driver="GeoJSON")
        print(f"  {tif.name}: {len(clipped)} bodů → {out_path}")
        outputs[idx] = out_path

    return outputs


if __name__ == "__main__":
    clip_addresses()
