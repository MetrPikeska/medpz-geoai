"""Vehicle detection from orthophoto using YOLOv8-OBB + SAHI tiling."""

from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

INPUT = Path("data/raw/105000312_0000_00_0000_P00_01.tif")
OUTPUT_VECTORS = Path("data/vectors/vehicles.gpkg")
OUTPUT_PREVIEW = Path("outputs/detection_preview.jpg")

VEHICLE_CLASSES = {"small vehicle", "large vehicle"}
MODEL = "yolov8n-obb.pt"

# Tile size and overlap – tune for vehicle scale in this orthophoto
SLICE_SIZE = 640
OVERLAP = 0.2
CONF = 0.5


def run_detection(
    input_path: Path = INPUT,
    output_vectors: Path = OUTPUT_VECTORS,
) -> gpd.GeoDataFrame:
    output_vectors.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PREVIEW.parent.mkdir(exist_ok=True)

    with rasterio.open(input_path) as src:
        crs = src.crs
        transform = src.transform
        height = src.height

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=MODEL,
        confidence_threshold=CONF,
        device="cpu",
    )

    print(f"Running tiled inference on {input_path.name} (tiles {SLICE_SIZE}px, overlap {OVERLAP})...")
    result = get_sliced_prediction(
        image=str(input_path),
        detection_model=detection_model,
        slice_height=SLICE_SIZE,
        slice_width=SLICE_SIZE,
        overlap_height_ratio=OVERLAP,
        overlap_width_ratio=OVERLAP,
        verbose=1,
    )

    # Filter vehicle classes and build GeoDataFrame
    records = []
    for pred in result.object_prediction_list:
        if pred.category.name not in VEHICLE_CLASSES:
            continue
        b = pred.bbox  # BoundingBox with minx, miny, maxx, maxy in pixel coords
        # Convert pixel coords to raster world coords via affine transform
        wx1, wy1 = transform * (b.minx, b.miny)
        wx2, wy2 = transform * (b.maxx, b.maxy)
        records.append({
            "class": pred.category.name,
            "confidence": round(pred.score.value, 3),
            "geometry": box(wx1, wy1, wx2, wy2),
        })

    gdf = gpd.GeoDataFrame(records, crs=crs)
    print(f"\nDetected {len(gdf)} vehicles")
    if len(gdf):
        print(gdf["class"].value_counts().to_string())
        gdf.to_file(output_vectors, driver="GPKG")
        print(f"Saved: {output_vectors}")

    # Preview – draw only vehicle boxes, downscaled to max 2000px wide
    import cv2
    img = cv2.imread(str(input_path))
    scale = min(1.0, 2000 / img.shape[1])
    colors = {"small vehicle": (0, 255, 0), "large vehicle": (0, 128, 255)}
    for pred in result.object_prediction_list:
        if pred.category.name not in VEHICLE_CLASSES:
            continue
        b = pred.bbox
        cv2.rectangle(img,
                      (int(b.minx), int(b.miny)),
                      (int(b.maxx), int(b.maxy)),
                      colors.get(pred.category.name, (255, 0, 0)), 3)
    if scale < 1.0:
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    cv2.imwrite(str(OUTPUT_PREVIEW), img)
    print(f"Preview: {OUTPUT_PREVIEW}")

    return gdf


if __name__ == "__main__":
    run_detection()
