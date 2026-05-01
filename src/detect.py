"""Vehicle detection from orthophoto using YOLOv8-OBB + SAHI tiling."""

import argparse
from pathlib import Path

import cv2
import geopandas as gpd
import rasterio
from shapely.geometry import box
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

INPUT = Path("data/raw/105000312_0000_00_0000_P00_01.tif")
OUTPUT_VECTORS = Path("data/vectors/vehicles.gpkg")
OUTPUT_PREVIEW = Path("outputs/detection_preview.jpg")

VEHICLE_CLASSES = {"small vehicle", "large vehicle"}
MODEL = "yolov8n-obb.pt"


def run_detection(
    input_path: Path = INPUT,
    output_vectors: Path = OUTPUT_VECTORS,
    conf: float = 0.25,
    slice_size: int = 640,
    overlap: float = 0.2,
) -> gpd.GeoDataFrame:
    output_vectors.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PREVIEW.parent.mkdir(exist_ok=True)

    with rasterio.open(input_path) as src:
        crs = src.crs
        transform = src.transform

    print(f"conf={conf}  slice={slice_size}px  overlap={overlap}")
    print(f"Input: {input_path.name}")

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=MODEL,
        confidence_threshold=conf,
        device="cpu",
    )

    result = get_sliced_prediction(
        image=str(input_path),
        detection_model=detection_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        verbose=1,
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
            "geometry": box(wx1, wy1, wx2, wy2),
        })

    gdf = gpd.GeoDataFrame(records, crs=crs)
    print(f"\nDetected {len(gdf)} vehicles")
    if len(gdf):
        print(gdf["class"].value_counts().to_string())
        print(f"Confidence stats:\n{gdf['confidence'].describe().round(3).to_string()}")
        gdf.to_file(output_vectors, driver="GPKG")
        print(f"Saved: {output_vectors}")

    # Preview – draw only vehicle boxes, resize to max 2000px wide
    img = cv2.imread(str(input_path))
    colors = {"small vehicle": (0, 255, 0), "large vehicle": (0, 128, 255)}
    for pred in result.object_prediction_list:
        if pred.category.name not in VEHICLE_CLASSES:
            continue
        b = pred.bbox
        cv2.rectangle(img,
                      (int(b.minx), int(b.miny)),
                      (int(b.maxx), int(b.maxy)),
                      colors.get(pred.category.name, (255, 0, 0)), 3)
    scale = min(1.0, 2000 / img.shape[1])
    if scale < 1.0:
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    cv2.imwrite(str(OUTPUT_PREVIEW), img)
    print(f"Preview: {OUTPUT_PREVIEW}")

    return gdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect vehicles in orthophoto")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--slice-size", type=int, default=640, help="Tile size in px (default: 640)")
    parser.add_argument("--overlap", type=float, default=0.2, help="Tile overlap ratio (default: 0.2)")
    args = parser.parse_args()

    run_detection(
        input_path=args.input,
        conf=args.conf,
        slice_size=args.slice_size,
        overlap=args.overlap,
    )
