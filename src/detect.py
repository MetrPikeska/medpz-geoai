"""Vehicle detection from orthophoto using YOLOv8-OBB trained on DOTA dataset."""

from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box
from ultralytics import YOLO

INPUT = Path("data/processed/crop_1000x1000.tif")
OUTPUT_VECTORS = Path("data/vectors/vehicles.gpkg")

# DOTA classes that represent vehicles
VEHICLE_CLASSES = {"small vehicle", "large vehicle"}

# yolov8n-obb is downloaded automatically on first run (~6 MB)
MODEL = "yolov8n-obb.pt"


def run_detection(
    input_path: Path = INPUT,
    output_vectors: Path = OUTPUT_VECTORS,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 1024,
) -> gpd.GeoDataFrame:
    output_vectors.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src:
        transform = src.transform
        height = src.height
        crs = src.crs

    model = YOLO(MODEL)

    # DOTA class names for this model
    class_names = model.names  # dict {id: name}
    vehicle_ids = {k for k, v in class_names.items() if v in VEHICLE_CLASSES}
    print(f"Vehicle class IDs: {vehicle_ids} → {[class_names[i] for i in vehicle_ids]}")

    results = model.predict(
        source=str(input_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=True,
    )[0]

    records = []
    if results.obb is not None:
        for i, cls_id in enumerate(results.obb.cls.cpu().numpy()):
            cls_id = int(cls_id)
            if cls_id not in vehicle_ids:
                continue
            conf_score = float(results.obb.conf[i].cpu().numpy())
            # xyxyxyxy: 4 corner points of oriented bounding box
            pts = results.obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2)
            x1, y1 = pts[:, 0].min(), pts[:, 1].min()
            x2, y2 = pts[:, 0].max(), pts[:, 1].max()
            # Convert pixel (col, row) → raster world coordinates via affine transform
            # Y is flipped: image row 0 = raster top (max Y), row height = raster bottom (min Y)
            wx1, wy2 = transform * (x1, y1)
            wx2, wy1 = transform * (x2, y2)
            records.append({
                "class": class_names[cls_id],
                "confidence": round(conf_score, 3),
                "geometry": box(wx1, wy1, wx2, wy2),
            })

    # Save preview image with bounding boxes
    preview_path = Path("outputs/detection_preview.jpg")
    preview_path.parent.mkdir(exist_ok=True)
    preview = results.plot(labels=True, conf=True, line_width=2)
    import cv2
    cv2.imwrite(str(preview_path), preview)
    print(f"Preview: {preview_path}")

    gdf = gpd.GeoDataFrame(records, crs=crs)
    print(f"\nDetected {len(gdf)} vehicles")
    if len(gdf):
        print(gdf["class"].value_counts().to_string())
        gdf.to_file(output_vectors, driver="GPKG")
        print(f"Saved to: {output_vectors}")

    return gdf


if __name__ == "__main__":
    run_detection()
