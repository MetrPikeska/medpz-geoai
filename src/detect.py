"""Vehicle detection from orthophoto using YOLOv8-OBB trained on DOTA dataset."""

from pathlib import Path
import geopandas as gpd
import numpy as np
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
            records.append({
                "class": class_names[cls_id],
                "confidence": round(conf_score, 3),
                "geometry": box(x1, y1, x2, y2),
            })

    gdf = gpd.GeoDataFrame(records, crs=None)
    print(f"\nDetected {len(gdf)} vehicles")
    if len(gdf):
        print(gdf["class"].value_counts().to_string())
        gdf.to_file(output_vectors, driver="GPKG")
        print(f"Saved to: {output_vectors}")

    return gdf


if __name__ == "__main__":
    run_detection()
