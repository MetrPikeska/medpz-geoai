"""Batch parameter tuning – runs detection with multiple parameter combinations."""

from pathlib import Path
from itertools import product

import cv2
import rasterio
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

INPUT = Path("data/processed/crop_1000x1000.tif")
VEHICLE_CLASSES = {"small vehicle", "large vehicle"}

# modely: n=nano, s=small, m=medium – stáhnou se automaticky
PARAM_GRID = {
    "model":      ["yolov8n-obb.pt", "yolov8s-obb.pt", "yolov8m-obb.pt"],
    "conf":       [0.05, 0.1, 0.2],
    "slice_size": [128, 256, 512],
    "overlap":    [0.3],
}


def detect(input_path, model_path, conf, slice_size, overlap):
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=conf,
        device="cpu",
    )
    result = get_sliced_prediction(
        image=str(input_path),
        detection_model=model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        verbose=0,
    )
    vehicles = [p for p in result.object_prediction_list if p.category.name in VEHICLE_CLASSES]
    return vehicles


def save_preview(input_path, vehicles, output_path):
    img = cv2.imread(str(input_path))
    colors = {"small vehicle": (0, 255, 0), "large vehicle": (0, 128, 255)}
    for pred in vehicles:
        b = pred.bbox
        cv2.rectangle(img, (int(b.minx), int(b.miny)), (int(b.maxx), int(b.maxy)),
                      colors.get(pred.category.name, (255, 0, 0)), 2)
    cv2.putText(img, f"vehicles: {len(vehicles)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imwrite(str(output_path), img)


def run_batch(input_path: Path = INPUT):
    out_dir = Path("outputs/tune")
    out_dir.mkdir(parents=True, exist_ok=True)

    combinations = list(product(
        PARAM_GRID["model"],
        PARAM_GRID["conf"],
        PARAM_GRID["slice_size"],
        PARAM_GRID["overlap"],
    ))

    print(f"Running {len(combinations)} combinations on {input_path.name}\n")
    summary = []

    for model_path, conf, slice_size, overlap in combinations:
        model_tag = model_path.replace("yolov8", "").replace("-obb.pt", "")
        tag = f"{model_tag}_conf{conf}_slice{slice_size}_overlap{overlap}"
        print(f"[{tag}] ...", end=" ", flush=True)

        vehicles = detect(input_path, model_path, conf, slice_size, overlap)

        n_small = sum(1 for v in vehicles if v.category.name == "small vehicle")
        n_large = len(vehicles) - n_small
        avg_w = sum(v.bbox.maxx - v.bbox.minx for v in vehicles) / max(len(vehicles), 1)
        avg_h = sum(v.bbox.maxy - v.bbox.miny for v in vehicles) / max(len(vehicles), 1)
        print(f"{len(vehicles)} vehicles  (small={n_small}, large={n_large})  avg_box={avg_w:.0f}×{avg_h:.0f}px")

        save_preview(input_path, vehicles, out_dir / f"preview_{tag}.jpg")

        summary.append({
            "model": model_tag, "conf": conf, "slice": slice_size,
            "overlap": overlap, "total": len(vehicles),
            "small": n_small, "large": n_large,
            "avg_w": avg_w, "avg_h": avg_h,
        })

    print("\n--- SUMMARY ---")
    print(f"{'model':>4} {'conf':>5} {'slice':>5} {'total':>6} {'small':>6} {'large':>6} {'avg_box':>10}")
    print("-" * 52)
    for r in summary:
        print(f"{r['model']:>4} {r['conf']:>5} {r['slice']:>5} "
              f"{r['total']:>6} {r['small']:>6} {r['large']:>6} "
              f"  {r['avg_w']:.0f}×{r['avg_h']:.0f}px")

    print(f"\nPreviews: {out_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    args = parser.parse_args()
    run_batch(args.input)
