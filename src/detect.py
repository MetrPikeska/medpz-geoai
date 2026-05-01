"""SAM2 vehicle detection from orthophoto crop."""

import numpy as np
from pathlib import Path
from samgeo import SamGeo

INPUT = Path("data/processed/crop_1000x1000.tif")
OUTPUT_MASK = Path("outputs/masks.tif")
OUTPUT_VECTORS = Path("data/vectors/segments.gpkg")


def run_segmentation(
    input_path: Path = INPUT,
    output_mask: Path = OUTPUT_MASK,
    output_vectors: Path = OUTPUT_VECTORS,
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    min_area: int = 100,
) -> None:
    sam = SamGeo(
        model_type="vit_h",
        automatic=True,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
    )

    sam.generate(str(input_path), str(output_mask))

    sam.tiff_to_vector(str(output_mask), str(output_vectors))

    print(f"Masks saved to: {output_mask}")
    print(f"Vectors saved to: {output_vectors}")


if __name__ == "__main__":
    run_segmentation()
