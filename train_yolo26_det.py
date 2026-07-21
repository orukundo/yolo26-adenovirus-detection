# -*- coding: utf-8 -*-
"""
YOLO26 object detection benchmark for adenovirus in TEM images.

This script trains YOLO26n/s/m/l/x object detection model variants using a selected
augmentation profile, evaluating each model on validation and test splits,
saving the best-performing weights, and exporting predictions and comparison
metrics.

@author: Olivier.Rukundo, Ph.D., University of Limerick, July 15, 2026
"""

import csv
import shutil
import time
from pathlib import Path

import torch
from ultralytics import YOLO


base_path = Path(__file__).resolve().parent

models_to_train = {
    "26n": "yolo26n.pt",
    "26s": "yolo26s.pt",
    "26m": "yolo26m.pt",
    "26l": "yolo26l.pt",
    "26x": "yolo26x.pt"
}

epochs = 300
patience = 100
image_size = 1376
batch_size = 2
random_seed = 42

augmentation_profiles = {
    "no_augmentation": {
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "close_mosaic": 0
    },

    "geometric_only": {
        "degrees": 180.0,
        "translate": 0.10,
        "scale": 0.20,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.5,
        "fliplr": 0.5,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "close_mosaic": 0
    },

    "geometric_mosaic": {
        "degrees": 180.0,
        "translate": 0.10,
        "scale": 0.20,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "close_mosaic": 10
    }
}

augmentation_name = "geometric_only"
augmentation_settings = augmentation_profiles[augmentation_name]

device = 0 if torch.cuda.is_available() else "cpu"

dataset_yaml_path = base_path / "dataset.yaml"
runs_folder = base_path / "runs"
models_folder = base_path / "trained_models"
results_folder = base_path / "comparison_results"

validation_csv = results_folder / "yolo26_validation_comparison.csv"
test_csv = results_folder / "yolo26_test_comparison.csv"
failures_csv = results_folder / "yolo26_failed_models.csv"

models_folder.mkdir(parents=True, exist_ok=True)
results_folder.mkdir(parents=True, exist_ok=True)
runs_folder.mkdir(parents=True, exist_ok=True)

dataset_yaml = f"""
path: {base_path}
train: train/images
val: val/images
test: test/images

names:
  0: adenovirus
"""

dataset_yaml_path.write_text(
    dataset_yaml.strip() + "\n",
    encoding="utf-8"
)

print(f"Dataset YAML: {dataset_yaml_path}")
print(f"Device: {device}")
print(f"Models: {', '.join(models_to_train.keys())}")


fieldnames = [
    "Model",
    "Weights",
    "mAP50-95",
    "mAP50",
    "Precision",
    "Recall",
    "Parameters",
    "Inference time (ms/image)",
    "Training time (seconds)",
    "Training time (minutes)",
    "Training time (hours)",
    "Best model path"
]


def write_results_csv(csv_path, rows):
    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(rows)


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.model.parameters()
    )


def get_metric_value(metrics, attribute, default=0.0):
    value = getattr(metrics.box, attribute, default)

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_inference_time(metrics):
    speed = getattr(metrics, "speed", {})

    if isinstance(speed, dict):
        return float(speed.get("inference", 0.0))

    return 0.0


def evaluate_model(
    trained_model,
    split,
    model_name,
    training_seconds,
    parameter_count,
    best_model_path
):
    evaluation = trained_model.val(
        data=str(dataset_yaml_path),
        split=split,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        plots=True,
        save_json=False,
        project=str(results_folder / split),
        name=model_name,
        exist_ok=True,
        verbose=True
    )

    return {
        "Model": model_name,
        "Weights": models_to_train[model_name],
        "mAP50-95": round(
            get_metric_value(evaluation, "map"),
            6
        ),
        "mAP50": round(
            get_metric_value(evaluation, "map50"),
            6
        ),
        "Precision": round(
            get_metric_value(evaluation, "mp"),
            6
        ),
        "Recall": round(
            get_metric_value(evaluation, "mr"),
            6
        ),
        "Parameters": parameter_count,
        "Inference time (ms/image)": round(
            get_inference_time(evaluation),
            4
        ),
        "Training time (seconds)": round(
            training_seconds,
            2
        ),
        "Training time (minutes)": round(
            training_seconds / 60,
            2
        ),
        "Training time (hours)": round(
            training_seconds / 3600,
            3
        ),
        "Best model path": str(best_model_path)
    }


validation_results = []
test_results = []
failed_models = []

for model_name, weights in models_to_train.items():
    print("\n" + "=" * 70)
    print(f"Training YOLO{model_name}")
    print(f"Initial weights: {weights}")
    print("=" * 70)

    try:
        model = YOLO(weights)

        parameter_count = count_parameters(model)

        print(f"Parameters: {parameter_count:,}")
        print("Starting training...")

        start_time = time.perf_counter()

        training_result = model.train(
            data=str(dataset_yaml_path),
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            patience=patience,
            project=str(runs_folder),
            name=f"adenovirus_yolo{model_name}",
            exist_ok=True,
            device=device,
            seed=random_seed,
            deterministic=True,
            optimizer="auto",
            plots=True,
            save=True,
            verbose=True,
            **augmentation_settings
        )

        training_seconds = time.perf_counter() - start_time

        training_directory = Path(training_result.save_dir)

        best_model_source = (
            training_directory
            / "weights"
            / "best.pt"
        )

        if not best_model_source.exists():
            raise FileNotFoundError(
                f"Best weights not found: {best_model_source}"
            )

        best_model_destination = (
            models_folder
            / f"best_adenovirus_yolo{model_name}.pt"
        )

        shutil.copy2(
            best_model_source,
            best_model_destination
        )

        print(
            f"Training completed in "
            f"{training_seconds / 3600:.3f} hours"
        )

        print(
            f"Best model saved to: "
            f"{best_model_destination}"
        )

        trained_model = YOLO(
            str(best_model_destination)
        )

        parameter_count = count_parameters(
            trained_model
        )

        print("Evaluating validation split...")

        validation_row = evaluate_model(
            trained_model=trained_model,
            split="val",
            model_name=model_name,
            training_seconds=training_seconds,
            parameter_count=parameter_count,
            best_model_path=best_model_destination
        )

        validation_results.append(validation_row)

        write_results_csv(
            validation_csv,
            validation_results
        )

        print("Evaluating test split...")

        test_row = evaluate_model(
            trained_model=trained_model,
            split="test",
            model_name=model_name,
            training_seconds=training_seconds,
            parameter_count=parameter_count,
            best_model_path=best_model_destination
        )

        test_results.append(test_row)

        write_results_csv(
            test_csv,
            test_results
        )

        print(f"\nYOLO{model_name} results")
        print(
            f"Validation mAP50-95: "
            f"{validation_row['mAP50-95']:.6f}"
        )
        print(
            f"Test mAP50-95: "
            f"{test_row['mAP50-95']:.6f}"
        )
        print(
            f"Test precision: "
            f"{test_row['Precision']:.6f}"
        )
        print(
            f"Test recall: "
            f"{test_row['Recall']:.6f}"
        )
        print(
            f"Inference time: "
            f"{test_row['Inference time (ms/image)']:.4f} ms/image"
        )

        del model
        del trained_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as error:
        print(f"YOLO{model_name} failed")
        print(f"Reason: {error}")

        failed_models.append(
            {
                "Model": model_name,
                "Weights": weights,
                "Error": str(error)
            }
        )

        with failures_csv.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "Model",
                    "Weights",
                    "Error"
                ]
            )
            writer.writeheader()
            writer.writerows(failed_models)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


print("\n" + "=" * 70)
print("YOLO26 MODEL COMPARISON COMPLETED")
print("=" * 70)

print(f"Validation CSV: {validation_csv}")
print(f"Test CSV: {test_csv}")

if failed_models:
    print(f"Failed-model report: {failures_csv}")

print("\nTest-set comparison")

header = (
    f"{'Model':<8}"
    f"{'mAP50-95':<14}"
    f"{'Precision':<14}"
    f"{'Recall':<14}"
    f"{'Parameters':<16}"
    f"{'Inference ms':<16}"
    f"{'Training hours':<16}"
)

print(header)
print("-" * len(header))

for row in test_results:
    print(
        f"{row['Model']:<8}"
        f"{row['mAP50-95']:<14.6f}"
        f"{row['Precision']:<14.6f}"
        f"{row['Recall']:<14.6f}"
        f"{row['Parameters']:<16,}"
        f"{row['Inference time (ms/image)']:<16.4f}"
        f"{row['Training time (hours)']:<16.3f}"
    )
