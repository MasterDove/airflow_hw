import os
import glob
import json
import dill
import pandas as pd
from datetime import datetime

PROJECT_PATH = os.environ.get("PROJECT_PATH", ".")


def load_latest_model(models_dir: str):
    model_files = glob.glob(os.path.join(models_dir, "*.pkl"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {models_dir}")

    latest_model = max(model_files, key=os.path.getmtime)
    with open(latest_model, "rb") as f:
        model = dill.load(f)

    return model, latest_model


def load_test_data(test_dir: str) -> pd.DataFrame:
    records = []

    for file_path in glob.glob(os.path.join(test_dir, "*.json")):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "car_id" not in data:
            data["car_id"] = os.path.splitext(os.path.basename(file_path))[0]

        records.append(data)

    if not records:
        raise FileNotFoundError(f"No test json files found in {test_dir}")

    return pd.DataFrame(records)


def save_predictions(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    filename = f"preds_{datetime.now().strftime('%Y%m%d%H%M')}.csv"
    full_path = os.path.join(output_dir, filename)
    df.to_csv(full_path, index=False)
    return full_path


def predict():
    models_dir = os.path.join(PROJECT_PATH, "data", "models")
    test_dir = os.path.join(PROJECT_PATH, "data", "test")
    preds_dir = os.path.join(PROJECT_PATH, "data", "predictions")

    model, model_path = load_latest_model(models_dir)
    df_test = load_test_data(test_dir)

    car_id = df_test["car_id"].astype(str)
    X = df_test.drop(columns=["car_id"], errors="ignore")

    preds = model.predict(X)

    result = pd.DataFrame({
        "car_id": car_id,
        "pred": preds
    })

    output_file = save_predictions(result, preds_dir)

    print(f"Model loaded from: {model_path}")
    print(f"Predictions saved to: {output_file}")

    return output_file


if __name__ == "__main__":
    predict()
