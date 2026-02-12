import logging
import os
from datetime import datetime

import dill
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

path = os.environ.get("PROJECT_PATH", "/opt/airflow/project")


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        "id",
        "url",
        "region",
        "region_url",
        "price",
        "manufacturer",
        "image_url",
        "description",
        "posting_date",
        "lat",
        "long",
    ]
    df = df.copy()
    cols = [c for c in columns_to_drop if c in df.columns]
    return df.drop(cols, axis=1)


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "year" not in df.columns:
        return df

    q25 = df["year"].quantile(0.25)
    q75 = df["year"].quantile(0.75)
    iqr = q75 - q25
    lower = q25 - 1.5 * iqr
    upper = q75 + 1.5 * iqr

    df.loc[df["year"] < lower, "year"] = round(lower)
    df.loc[df["year"] > upper, "year"] = round(upper)
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "model" in df.columns:
        def short_model(x):
            if pd.isna(x):
                return x
            return str(x).lower().split(" ")[0]

        df.loc[:, "short_model"] = df["model"].apply(short_model)

    if "year" in df.columns:
        df.loc[:, "age_category"] = df["year"].apply(
            lambda x: "new" if x > 2013 else ("old" if x < 2006 else "average")
        )

    return df


def pipeline() -> None:
    csv_path = f"{path}/data/train/homework.csv"
    df = pd.read_csv(csv_path)

    X = df.drop("price_category", axis=1)
    y = df["price_category"]

    numerical_features = make_column_selector(dtype_include=["int64", "float64"])
    categorical_features = make_column_selector(dtype_include=object)

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("numerical", numerical_transformer, numerical_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    preprocessor = Pipeline(
        steps=[
            ("filter", FunctionTransformer(filter_data, validate=False)),
            ("outlier_remover", FunctionTransformer(remove_outliers, validate=False)),
            ("feature_creator", FunctionTransformer(create_features, validate=False)),
            ("column_transformer", column_transformer),
        ]
    )

    models = [
        LogisticRegression(solver="liblinear"),
        RandomForestClassifier(),
        SVC(),
    ]

    best_score = 0.0
    best_pipe = None

    for model in models:
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )

        score = cross_val_score(pipe, X, y, cv=4, scoring="accuracy")
        logging.info(
            f"model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}"
        )

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    logging.info(
        f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}'
    )

    best_pipe.fit(X, y)

    os.makedirs(f"{path}/data/models", exist_ok=True)
    model_filename = f'{path}/data/models/cars_pipe_{datetime.now().strftime("%Y%m%d%H%M")}.pkl'

    with open(model_filename, "wb") as file:
        dill.dump(best_pipe, file)

    logging.info(f"Model is saved as {model_filename}")


if __name__ == "__main__":
    pipeline()
