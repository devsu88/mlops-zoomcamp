import pickle
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import os

import mlflow
from prefect import flow, task, get_run_logger

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("nyc-taxi-linreg")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


@task
def read_dataframe(filename):
    logger = get_run_logger()
    df = pd.read_parquet(filename)
    logger.info(f"Read {len(df)} rows from {filename} before preprocessing")

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    logger.info(f"Read {len(df)} rows from {filename} after preprocessing")

    return df


@task
def prepare_features(df, dv=None):
    categorical = ['PULocationID', 'DOLocationID']

    dicts = df[categorical].astype(str).to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


@task
def train_model(X, y, dv):
    with mlflow.start_run() as run:
        lr = LinearRegression()
        lr.fit(X, y)
        y_pred = lr.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(lr, artifact_path="linreg_model")

        mlflow.log_param("intercept", lr.intercept_)

        # Salva anche il DictVectorizer
        with open("models/dv.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/dv.b", artifact_path="preprocessor")

        model_path = "models/dv.b"
        model_size = os.path.getsize(model_path)
        mlflow.log_metric("model_size_bytes", model_size)

        return run.info.run_id


@flow(name="Homework Pipeline")
def main_flow(reference_date: str = None):
    logger = get_run_logger()

    if reference_date:
        run_date = datetime.strptime(reference_date, "%Y-%m-%d")
    else:
        run_date = datetime.today()

    train_date = run_date
    filename = f"data/yellow_tripdata_{train_date.year}-{train_date.month:02d}.parquet"

    logger.info(f"Training on file: {filename}")

    df = read_dataframe(filename)
    X, dv = prepare_features(df)
    y = df['duration'].values

    run_id = train_model(X, y, dv)
    logger.info(f"MLflow run_id: {run_id}")

    return run_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run NYC Taxi linear regression pipeline with Prefect.")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Reference date in format YYYY-MM-DD. Used to determine training month."
    )
    args = parser.parse_args()

    main_flow(reference_date=args.date)
