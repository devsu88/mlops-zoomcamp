# pipeline.py

import pickle
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import mlflow
from prefect import flow, task, get_run_logger

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


@task
def read_dataframe(year, month):
    logger = get_run_logger()
    url = f'data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    logger.info(f"Read {len(df)} rows for {year}-{month:02d}")
    return df


@task
def create_X(df, dv=None):
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


@task
def train_model(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        return run.info.run_id


@flow(name="NYC Taxi Monthly Training Flow")
def main_flow(reference_date: str = None):
    logger = get_run_logger()

    # Calcola le date di riferimento
    if reference_date:
        run_date = datetime.strptime(reference_date, "%Y-%m-%d")
    else:
        run_date = datetime.today()

    train_date = run_date - relativedelta(months=2)
    val_date = run_date - relativedelta(months=1)

    logger.info(f"Training on: {train_date.year}-{train_date.month:02d}")
    logger.info(f"Validation on: {val_date.year}-{val_date.month:02d}")

    df_train = read_dataframe(train_date.year, train_date.month)
    df_val = read_dataframe(val_date.year, val_date.month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    y_train = df_train['duration'].values
    y_val = df_val['duration'].values

    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    logger.info(f"MLflow run_id: {run_id}")

    return run_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run NYC Taxi pipeline with Prefect.")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Reference date in format YYYY-MM-DD. Used to determine training/validation months."
    )
    args = parser.parse_args()

    main_flow(reference_date=args.date)
