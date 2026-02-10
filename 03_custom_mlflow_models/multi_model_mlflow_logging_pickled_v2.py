# Databricks notebook source
# MAGIC %md ### Custom MLflow model and distributed training with PandasUDFs

# COMMAND ----------

# MAGIC %pip install mlflow==3.0.1
# MAGIC %restart_python

# COMMAND ----------

import joblib
import pickle
from itertools import islice
import urllib.request
import json
import os
import requests
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from pyspark.sql.types import StructType, StructField, StringType, FloatType, BinaryType, DoubleType, ArrayType
from pyspark.sql.functions import col
import pyspark.sql.functions as func
import mlflow
from mlflow.tracking import MlflowClient
from databricks.sdk import WorkspaceClient

# COMMAND ----------

mlflow.sklearn.autolog(disable=True)
exerpiment_path = "/Users/marshall.carter@databricks.com/workshop_experiment_mlc"
mlflow.set_experiment(exerpiment_path) 
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

dbutils.widgets.text('catalog_name','','Enter catalog name')
dbutils.widgets.text('schema_name','','Enter schema name')

# COMMAND ----------

catalog_name = dbutils.widgets.get('catalog_name')
schema_name = dbutils.widgets.get('schema_name')
uc_location = f"{catalog_name}.{schema_name}"
output_delta_location = f"{uc_location}.regression_models"
print(f"UC location: {uc_location}")

# COMMAND ----------

client = MlflowClient()

# COMMAND ----------

mlflow.autolog(disable=True)

# COMMAND ----------

# MAGIC %md ## Create a synthetic training dataset
# MAGIC
# MAGIC A regression model will be fit for each group; vary the number of groups to align with your actual data volumes.

# COMMAND ----------

# Configure the data volumes
n_skus = 10000
n_features_per_sku = 10
n_samples_per_sku = 100

# DBFS directory or UC volume to store model artifacts
regression_model_dbfs_directory = '/Volumes/shared/mlc_schema/regression_models'
"""
regression_model_dbfs_directory_python = f"/{regression_model_dbfs_directory.replace(':', '')}"

# Create the directory or delete and recreate if exists
try: 
  dbutils.fs.rm(regression_model_dbfs_directory, recurse=True)
  dbutils.fs.mkdirs(regression_model_dbfs_directory)
except:
  dbutils.fs.mkdirs(regression_model_dbfs_directory)

dbutils.fs.ls(regression_model_dbfs_directory)
"""

# COMMAND ----------

def create_skus(skus=n_skus):
  skus = [[f'sku_{str(n+1).zfill(2)}'] for n in range(n_skus)]
  schema = StructType()
  schema.add('sku', StringType())
  
  return spark.createDataFrame(skus, schema=schema)

# COMMAND ----------

skus = create_skus()
display(skus.limit(10))

# COMMAND ----------

skus.count()

# COMMAND ----------

# MAGIC %md ### Create sku-level features using a PandasUDF
# MAGIC
# MAGIC This PandasUDF to distribute the computation at the group level

# COMMAND ----------

def get_feature_col_names(n_features_per_group=n_features_per_sku):
  return [f"features_{n}" for n in range(n_features_per_group)]


def configure_features_udf(n_features_per_group=n_features_per_sku, n_samples_per_group=n_samples_per_sku):

  def create_sku_features(group_data: pd.DataFrame) -> pd.DataFrame:

    features, target = make_regression(n_samples=n_samples_per_sku, n_features=n_features_per_group)
    feature_names = get_feature_col_names()
    df = pd.DataFrame(features, columns=feature_names)

    df['target'] = target.tolist()

    group_name = group_data["sku"].loc[0]
    df['sku'] = group_name

    col_order = ['sku'] + feature_names + ['target']

    return df[col_order]
  
  return create_sku_features


spark_schema = StructType()
spark_schema.add('sku', StringType())
for feature_name in get_feature_col_names():
  spark_schema.add(feature_name, FloatType())
spark_schema.add('target', FloatType())

# COMMAND ----------

udf = configure_features_udf()
features = skus.groupBy('sku').applyInPandas(udf, spark_schema)
display(features.limit(10))

# COMMAND ----------

print(f"Number of training observations: {features.count():,} across {n_skus:,} skus")

# COMMAND ----------

# MAGIC %md #### A Pandas UDF that trains a regression model for each group and returns the model as serialized (pickled) bytes.

# COMMAND ----------

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def config_models_udf(*, sample_fraction: float | None = None, seed: int | None = None):
    rng = np.random.default_rng(seed)

    def fit_group_models(group_data: pd.DataFrame) -> pd.DataFrame:
        sku_name = group_data["sku"].iloc[0]

        features_df = group_data.drop(columns=["sku", "target"])
        feature_names = list(features_df.columns)

        if sample_fraction is not None:
            if not 0 < sample_fraction <= 1:
                raise ValueError("sample_fraction must be in (0, 1].")
            sample_size = max(1, int(len(feature_names) * sample_fraction))
            feature_names = list(rng.choice(feature_names, size=sample_size, replace=False))
            features_df = features_df[feature_names]

        X = features_df
        y = group_data["target"]

        model = LinearRegression().fit(X.to_numpy(), np.asarray(y))
        model_bytes = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)

        predictions = model.predict(X.to_numpy())
        mse = mean_squared_error(y, predictions)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)

        return pd.DataFrame(
            {
                "sku": [sku_name],
                "model_bytes": [model_bytes],
                "metric_mse": [float(mse)],
                "metric_rmse": [rmse],
                "metric_mae": [float(mae)],
                "metric_r2": [float(r2)],
                "feature_names": [feature_names],
            }
        )

    return fit_group_models

spark_schema = StructType([
    StructField("sku", StringType(), nullable=False),
    StructField("model_bytes", BinaryType(), nullable=False),
    StructField("metric_mse", DoubleType(), nullable=False),
    StructField("metric_rmse", DoubleType(), nullable=False),
    StructField("metric_mae", DoubleType(), nullable=False),
    StructField("metric_r2", DoubleType(), nullable=False),
    StructField("feature_names", ArrayType(StringType()), nullable=False),
])

# COMMAND ----------

# MAGIC %md Note: The PandasUDF example returns only the sku name and model bytes, though it should also be configured to return other useful metadata, like evaluations statistics. These statistics and other metadata could then be saved in .csv format as an additional MLflow model artifact that could load loaded into a notebook for analysis later.

# COMMAND ----------

fit_models_udf = config_models_udf(sample_fraction = 0.75)
# The training was done using a Spark cluster with 32 CPU cores total across its workers
features = features.repartition(32, 'sku')
fitted_models = features.groupBy('sku').applyInPandas(fit_models_udf, schema=spark_schema)

fitted_models.write.mode('overwrite').format('delta').saveAsTable(output_delta_location)
display(spark.table(output_delta_location).limit(10))

# COMMAND ----------

# MAGIC %md ####Collect the pickled bytes into a single pickled object that can be efficiently logged to MLflow

# COMMAND ----------

model_pd = spark.table(output_delta_location).toPandas()

bundle = {
    row["sku"]: {
        "model_bytes": row["model_bytes"],
        "feature_names": row["feature_names"],   # already a list from the UDF
    }
    for _, row in model_pd.iterrows()
}

bundle_path = f"{regression_model_dbfs_directory}/all_regression_models.pkl"
#bundle_path = "/dbfs/Users/marshall.carter@databricks.com/all_regression_models.pkl"
with open(bundle_path, "wb") as f:
    pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

# COMMAND ----------

size_bytes = os.path.getsize(bundle_path)
size_mb = size_bytes / (1024 * 1024)

print(f"Pickle size: {size_bytes:,} bytes ({size_mb:.2f} MB)")

# COMMAND ----------

metrics_df = spark.table(output_delta_location).select(
       "sku",
       "metric_mse",
       "metric_rmse",
       "metric_mae",
       "metric_r2",
       "feature_names",
   ).toPandas()
metrics_path = f"{regression_model_dbfs_directory}/regression_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)

# COMMAND ----------

metrics_df.head()

# COMMAND ----------

# MAGIC %md #### Define the custom MLflow model that loads the pickled models

# COMMAND ----------

class AllInMemoryModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        bundle_path = context.artifacts["regression_models"]
        with open(bundle_path, "rb") as f:
            raw = pickle.load(f)

        self.models = {}
        for sku, entry in raw.items():
            model = pickle.loads(entry["model_bytes"])
            feature_names = entry["feature_names"]
            self.models[sku] = {"model": model, "feature_names": feature_names}

    def predict(self, context, model_input):
        key_col = "sku"
        preds = []
        for _, row in model_input.iterrows():
            sku = row[key_col]
            entry = self.models.get(sku)
            if entry is None:
                raise KeyError(f"No model found for sku={sku}")

            feature_names = entry["feature_names"]
            missing = [c for c in feature_names if c not in row.index]
            if missing:
                raise KeyError(f"Missing feature(s) {missing} for sku={sku}")

            X = row[feature_names].to_numpy().reshape(1, -1)
            preds.append(entry["model"].predict(X)[0])

        return pd.DataFrame({"predictions": preds})

# COMMAND ----------

# MAGIC %md ####Log the MLflow model and model artifacts to a single Experiment Run

# COMMAND ----------

with mlflow.start_run() as run:
  run_id = run.info.run_id

  input_example = features.drop("target").toPandas()
  input_example = input_example.sample(5)
  input_example.reset_index(inplace=True, drop=True)

  mlflow.pyfunc.log_model(name="model", 
                          python_model=AllInMemoryModel(), 
                          artifacts={"regression_models": bundle_path},
                          input_example=input_example)
  
  mlflow.log_artifact(metrics_path, artifact_path="evaluations")

  avg_metrics = {
    "avg_mse": metrics_df["metric_mse"].mean(),
    "avg_rmse": metrics_df["metric_rmse"].mean(),
    "avg_mae": metrics_df["metric_mae"].mean(),
    "avg_r2": metrics_df["metric_r2"].mean(),
  }
  mlflow.log_metrics(avg_metrics)

# COMMAND ----------

local_path = client.download_artifacts(run_id, "evaluations/regression_metrics.csv")
metrics_df = pd.read_csv(local_path)
metrics_df.head()

# COMMAND ----------

metrics_df.shape

# COMMAND ----------

import mlflow
from pyspark.sql.functions import struct, col

model_uri = f"runs:/{run_id}/model"

# Load model as a Spark UDF. Override result_type if the model does not return double values.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)

predictions_df = features.drop("target")

# Predict on a Spark DataFrame.
predictions_df = predictions_df.withColumn('predictions', loaded_model(struct(*map(col, predictions_df.columns))))
display(predictions_df)

# COMMAND ----------

print(f"Number of predictions: {predictions_df.count():,}")
