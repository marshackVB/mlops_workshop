# Databricks notebook source
# MAGIC %md ### Feature table transformations examples. 

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering==0.14.0
# MAGIC %restart_python

# COMMAND ----------

from datetime import datetime, timedelta
from copy import deepcopy
import time

from databricks.feature_store import FeatureStoreClient, FeatureLookup, feature_table
from pyspark.sql.functions import col
import pyspark.sql.functions as func

# COMMAND ----------

dbutils.widgets.text('catalog_name','','Enter catalog name')
dbutils.widgets.text('schema_name','','Enter schema name')

# COMMAND ----------

catalog_name = dbutils.widgets.get('catalog_name')
schema_name = dbutils.widgets.get('schema_name')
catalog_schema_name = f"{catalog_name}.{schema_name}"
print(catalog_schema_name)

# COMMAND ----------

fs = FeatureStoreClient()

# COMMAND ----------

# MAGIC %md # Creating an initial feature table

# COMMAND ----------

# MAGIC %md ### Specify a feature transformation function

# COMMAND ----------

def compute_passenger_demographic_features(df):
             # Extract prefic from name, such as Mr. Mrs., etc.
  return (df.withColumn('NamePrefix', func.regexp_extract(col('Name'), '([A-Za-z]+)\.', 1))
             # Extract a secondary name in the Name column if one exists
            .withColumn('NameSecondary_extract', func.regexp_extract(col('Name'), '\(([A-Za-z ]+)\)', 1))            
            .drop('NameSecondary_extract')
            .selectExpr("PassengerId",
                        "Name",
                        "Sex",
                        "case when Age = 'NaN' then NULL else Age end as Age",
                        "SibSp",
                        "NamePrefix"))

# COMMAND ----------

df = spark.table(f"{catalog_schema_name}.passenger_demographic_features")
passenger_demographic_features = compute_passenger_demographic_features(df)
display(passenger_demographic_features)

# COMMAND ----------

# MAGIC %md ### Create and populate the feature table

# COMMAND ----------

feature_table_name = f"{catalog_schema_name}.demographic_features_backfill"

# If the feature table has already been created, no need to recreate
try:
  fs.get_table(feature_table_name)
  print("Feature table entry already exists")
  pass
  
except Exception:
  fs.create_table(name = feature_table_name,
                          primary_keys = 'PassengerId',
                          schema = passenger_demographic_features.schema,
                          description = 'Demographic-related features for Titanic passengers')

# Overwrite records if they exist, otherwise append.
fs.write_table(
  name= feature_table_name,
  df = passenger_demographic_features,
  mode = 'merge'
  )

# COMMAND ----------

# MAGIC %md View feature table

# COMMAND ----------

features = fs.read_table(
  name = feature_table_name,
)
display(features)

# COMMAND ----------

# MAGIC %md ### Calculate new feature table columns. 
# MAGIC These columns will need to be attached to the existing feature table. There are [two ways](https://docs.databricks.com/machine-learning/feature-store/feature-tables.html#add-new-features-to-an-existing-feature-table) to update an existing feature table.  
# MAGIC
# MAGIC   1. Create a table of new features to join to the existing table (example below)
# MAGIC   2. Refactor the original feature transformation function to include the new column.

# COMMAND ----------

def compute_updated_features(df):
  
  return (df.withColumn('NameSecondary_extract', func.regexp_extract(col('Name'), '\(([A-Za-z ]+)\)', 1))
             # Create a feature indicating if a secondary name is present in the Name column
            .selectExpr("*", """case when length(NameSecondary_extract) > 0 then NameSecondary_extract 
                                else NULL end as NameSecondary""")
            .drop('NameSecondary_extract')
            .selectExpr("PassengerId",
                        "NameSecondary",
                        "case when NameSecondary is not NULL then '1' else '0' end as NameMultiple"))

# COMMAND ----------

df = spark.table(f"{catalog_schema_name}.passenger_demographic_features")
passenger_demographic_feature_updates = compute_updated_features(df)
display(passenger_demographic_feature_updates)

# COMMAND ----------

# MAGIC %md ### Add features to existing table. 
# MAGIC In this case, the new columns to not exist for any observations in the feature store; so the effect is a left join.  
# MAGIC
# MAGIC Note that the merge command can also be used to update [only a subset of rows](https://docs.databricks.com/machine-learning/feature-store/feature-tables.html#update-only-specific-rows-in-a-feature-tableIn) in a feature table.

# COMMAND ----------

# The write table operation has not changed; only the table writing
# to the feature table is different
fs.write_table(
  name= feature_table_name,
  df = passenger_demographic_feature_updates,
  mode = 'merge'
)

# COMMAND ----------

features = fs.read_table(
  name = feature_table_name,
)
display(features)

# COMMAND ----------

# MAGIC %md # Working with time series features

# COMMAND ----------

cols = ["PassengerId", "Name", "Sex", "Age", "SibSp", "NamePrefix", "NameSecondary", "Name"]

# COMMAND ----------

ticket_features_table = spark.table(f"{catalog_schema_name}.passenger_ticket_feautures")
demographic_features_table = spark.table(f"{catalog_schema_name}.passenger_demographic_features")

#demographic_features_table = fs.read_table(name = f"{catalog_schema_name}.demographic_features")
#ticket_features_table = fs.read_table(name = f"{catalog_schema_name}.ticket_features")

# COMMAND ----------

# MAGIC %md ### Generate time series tables

# COMMAND ----------

def convert_timestamp(ts):
  ts = datetime.strptime(ts, "%m/%d/%Y")
  ts = datetime.timestamp(ts)
  return ts
 
def convert_timestamp_lst(timestamps_lst):
  converted_timestamps = [convert_timestamp(ts) for ts in timestamps_lst]
  return converted_timestamps

def create_timestamp_df(df, timestamps):
  timestamps_cp = deepcopy(timestamps)
  first_ts = timestamps_cp.pop(0)
  df_ts = df.withColumn('ts', func.lit(first_ts).cast("timestamp"))
 
  for ts in timestamps_cp:
    next_ts = df.withColumn('ts', func.lit(ts).cast("timestamp"))
    df_ts = df_ts.unionAll(next_ts)

  return df_ts

timestamps_strs = ["12/01/2025", "01/01/2026", "02/01/2026"]
timestamps = convert_timestamp_lst(timestamps_strs)
demographic_features_table_ts = create_timestamp_df(demographic_features_table, timestamps)
display(demographic_features_table_ts)

# COMMAND ----------

display(demographic_features_table_ts.groupBy(col('ts')).count())

# COMMAND ----------

# MAGIC %md Perform same operations for ticket features table

# COMMAND ----------

ticket_features_table_ts = create_timestamp_df(ticket_features_table, timestamps)
display(ticket_features_table_ts)

# COMMAND ----------

# MAGIC %md ### Create and populate the time series feature tables

# COMMAND ----------

ts_feature_table_names = [(f'{catalog_schema_name}.demographic_features_ts', demographic_features_table_ts), 
                          (f'{catalog_schema_name}.ticket_features_ts', ticket_features_table_ts)]

# Create Feature Tables
for table_name, source_df in ts_feature_table_names:
  try:
    fs.get_table(table_name)
    print("Feature table entry already exists")
    pass
    
  except Exception:
    fs.create_table(name = table_name,
                           primary_keys = 'PassengerId',
                           timestamp_keys= "ts",
                           schema = source_df.schema,
                           description = 'Demographic-related features for Titanic passengers with timestamps')

  # Write to Feature Tables
  fs.write_table(
      name= table_name,
      df = source_df,
      mode = 'merge'
  )

# COMMAND ----------

# MAGIC %md ### Generate distict records for model training, including each record's required 'as of date' for its features

# COMMAND ----------

import numpy as np
import pandas as pd

# COMMAND ----------

df = spark.table(f"{catalog_schema_name}.passenger_labels").select("PassengerId", "Survived").distinct().toPandas()
distinct_ids = df.shape[0]
random_timestamps = pd.DataFrame([np.random.choice(timestamps) for n in range(distinct_ids)], columns=['ts'])
df = pd.concat([df, random_timestamps], axis=1)

id_label_ts = (spark.createDataFrame(df)
                    .select("PassengerId", "Survived", col("ts").cast("timestamp").alias("as_of_ts")))
display(id_label_ts)

# COMMAND ----------

display(id_label_ts.groupBy('as_of_ts').count())
display(id_label_ts.count())

# COMMAND ----------

# MAGIC %md ### Specify feature lookup logic

# COMMAND ----------

# Specify features lookups
feature_lookups = [

  FeatureLookup(
    table_name=f"{catalog_schema_name}.demographic_features_ts",
    lookup_key="PassengerId",
    timestamp_lookup_key="as_of_ts"
  ),
  FeatureLookup(
    table_name=f"{catalog_schema_name}.ticket_features_ts",
    lookup_key="PassengerId",
    timestamp_lookup_key="as_of_ts"
  )
]

# COMMAND ----------

# MAGIC %md ### Join the right features for each observations 'as of' timestamp

# COMMAND ----------

point_in_time_training_set = fs.create_training_set(
                                id_label_ts,
                                feature_lookups=feature_lookups,
                                exclude_columns=[],
                                label="Survived",
                              )

point_in_time_training_df = point_in_time_training_set.load_df()

display(point_in_time_training_df)
display(point_in_time_training_df.groupBy(col('as_of_ts')).count())

# COMMAND ----------

"""
fs.drop_table(
  name='default.ticket_features_ts'
)
"""
