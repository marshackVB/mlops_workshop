# Databricks notebook source
# MAGIC %md ## Create example datasets

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, DoubleType, IntegerType, StringType
import pandas as pd

# COMMAND ----------

dbutils.widgets.text('catalog_name','','Enter catalog name')
dbutils.widgets.text('schema_name','','Enter schema name')

# COMMAND ----------

catalog_name = dbutils.widgets.get('catalog_name')
schema_name = dbutils.widgets.get('schema_name')
catalog_schema_name = f"{catalog_name}.{schema_name}"
print(catalog_schema_name)

# COMMAND ----------

# MAGIC %md Create Delta tables from csv files

# COMMAND ----------

path = "/Workspace/Users/marshall.carter@databricks.com/mlops_workshop/02_feature_store/data"

# Enter your dbfs locations for each data file
file_locations = {'ticket': f'{path}/passenger_ticket.csv',
                  'demographic': f'{path}/passenger_demographic.csv',
                  'labels': f'{path}/passenger_labels.csv'}

dtypes={
        "PassengerId": "string",
        "Ticket": "string",
        "Cabin": "string",
        "Embarked": "string",
        "Pclass": "string",
        "Parch": "string",
        "SibSp": "string",
        "Survived": "Int64",   
        "Fare": "float64",
        "Age": "float64",
    }

# Create Spark DataFrame schemas
passenger_ticket_types = [('PassengerId',     StringType()),
                          ('Ticket',          StringType()),
                          ('Fare',            DoubleType()),
                          ('Cabin',           StringType()),
                          ('Embarked',        StringType()),
                          ('Pclass',          StringType()),
                          ('Parch',           StringType())]

passenger_demographic_types = [('PassengerId',StringType()),
                                ('Name',       StringType()),
                                ('Sex',        StringType()),
                                ('Age',        DoubleType()),
                                ('SibSp',      StringType())]

passenger_label_types = [('PassengerId',StringType()),
                          ('Survived',   IntegerType())]


def create_schema(col_types):
  struct = StructType()
  for col_name, type in col_types:
    struct.add(col_name, type)
  return struct
  
passenger_ticket_schema =      create_schema(passenger_ticket_types)
passenger_dempgraphic_schema = create_schema(passenger_demographic_types)
passenger_label_schema =       create_schema(passenger_label_types)
  
  
def create_pd_dataframe(csv_file_path, schema):
  df = pd.read_csv(csv_file_path)
  cols = list(df.columns)
  schema = {c: dtypes[c] for c in cols if c in dtypes}
  df = pd.read_csv(csv_file_path, dtype=schema)
  return df

passenger_demographic_features = create_pd_dataframe(file_locations['demographic'], passenger_dempgraphic_schema)

# COMMAND ----------

passenger_demographic_features.head()

# COMMAND ----------

path = "/Workspace/Users/marshall.carter@databricks.com/mlops_workshop/02_feature_store/data"

# Enter your dbfs locations for each data file
file_locations = {'ticket': f'{path}/passenger_ticket.csv',
                  'demographic': f'{path}/passenger_demographic.csv',
                  'labels': f'{path}/passenger_labels.csv'}

dtypes={
        "PassengerId": "string",
        "Ticket": "string",
        "Cabin": "string",
        "Embarked": "string",
        "Pclass": "string",
        "Parch": "string",
        "SibSp": "string",
        "Survived": "Int64",   
        "Fare": "float64",
        "Age": "float64",
    }

def create_tables(file_locations=file_locations):

  # Create Spark DataFrame schemas
  passenger_ticket_types = [('PassengerId',     StringType()),
                            ('Ticket',          StringType()),
                            ('Fare',            DoubleType()),
                            ('Cabin',           StringType()),
                            ('Embarked',        StringType()),
                            ('Pclass',          StringType()),
                            ('Parch',           StringType())]

  passenger_demographic_types = [('PassengerId',StringType()),
                                 ('Name',       StringType()),
                                 ('Sex',        StringType()),
                                 ('Age',        DoubleType()),
                                 ('SibSp',      StringType())]

  passenger_label_types = [('PassengerId',StringType()),
                           ('Survived',   IntegerType())]
  
  
  def create_schema(col_types):
    struct = StructType()
    for col_name, type in col_types:
      struct.add(col_name, type)
    return struct
  
  passenger_ticket_schema =      create_schema(passenger_ticket_types)
  passenger_dempgraphic_schema = create_schema(passenger_demographic_types)
  passenger_label_schema =       create_schema(passenger_label_types)
  
  
  def create_pd_dataframe(csv_file_path, schema):
    df = pd.read_csv(csv_file_path)
    cols = list(df.columns)
    pandas_schema = {c: dtypes[c] for c in cols if c in dtypes}
    df = pd.read_csv(csv_file_path, dtype=pandas_schema)
    return spark.createDataFrame(df, schema = schema)
  
  passenger_ticket_features =      create_pd_dataframe(file_locations['ticket'], passenger_ticket_schema)
  passenger_demographic_features = create_pd_dataframe(file_locations['demographic'], passenger_dempgraphic_schema)
  passenger_labels =               create_pd_dataframe(file_locations['labels'], passenger_label_schema)
  
  
  def write_to_delta(spark_df, delta_table_name):
    spark_df.write.mode('overwrite').format('delta').saveAsTable(delta_table_name)
    
  delta_tables = {"ticket":       f"{catalog_schema_name}.passenger_ticket_feautures",
                  "demographic":  f"{catalog_schema_name}.passenger_demographic_features",
                  "labels":       f"{catalog_schema_name}.passenger_labels"}
    
  write_to_delta(passenger_ticket_features,      delta_tables['ticket'])
  write_to_delta(passenger_demographic_features, delta_tables['demographic'])
  write_to_delta(passenger_labels,               delta_tables['labels'])
  
  
  out = f"""The following tables were created:
          - {delta_tables['ticket']}
          - {delta_tables['demographic']}
          - {delta_tables['labels']}
       """
  
  print(out)

# COMMAND ----------

create_tables()

# COMMAND ----------

# MAGIC %md To drop tables

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC --DROP TABLE IF EXISTS shared.mlc_schema.passenger_ticket_feautures;
# MAGIC --DROP TABLE IF EXISTS shared.mlc_schema.passenger_demographic_features;
# MAGIC --DROP TABLE IF EXISTS shared.mlc_schema.passenger_labels;
