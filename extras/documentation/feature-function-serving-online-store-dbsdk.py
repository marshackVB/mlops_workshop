# Databricks notebook source
# MAGIC %md # Feature Serving example notebook
# MAGIC
# MAGIC Feature Serving lets you serve pre-materialized features and run on-demand computation for features. 
# MAGIC
# MAGIC This notebook illustrates how to:
# MAGIC 1. Create a `FeatureSpec`. A `FeatureSpec` defines a set of features (prematerialized and on-demand) that are served together. 
# MAGIC 2. Set up the Databricks Online Feature Store to serve the features.
# MAGIC 3. Serve the features. To serve features, you create a Feature Serving endpoint with the `FeatureSpec`.
# MAGIC
# MAGIC ### Requirements
# MAGIC This notebook requires Databricks Runtime for Machine Learning 16.4 LTS ML or above.

# COMMAND ----------

# MAGIC %md ## Set up the Feature Table

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering>=0.13.0.1
# MAGIC %pip install mlflow>=3.8.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Specify the catalog and schema to use. You must have USE_CATALOG privilege on the catalog and USE_SCHEMA and CREATE_TABLE privileges on the schema.
# Change the catalog and schema here if necessary.

catalog_name = "main"
schema_name = "default"

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

feature_table_name = f"{catalog_name}.{schema_name}.location_features"
online_table_name = f"{catalog_name}.{schema_name}.location_features_online"
function_name = f"{catalog_name}.{schema_name}.distance"

# COMMAND ----------

# You must have `CREATE CATALOG` privileges on the catalog.
# If necessary, change the catalog and schema name here.
username = spark.sql("SELECT current_user()").first()["current_user()"]
username = username.split(".")[0]

# spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {catalog_name}.{schema_name}")
spark.sql(f"USE {catalog_name}.{schema_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up the Online Feature Store
# MAGIC
# MAGIC Create an online store and publish the table to it.
# MAGIC
# MAGIC For more details, see the Databricks documentation ([AWS](https://docs.databricks.com/aws/en/machine-learning/feature-store/online-feature-store) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/online-feature-store)). 

# COMMAND ----------

# Read in the dataset
destination_location_df = spark.read.option("inferSchema", "true").load("/databricks-datasets/travel_recommendations_realtime/raw_travel_data/fs-demo_destination-locations/", format="csv", header="true")

# Create the feature table
fe.create_table(
  name = feature_table_name,
  primary_keys="destination_id",
  df = destination_location_df,
  description = "Destination location features."
)

# COMMAND ----------

# Enable Change Data Feed to enable CONTINOUS and TRIGGERED publish modes

spark.sql(f"ALTER TABLE {feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = 'true')")

# COMMAND ----------

# Create an online store with specified capacity
online_store_name = f"{username}-online-store"

fe.create_online_store(
    name=online_store_name,
    capacity="CU_2"  # Valid options: "CU_1", "CU_2", "CU_4", "CU_8"
)

# COMMAND ----------

# Wait until the state is AVAILABLE
online_store = fe.get_online_store(name=online_store_name)
online_store.state

# COMMAND ----------

# Publish the table

published_table = fe.publish_table(
    online_store=online_store,
    source_table_name=feature_table_name,
    online_table_name=online_table_name
)

print(published_table)

# COMMAND ----------

# MAGIC %md ## Create the function

# COMMAND ----------

# MAGIC %md The next cell defines a function that calculates the distance between the destination and the user's current location.

# COMMAND ----------

# DBTITLE 1,Haversine Distance Calculator Function
# Define the function. This function calculates the distance between two locations. 
spark.sql(f"""
CREATE OR REPLACE FUNCTION {function_name}(latitude DOUBLE, longitude DOUBLE, user_latitude DOUBLE, user_longitude DOUBLE)
RETURNS DOUBLE
LANGUAGE PYTHON AS
$$
import math
lat1 = math.radians(latitude)
lon1 = math.radians(longitude)
lat2 = math.radians(user_latitude)
lon2 = math.radians(user_longitude)

# Earth's radius in kilometers
radius = 6371

# Haversine formula
dlat = lat2 - lat1
dlon = lon2 - lon1
a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
distance = radius * c

return distance
$$""")

# COMMAND ----------

# DBTITLE 1,Databricks Travel Feature Spec Generator
from databricks.feature_engineering import FeatureLookup, FeatureFunction

features=[
  FeatureLookup(
    table_name=feature_table_name,
    lookup_key="destination_id"
  ),
  FeatureFunction(
    udf_name=function_name, 
    output_name="distance",
    input_bindings={
      "latitude": "latitude", 
      "longitude": "longitude", 
      "user_latitude": "user_latitude", 
      "user_longitude": "user_longitude"
    },
  ),
]

feature_spec_name = f"{catalog_name}.{schema_name}.travel_spec"
try: 
  fe.create_feature_spec(name=feature_spec_name, features=features, exclude_columns=None)
except Exception as e:
  if "already exists" in str(e):
    pass
  else:
    raise e

# COMMAND ----------

# MAGIC %md You can now view the `FeatureSpec` (`travel_spec`) and the distance function (`distance`) in Catalog Explorer. Click **Catalog** in the sidebar. In the Catalog Explorer, navigate to your schema in the **main** catalog. The `FeatureSpec` and the function appear under **Functions**. 
# MAGIC
# MAGIC <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/catalog-explorer.png"/>

# COMMAND ----------

# MAGIC %md ## Create a Feature Serving endpoint

# COMMAND ----------

# DBTITLE 1,Databricks Endpoint Creation Script
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

# Create endpoint
endpoint_name = "fse-location"

workspace = WorkspaceClient()

try:
  status = workspace.serving_endpoints.create_and_wait(
    name=endpoint_name,
    config = EndpointCoreConfigInput(
      name=endpoint_name,
      served_entities=[
      ServedEntityInput(
        entity_name=feature_spec_name,
        scale_to_zero_enabled=True,
        workload_size="Small"
      )
      ]
    )
  )
  print(status)
except Exception as e:
  if "already exists" in str(e):
    print(f"Not creating endpoint {endpoint_name} since it already exists.")
  else:
    raise e


# COMMAND ----------

# Get the status of the endpoint
status = workspace.serving_endpoints.get(name=endpoint_name)
print(status)

# COMMAND ----------

# MAGIC %md You can now view the status of the Feature Serving Endpoint in the table on the **Serving endpoints** page. Click **Serving** in the sidebar to display the page.

# COMMAND ----------

# MAGIC %md ## Query

# COMMAND ----------

# DBTITLE 1,MLflow Databricks Prediction Client
import mlflow.deployments
from pprint import pprint

client = mlflow.deployments.get_deploy_client("databricks")
response = client.predict(
    endpoint=endpoint_name,
    inputs={
        "dataframe_records": [
            {"destination_id": 1, "user_latitude": 37, "user_longitude": -122},
            {"destination_id": 2, "user_latitude": 37, "user_longitude": -122},
        ]
    },
)

pprint(response)

# COMMAND ----------

# MAGIC %md ## Clean up
# MAGIC
# MAGIC When you are finished, delete the `FeatureSpec`, feature endpoint, online table, and online store. 

# COMMAND ----------

# Delete the FeatureSpec
# fe.delete_feature_spec(name=feature_spec_name)

# COMMAND ----------

# Delete the feature endpoint
# workspace.serving_endpoints.delete(name=endpoint_name)

# COMMAND ----------

# Delete the online table
# workspace.feature_store.delete_online_table(online_table_name=online_table_name)

# COMMAND ----------

# Delete the online store
# fe.delete_online_store(name=online_store_name)
