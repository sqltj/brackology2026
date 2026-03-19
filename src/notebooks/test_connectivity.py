# Databricks notebook source
# Test basic connectivity and capabilities

# COMMAND ----------

print("Cell 1: Basic Python works")
print(f"Spark version: {spark.version}")

# COMMAND ----------

# MAGIC %pip install requests
# MAGIC %restart_python

# COMMAND ----------

print("Cell 3: After restart_python")

# COMMAND ----------

import requests
try:
    resp = requests.get("https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams?limit=5&groups=50",
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    print(f"ESPN API status: {resp.status_code}")
    data = resp.json()
    teams = data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
    for t in teams:
        team = t.get("team", t)
        print(f"  {team.get('id')}: {team.get('displayName')}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# COMMAND ----------

from pyspark.sql import Row
from pyspark.sql.types import *

schema = StructType([StructField("id", IntegerType()), StructField("name", StringType())])
df = spark.createDataFrame([Row(id=1, name="test")], schema=schema)
df.write.mode("overwrite").saveAsTable("bracketology.raw.connectivity_test")
count = spark.table("bracketology.raw.connectivity_test").count()
print(f"Table write test: {count} rows")
spark.sql("DROP TABLE IF EXISTS bracketology.raw.connectivity_test")
print("Test passed!")
