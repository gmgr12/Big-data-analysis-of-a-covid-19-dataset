from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff, to_date, lit, min, mean
import plotly.express as px
from pyspark.ml.stat import Correlation
from pyspark.ml.evaluation import RegressionEvaluator

# Data Visualization and Profiling
# Create SparkSession
spark = SparkSession \
            .builder \
            .master("local[2]") \
            .appName("Example: Fetch from RDD") \
            .config("spark.executor.memory", "1g") \
            .config("spark.driver.memory", "1g") \
            .config("spark.cores.max", "2") \
            .getOrCreate()

# Read the covid csv file
covid_df = spark.read.format("csv") \
            .option("header","true").option("inferSchema","true") \
            .load("/mnt/c/Users/pooja/Documents/Masters/Introduction_to_Big_Data(EGD)/Project/owid-covid-data.csv")

# Print the schema for covid dataset
covid_df.printSchema()

# Print the total number of rows and columns in dataset
print(f"The total number of samples is {covid_df.count()}, with each sample corresponding to {len(covid_df.columns)} features.")

# Show 5 records
print(covid_df.show(5, False))

"""
# Data Preprocessing (Fixing NULL values)
# Get the data_type for all columns in column_name: Data_Type format
print("Data types of columns:")
for col_name, col_type in covid_df.dtypes:
    print(f"{col_name}: {col_type}")
"""

# Get the data_type of all columns only Data_Type
column_types = [col_type for col_name, col_type in covid_df.dtypes]
print(column_types)

# Count the number of double, string and date columns
num_double_columns = column_types.count('double')
num_string_columns = column_types.count('string')
num_date_columns = column_types.count('date')

# Print the counts
print(f"Number of double columns: {num_double_columns}")
print(f"Number of string columns: {num_string_columns}")
print(f"Number of date columns: {num_date_columns}")

# Show 5 records of all string fields 
covid_df.select("iso_code","location","continent","date","tests_units").show(5)

"""
# Get all the nulls count in all the columns
for c in covid_df.columns :
    miss_vals = covid_df.select([F.count(F.when(F.isnull(c), c)).alias(c)])
    miss_dic = miss_vals.collect()[0].asDict()
    print(miss_dic)
"""

# Filling all NULL values
# Fix the continent null values

covid_df.select("continent").distinct().show()
covid_dfc = covid_df.fillna({'continent':'OWID'})
covid_dfc.select("continent").distinct().show()

#Fix the tests_units null values
covid_df.select("tests_units").distinct().show()
covid_dft = covid_dfc.fillna({'tests_units':'no_info'})
covid_dft.select("tests_units").distinct().show()

#Fix all double fields with null values covid_dff is final data frame
covid_dff = covid_dft.fillna(0)

"""
#Final check on nulls
# Get all the nulls count in all the columns
for c in covid_dff.columns :
    miss_vals = covid_dff.select([F.count(F.when(F.isnull(c), c)).alias(c)])
    miss_dic = miss_vals.collect()[0].asDict()
    print(miss_dic)
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Create new dataframe with records only between 2020-08-05 and 2020-09-05
dates = ("2020-07-03", "2022-08-05")
covid_dfd = covid_dff.where(F.col('date').between(*dates))

owid_covid_df = covid_dfd.where(F.col('continent') != "OWID")

# Select relevant features
selected_data1 = owid_covid_df.select('reproduction_rate', 'continent')
selected_data1.show(5)

# Convert Spark DataFrame to Pandas DataFrame
df = selected_data1.toPandas()


# Plotting the smoothed data
plt.figure(figsize=(10, 6))
ax = sns.barplot(data = df, y= 'reproduction_rate' , x = 'continent', ci=None, palette='muted', width=0.5)
plt.xlabel('continent')
plt.ylabel('Reproduction Rate')
plt.title('Reproduction Rate from 2020-07-03 to 2022-08-05')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
