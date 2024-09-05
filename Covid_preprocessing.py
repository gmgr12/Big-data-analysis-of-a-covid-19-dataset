from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.conf import SparkConf
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

# Data Visualization and Profiling
# Create SparkSession

spark = SparkSession \
            .builder \
            .master("local[2]") \
            .appName("Example: Fetch from RDD") \
            .config("spark.sql.debug.maxToStringFields", 100) \
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
print(f"The total number of records is {covid_df.count()}, with each record corresponding to {len(covid_df.columns)} features(columns).")

# Set the maximum number of rows and columns to display
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

print(covid_df.describe().toPandas().transpose())

train_data, test_data = covid_df.randomSplit([0.75, 0.25], seed = 42)
test_data1 = test_data.select('total_cases', 'new_cases', 'total_deaths', 'new_deaths',\
                              'reproduction_rate', 'icu_patients', 'hosp_patients', 'total_tests', 'new_tests', 'positive_rate', \
                              'tests_units', 'total_vaccinations', 'people_vaccinated',\
                                'excess_mortality', 'population')

numeric_features = [t[0] for t in test_data1.dtypes if t[1] == 'int' or t[1] == 'double']
sampled_data = test_data1.select(numeric_features).sample(False, 0.8).toPandas()
axs = pd.plotting.scatter_matrix(sampled_data, figsize=(10, 10))
n = len(sampled_data.columns)
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())

plt.show()

# Show 5 records
print(covid_df.show(5, False))

# Data Preprocessing (Fixing NULL values)
# Get the data_type for all columns in column_name: Data_Type format
print("Data types of columns:")
for col_name, col_type in covid_df.dtypes:
    print(f"{col_name}: {col_type}")

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

# Get all the nulls in all the columns
for c in covid_df.columns :
    miss_vals = covid_df.select([F.count(F.when(F.isnull(c), c)).alias(c)])
    miss_dic = miss_vals.collect()[0].asDict()
    print(miss_dic)

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

#Final check on nulls
# Get all the nulls in all the columns
for c in covid_dff.columns :
    miss_vals = covid_dff.select([F.count(F.when(F.isnull(c), c)).alias(c)])
    miss_dic = miss_vals.collect()[0].asDict()
    print(miss_dic)
