from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

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
# Hospitalized Patients in ICU and Normal beds on global basis
# From 2020-05-28 to 2020-08-28

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Create new dataframe with record only between 2020-05-28 and 2020-08-28
dates = ("2020-05-28", "2020-08-28")
covid_dfd = covid_dff.where(F.col('date').between(*dates))

# Create a list with only dates
dates_frame = covid_dfd.select("date").distinct().orderBy('date').collect()
dates_list = [str(dates_frame[x][0]) for x in range(len(dates_frame))]
print(dates_list)

covid_dt = covid_dfd.orderBy("date", ascending=True).groupBy("date")

# Create a list with sum of hosp_patients - with normal beds
hosps = covid_dt.agg(F.sum("hosp_patients")).collect()
hosps = [hosps[i][1] for i in range(len(hosps))]
print(hosps)

# Create a list with sum of icu patients
icus = covid_dt.agg(F.sum("icu_patients")).collect()
icus = [icus[i][1] for i in range(len(icus))]
print(icus)

sns.set(style = "darkgrid")

# replaced the year part of date for graph purpose
alt_dts_list = [dt.replace('2020-', '') for dt in dates_list]
print(alt_dts_list)
tick_marks = np.arange(len(alt_dts_list))

# Plot of Graph
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Shades of blue
colors = [(72/255, 99/255, 147/255), (129/255, 143/255, 163/255)]
labels = ['Hospital', 'ICUs']

for pat, col, label, ax in zip([hosps, icus], colors, labels, axes):
    ax.plot(alt_dts_list, pat, linestyle='solid', color=col)
    ax.set_xlabel("Date in between 2020-05-28 to 2020-08-28")
    ax.set_ylabel("Number of Patients")
    ax.set_title(f"Daily Number of Patients in {label}", fontsize=14)
    ax.set_xticks(tick_marks[::5])
    ax.set_xticklabels(alt_dts_list[::5], rotation=45)

plt.show()
matplotlib.rc_file_defaults()

# Hospitalized Patients in ICU and Normal beds on global basis
# From 2021-05-20 to 2021-08-28

# Create new dataframe with record only between 2020-05-28 and 2020-08-28
dates = ("2021-05-28", "2021-08-28")
covid_dfd = covid_dff.where(F.col('date').between(*dates))

# Create a list with only dates
dates_frame = covid_dfd.select("date").distinct().orderBy('date').collect()
dates_list = [str(dates_frame[x][0]) for x in range(len(dates_frame))]
print(dates_list)

covid_dt = covid_dfd.orderBy("date", ascending=True).groupBy("date")

# Create a list with sum of hosp_patients - with normal beds
hosps = covid_dt.agg(F.sum("hosp_patients")).collect()
hosps = [hosps[i][1] for i in range(len(hosps))]
print(hosps)

# Create a list with sum of icu patients
icus = covid_dt.agg(F.sum("icu_patients")).collect()
icus = [icus[i][1] for i in range(len(icus))]
print(icus)

sns.set(style = "darkgrid")

# replaced the year part of date for graph purpose
alt_dts_list = [dt.replace('2021-', '') for dt in dates_list]
print(alt_dts_list)
tick_marks = np.arange(len(alt_dts_list))

# Plot of Graph
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = [(72/255, 99/255, 147/255), (129/255, 143/255, 163/255)]
labels = ['Hospital', 'ICUs']

for pat, col, label, ax in zip([hosps, icus], colors, labels, axes):
    ax.plot(alt_dts_list, pat, linestyle='solid', color=col)
    ax.set_xlabel("Date in between 2021-05-28 to 2021-08-28")
    ax.set_ylabel("Number of Patients")
    ax.set_title(f"Daily Number of Patients in {label}", fontsize=14)
    ax.set_xticks(tick_marks[::5])
    ax.set_xticklabels(alt_dts_list[::5], rotation=45)

plt.show()
matplotlib.rc_file_defaults()