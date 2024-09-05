from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff, to_date, lit, min, mean
import pandas as pd
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
# Take the most recent date from covid data
recent_day = '2021-02-28'
covid_flt = covid_dff.filter(F.col('date') == recent_day)

# Calculate the mean of smokers, diabetes, cardio
covid_mean_fem_smokers = covid_flt.filter(F.col('female_smokers') != 0.0). \
                            select(F.mean(F.col('female_smokers'))).collect()[0][0]
covid_mean_male_smokers = covid_flt.filter(F.col('male_smokers') != 0.0). \
                            select(F.mean(F.col('male_smokers'))).collect()[0][0]
covid_mean_diabetes = covid_flt.filter(F.col('diabetes_prevalence') != 0.0). \
                            select(F.mean(F.col('diabetes_prevalence'))).collect()[0][0]
covid_mean_cardio = covid_flt.filter(F.col('cardiovasc_death_rate') != 0.0). \
                            select(F.mean(F.col('cardiovasc_death_rate'))).collect()[0][0]

# Print the values
print(f'Based on data up to {recent_day}, the mean of female smokers is {covid_mean_fem_smokers:.2f}.')
print(f'The mean of male smokers is {covid_mean_male_smokers:.2f}.')
print(f'The mean of people suffering from diabetes (aged 20-79) is {covid_mean_diabetes:.2f}.')
print(f'The mean number of deaths per 100,000 people due to cardiovascular conditions is {covid_mean_cardio:.2f}.')

covid_flt = covid_flt.filter(F.col('diabetes_prevalence') != 0.0).filter(F.col('cardiovasc_death_rate') != 0.0). \
                            filter(F.col('female_smokers') != 0.0).filter(F.col('male_smokers') != 0.0)
covid_flt.orderBy("excess_mortality_cumulative_per_million", ascending=False). \
                            select(["location", "excess_mortality_cumulative_per_million", "female_smokers", \
                                     "male_smokers", "diabetes_prevalence", "cardiovasc_death_rate"]).show(5)

covid_flt.orderBy("excess_mortality_cumulative_per_million"). \
                            select(["location", "excess_mortality_cumulative_per_million", "female_smokers", \
                                     "male_smokers", "diabetes_prevalence", "cardiovasc_death_rate"]).show(5)







