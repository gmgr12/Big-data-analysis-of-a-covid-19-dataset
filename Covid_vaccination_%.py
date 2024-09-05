from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff, to_date, lit, min
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
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
# Create new dataframe with record only between 2020-01-05, 2024-01-05
dates = ("2021-01-05", "2024-01-05")
covid_dfd = covid_dff.where(F.col('date').between(*dates))

USA_covid_df = covid_dfd.where(F.col('location') == "United States")

USA_covid_df.show(5)

USA_total_cases = USA_covid_df.agg(F.sum('new_cases')).collect()
print(USA_total_cases)

USA_people_vaccinated= USA_covid_df.agg(F.max('people_vaccinated')).collect()
print(USA_people_vaccinated)

USA_people_vaccinated_value = USA_people_vaccinated[0][0]

USA_population= 338289856.0
USA_people_vaccinated_percentage = USA_people_vaccinated_value / USA_population * 100 
USA_people_vaccinated_percentage

USA_total_left_population= 100 - USA_people_vaccinated_percentage
USA_total_left_population

import matplotlib.pyplot as plt
labels = 'US population without vaccines', 'US population vaccinated from 2021 till 2024, JAN',
sizes = [USA_total_left_population, USA_people_vaccinated_percentage]
explode = (0, 0.1)  
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.show()

Portugal_covid_df = covid_dfd.where(F.col('location') == "Portugal")

Portugal_covid_df.show(5)

Portugal_total_cases = Portugal_covid_df.agg(F.sum('new_cases')).collect()
print(Portugal_total_cases)

Portugal_people_vaccinated= Portugal_covid_df.agg(F.max('people_vaccinated')).collect()
print(USA_people_vaccinated)

Portugal_people_vaccinated_value = Portugal_people_vaccinated[0][0]

Portugal_population= 10270857.0
Portugal_people_vaccinated_percentage = Portugal_people_vaccinated_value / Portugal_population * 100 
Portugal_people_vaccinated_percentage

Portugal_total_left_population= 100 - Portugal_people_vaccinated_percentage
Portugal_total_left_population

import matplotlib.pyplot as plt
labels = 'Portugal population without vaccines', 'Portugal population vaccinated from 2021 till 2024, JAN',
sizes = [Portugal_total_left_population, Portugal_people_vaccinated_percentage]
explode = (0, 0.1)  
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.show()


