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

# Fix the tests_units null values
covid_df.select("tests_units").distinct().show()
covid_dft = covid_dfc.fillna({'tests_units':'no_info'})
covid_dft.select("tests_units").distinct().show()

# Fix all double fields with null values covid_dff is final data frame
covid_dff = covid_dft.fillna(0)
"""
# Final check on nulls
# Get all the nulls count in all the columns
for c in covid_dff.columns :
    miss_vals = covid_dff.select([F.count(F.when(F.isnull(c), c)).alias(c)])
    miss_dic = miss_vals.collect()[0].asDict()
    print(miss_dic)
"""
# Linear Regression start

import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

# Create new dataframe with record only between 2020-01-05, 2024-01-05
dates = ("2020-01-05", "2024-01-05")
covid_dfd = covid_dff.where(F.col('date').between(*dates))

# Get the year, month, day from date
covid_df_train = covid_dfd.withColumn('year', F.year('date')) \
                             .withColumn('month', F.month('date')) \
                             .withColumn('day', F.day('date'))
covid_df_train.show(5)

# Select relevant features
selected_data1 = covid_df_train.select('year', 'month', 'day', 'total_cases', 'new_cases_smoothed', 'total_deaths', \
                                       'people_vaccinated', 'total_boosters', 'new_vaccinations_smoothed', \
                                        'people_fully_vaccinated', 'population', \
                                        'new_vaccinations', 'total_vaccinations')

selected_data1.show(20)

# Assemble features into a vector
assembler = VectorAssembler(inputCols=['year', 'month', 'day', 'total_cases', 'new_cases_smoothed', 'total_deaths', \
                                       'people_vaccinated', 'total_boosters', 'new_vaccinations_smoothed', \
                                        'people_fully_vaccinated', 'population', \
                                       'new_vaccinations'], outputCol='features')
output = assembler.transform(selected_data1)
output.show(10)

# Randomly split the data to test and train
train_data, test_data = output.randomSplit([0.75, 0.25], seed = 42)
train_data.show(10)
test_data.show(10)

# Create a Linear Regression model
lr = LinearRegression(featuresCol='features', labelCol='total_vaccinations')

# Fit the model on the training data
lr_model = lr.fit(train_data)

# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(lr_model.coefficients))
print("Intercept: %s" % str(lr_model.intercept))

pred_result = lr_model.transform(test_data)

pred_result.select('features', 'total_vaccinations', 'prediction')

# Summarize the model over the test set and print out some metrics
testSummary = lr_model.summary

print("RMSE: %f" % testSummary.rootMeanSquaredError)
print("MAE: %f" % testSummary.meanAbsoluteError)
print("MSE: %f" % testSummary.meanSquaredError)

pred_evaluator = RegressionEvaluator(predictionCol = 'prediction', labelCol = "total_vaccinations", metricName = "r2")
print("R2 on test data for linear regression= ", pred_evaluator.evaluate(pred_result))

# Get the sample of data not including total_vaccinations = 0.0
filtered_result = pred_result.filter(pred_result['total_vaccinations'] != 0.0)

# Selecting columns and formatting numbers
filtered_result.select(
    'features',
    F.format_number('total_vaccinations', 2).alias('total_vaccinations'),
    F.format_number('prediction', 2).alias('prediction')
).sample(False,0.1).show(15, truncate=False)





