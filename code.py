from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import IntegerType, DoubleType, FloatType, StringType
import pyspark.sql.functions as F
from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql.functions import col
import warnings
warnings.filterwarnings('ignore')

# Create a Spark session
spark = SparkSession.builder.appName('BankDepositClassification').getOrCreate()

df = spark.read.csv('/Users/imran/Documents/Folder1/XYZ_Bank_Deposit_Data_Classification.csv', header=True, sep=';',inferSchema=True)
df.show(5)

for column in df.columns:
    df = df.withColumnRenamed(column, column.replace('.', '_'))

# Exploratory Data Analysis (EDA) Findings


categorical_columns = [field.name for field in df.schema.fields if isinstance(field.dataType, StringType)]
numerical_columns = [field.name for field in df.schema.fields if isinstance(field.dataType,
(IntegerType, DoubleType, FloatType))]


print("Numeric Features:", numerical_columns)
print("Categorical Features:", categorical_columns)
print("Total numeric features:", len(numerical_columns))
print("Total categorical features:", len(categorical_columns))

# Null values for the dataset
null_values = {column: df.filter(df[column].isNull()).count() for column in df.columns}
print("Null Values:", null_values)

# Cardinality of all variables
cardinality = {column: df.select(column).distinct().count() for column in categorical_columns}
print("Cardinality:", cardinality)


df.describe().show()

import matplotlib.pyplot as plt
import seaborn as sns
import math

# Determine the number of rows needed
num_plots = len(numerical_columns)
num_cols = 3
num_rows = math.ceil(num_plots / num_cols)

plt.figure(figsize=(10, 3 * num_rows))

# For each numerical feature
for i, col_name in enumerate(numerical_columns, 1):
    plt.subplot(num_rows, num_cols, i)
    result = df.groupBy('y').agg({col_name: 'mean'}).collect()
    
    categories = [row['y'] for row in result]
    means = [row[f'avg({col_name})'] for row in result]

    sns.barplot(x=categories, y=means)
    plt.xlabel('Category of y', fontsize=8)
    plt.ylabel(f'Mean of {col_name}', fontsize=8)
    plt.title(f'Mean of {col_name} by Category of y', fontsize=10)

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt


max_plots = 15
selected_categorical_columns = [col for col in df.columns if col not in numerical_columns and col != 'y'][:max_plots]

# Define the layout of the subplots
num_cols = 3  # Number of columns in the subplot grid
num_rows = math.ceil(len(selected_categorical_columns) / num_cols)  # Number of rows needed

plt.figure(figsize=(10, 3 * num_rows))

# For each selected categorical feature
for i, col_name in enumerate(selected_categorical_columns, 1):
    plt.subplot(num_rows, num_cols, i)
    crosstab_result = df.stat.crosstab(col_name, 'y').collect()
    
    categories = [row[f'{col_name}_y'] for row in crosstab_result]
    counts_yes = [row['yes'] for row in crosstab_result]
    counts_no = [row['no'] for row in crosstab_result]

    plt.bar(categories, counts_yes, label='Yes', color='skyblue')
    plt.bar(categories, counts_no, bottom=counts_yes, label='No', color='salmon')
    plt.xlabel(col_name, fontsize=8)
    plt.ylabel('Counts')
    plt.title(f'Crosstab of {col_name} and y', fontsize=10)
    plt.legend()
    plt.xticks(rotation=45, fontsize=6)

plt.tight_layout()
plt.show()

# List to store variables to be excluded
exclude_columns = []

# Calculate pairwise correlations and identify high correlations
for i in range(len(numerical_columns)):
    for j in range(i+1, len(numerical_columns)):
        col1 = numerical_columns[i]
        col2 = numerical_columns[j]
        correlation = df.select(F.corr(col(col1), col(col2)).alias('correlation')).collect()[0]['correlation']
        
        # Check if correlation is higher than the threshold
        if abs(correlation) > 0.8:
            print(f"High correlation ({correlation}) between {col1} and {col2}")
            # Add to exclude list (here we choose col2, but you can choose based on criteria)
            exclude_columns.append(col2)

# Remove duplicates from the exclude list
exclude_columns = list(set(exclude_columns))

print(f"Columns to drop: {exclude_columns}")

# Updated columns list
final_columns = [col for col in df.columns if col not in exclude_columns]

# New DataFrame with excluded columns
df = df.select(*final_columns)

# Create a new binary column indicating whether the client was previously contacted
df = df.withColumn('previously_contacted', F.when(df['pdays'] == 999, 0).otherwise(1))

# replace value
df = df.withColumn('pdays', F.when(df['pdays'] == 999, F.lit(-1)).otherwise(df['pdays']))

categorical_columns = [field.name for field in df.schema.fields if isinstance(field.dataType, StringType)]
numerical_columns = [field.name for field in df.schema.fields if isinstance(field.dataType, (IntegerType, DoubleType, FloatType))]


# Target column
target_column = 'y'

# Convert target column to numeric
label_indexer = StringIndexer(inputCol=target_column, outputCol="label").fit(df)
df = label_indexer.transform(df)

df = df.drop('y')

# Remove target column from numerical columns if present
if target_column in categorical_columns:
    categorical_columns.remove(target_column)

# Process categorical columns
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in categorical_columns]
encoder = OneHotEncoder(inputCols=[indexer.getOutputCol() for indexer in indexers], outputCols=[col+"_encoded" for col in categorical_columns])

# Assemble features
assembled_inputs = [c+"_encoded" for c in categorical_columns] + numerical_columns
assembler = VectorAssembler(inputCols=assembled_inputs, outputCol='features')

from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import col


model_performance = []

# Model building
train, test = df.randomSplit([0.7, 0.3], seed=42)
# Define different classifiers
classifiers = [
    LogisticRegression(featuresCol='features', labelCol='label'),
    RandomForestClassifier(featuresCol='features', labelCol='label'),
    GBTClassifier(featuresCol='features', labelCol='label'),
    DecisionTreeClassifier(featuresCol='features', labelCol='label')
]

for classifier in classifiers:

    # Create Pipeline
    pipeline = Pipeline(stages=indexers + [encoder, assembler, classifier])

    # Train model
    model = pipeline.fit(train)

    # Make predictions
    predictions = model.transform(test)

    
    # Evaluate model
    evaluator = BinaryClassificationEvaluator(labelCol='label', metricName="accuracy")
    
    auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})

    print(f"AUC of the {classifier.__class__.__name__} is {auc:.3f}")

    model_performance.append((classifier.__class__.__name__, auc))


    # Confusion Matrix
    preds_and_labels = predictions.select(['prediction', 'label']).withColumn('label', col('label').cast(FloatType()))
    preds_and_labels = preds_and_labels.select(['prediction', 'label'])
    metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))

    print(f'Confusion Matrix for {classifier.__class__.__name__}:')
    print(metrics.confusionMatrix().toArray())
    print('\n')


# Sort models based on AUC and then runtime
model_performance.sort(key=lambda x: (-x[1]))

# Champion Model
champion_model_name, champion_model_auc = model_performance[0]
print(f"Champion Model: {champion_model_name} with AUC: {champion_model_auc} ")    



champion_model = GBTClassifier(featuresCol='features', labelCol='label')


# Create Pipeline for the champion model
champion_pipeline = Pipeline(stages=indexers + [encoder, assembler, champion_model])

# Train the champion model on the entire dataset
champion_model_trained = champion_pipeline.fit(df)

# Save the champion model
model_path = "/Users/imran/Documents/Folder1/saved_model"
champion_model_trained.save(model_path)

from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import ClusteringEvaluator

# Create a Spark session
spark = SparkSession.builder.appName('KMeansClustering').getOrCreate()

df = spark.read.csv('/Users/imran/Documents/Folder1/XYZ_Bank_Deposit_Data_Classification.csv', header=True, sep=';', inferSchema=True)

for column in df.columns:
    df = df.withColumnRenamed(column, column.replace('.', '_'))
    
# Identify categorical and numerical columns
categorical_columns = [field.name for field in df.schema.fields if isinstance(field.dataType, StringType)]
numerical_columns = [field.name for field in df.schema.fields if isinstance(field.dataType, (IntegerType, DoubleType, FloatType))]

# Target column
target_column = 'y'

# Convert target column to numeric
label_indexer = StringIndexer(inputCol=target_column, outputCol="label").fit(df)
df = label_indexer.transform(df)

# Remove target column from numerical columns if present
if target_column in numerical_columns:
    numerical_columns.remove(target_column)

# Label encoding for categorical columns
for categorical_col in categorical_columns:
    indexer = StringIndexer(inputCol=categorical_col, outputCol=categorical_col + "_index")
    df = indexer.fit(df).transform(df)

# Updated list of features for VectorAssembler
updated_feature_cols = [c + "_index" for c in categorical_columns] + numerical_columns

# Assemble features
assembler = VectorAssembler(inputCols=updated_feature_cols, outputCol="features")
df_vect = assembler.transform(df)

# Scale the features
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
df_scaled = scaler.fit(df_vect).transform(df_vect)

# Determine optimal number of clusters (Elbow Method)
cost = []
for k in range(2, 10):
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("scaledFeatures")
    model = kmeans.fit(df_scaled)
    predictions = model.transform(df_scaled)
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictions)
    cost.append(silhouette)

# Plot the Elbow Plot
plt.plot(range(2, 10), cost)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Elbow Method')
plt.show()

optimal_k = 4
final_kmeans = KMeans().setK(optimal_k).setSeed(1).setFeaturesCol("scaledFeatures")
final_model = final_kmeans.fit(df_scaled)
predictions = final_model.transform(df_scaled)
from pyspark.sql import functions as F

# List to hold all aggregation expressions
aggregations = []

# Create aggregation expressions for each numerical column
for col in numerical_columns:
    aggregations.append(F.mean(col).alias(col + '_mean'))
    aggregations.append(F.stddev(col).alias(col + '_stddev'))

# Perform all aggregations in a single operation
cluster_summary = predictions.groupBy('prediction').agg(*aggregations)
cluster_summary.show()

