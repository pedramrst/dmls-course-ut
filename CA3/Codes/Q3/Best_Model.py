import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import RegressionMetrics


path = "hdfs://raspberrypi-dml0:9000/rostami/data.csv"
seed = 166

if __name__ == '__main__':
    spark = SparkSession \
        .builder \
        .appName(sys.argv[1] if len(sys.argv) > 1 else "Best Model") \
        .getOrCreate()
    data = spark.read.options(inferSchema='True',delimiter=',',  header=True).csv(path)
    data.cache()
    train_df, test_df = data.randomSplit([0.8, 0.2], seed = seed)
    stage_assembler = VectorAssembler(inputCols = ['feature1', 'feature2', 'feature3', 'feature4'], outputCol = 'features')
    stage_regressor = GBTRegressor(featuresCol='features', labelCol='label')
    regression_pipeline = Pipeline(stages= [stage_assembler, stage_regressor])
    model = regression_pipeline.fit(train_df)
    out_df = model.transform(test_df)
    out_df.show(5)
    out_data = out_df.collect()
    exp_pred = []
    for row in out_data:
        exp_pred.append((row['label'], row['prediction']))
    exp_pred = spark.sparkContext.parallelize(exp_pred)
    metrics = RegressionMetrics(exp_pred)
    print(f'RMSE: {metrics.rootMeanSquaredError}, r2: {metrics.r2}')