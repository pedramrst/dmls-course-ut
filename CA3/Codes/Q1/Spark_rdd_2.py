from pyspark.sql import SparkSession
from pyspark.sql.functions import concat, col, lit

path = "hdfs://raspberrypi-dml0:9000/rostami/City.txt"

if __name__ == '__main__':
    spark = SparkSession \
        .builder \
        .appName("Spark rdd 2") \
        .getOrCreate()
    df = spark.read.text(path)
    data_collect = df.collect()
    all_cities = []
    for row in data_collect:
        line = row['value']
        cities = line.split(',')
        cities.sort()
        all_cities.append(cities)
    out_data = []
    for cities in all_cities:
        out_data.append((','.join(cities), ))
    out_df = spark.createDataFrame(out_data, ('cities', ))
    temp = out_df.select(concat(col("cities")))
    temp.coalesce(1).write.format("text").option("header", "false").mode("append").save("hdfs://raspberrypi-dml0:9000/rostami/Spark_rdd_2")