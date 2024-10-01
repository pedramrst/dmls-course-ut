from pyspark.sql import SparkSession
from collections import defaultdict
from pyspark.sql.functions import concat, col, lit

path = "hdfs://raspberrypi-dml0:9000/rostami/City.txt"

if __name__ == '__main__':
    spark = SparkSession \
        .builder \
        .appName("Pedram app 1") \
        .getOrCreate()
    df = spark.read.text(path)
    data_collect = df.collect()
    cities = defaultdict(lambda: 0)
    for row in data_collect:
        line = row['value']
        cieties_in_line = line.split(',')
        for city in cieties_in_line:
            cities[city] += 1
    out_data = []
    for city in list(cities.keys()):
        out_data.append((city, cities[city]))
    out_df = spark.createDataFrame(out_data, ('city', 'repetition'))
    temp = out_df.select(concat(col("city"), lit(": "), col("repetition")))
    temp.coalesce(1).write.format("text").option("header", "false").mode("append").save("hdfs://raspberrypi-dml0:9000/rostami/Spark_rdd_1")