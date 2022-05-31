#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Python script to run benchmark on a query with a file path.
'''


# Import command line arguments and helper functions
import sys

import numpy as np
import pandas as pd

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, col
from pyspark.sql.functions import row_number
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics


def main(spark):

    top_100 = spark.read.csv('hdfs:/user/yy3754/movie_100_small.csv', header = True)
    preds = list(top_100.select('movieId').toPandas()['movieId'])
    val = spark.read.parquet('hdfs:/user/yy3754/validation_final_small.parquet')

    # Parquet files can also be used to create a temporary view and then used in SQL statements.
    val.createOrReplaceTempView("val")
    window = Window.partitionBy(val['userId']).orderBy(val['rating'].desc())
    new = val.select(col('*'), row_number().over(window).alias('row_number')).where(col('row_number') <= 100).toPandas().groupby('userId')['movieId'].apply(list)
    rank = []
    for i in range(len(new)):
        tup = (preds, new[i])
        rank.append(tup)
  
    predictionAndLabels = sparkContext.parallelize(rank)
    metrics = RankingMetrics(predictionAndLabels)
    print('meanAP:',metrics.meanAveragePrecisionAt(100))
    print('NDCG:',metrics.ndcgAt(100))
        
   




# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('temp_view').getOrCreate()
    sparkContext=spark.sparkContext

    main(spark)

