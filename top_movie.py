#!/usr/bin/env python
# -*- coding: utf-8 -*-

import getpass
import sys
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
from pyspark.sql.functions import col


def weighted_calculate(user_count, min_rating, avg_each, avg_all):
    return((user_count/(user_count+min_rating))*avg_each)+((min_rating/(user_count+min_rating))*avg_all)

def main(spark):

    training = spark.read.parquet('hdfs:/user/yy3754/train_large.parquet', header = True)
    #validation = spark.read.parquet('hdfs:/user/yy3754/validation_large.parquet', header = True)
    #testing = spark.read.parquet('hdfs:/user/yy3754/test_final_large.parquet', header = True)

    training.createOrReplaceTempView("training")
    #validation.createOrReplaceTempView("validation")
    #testing.createOrReplaceTempView("testing")

    train_df = training.select('movieId','userId','rating').toPandas()

    count_avg = (
        train_df
        .groupby('movieId')
        .agg({'userId':'count', 'rating':'mean'})
        )
    count_avg.columns = ['user_count', 'rating_avg']

    avg_all = np.mean(count_avg['rating_avg']) 
    min_rating = np.percentile(count_avg['user_count'], 70) 
    count_avg = count_avg[count_avg['user_count'] >= m]
    avg_each = count_avg['rating_avg'] 
    user_count = count_avg['user_count'] 
    count_avg['weighted_rating'] = weighted_calculate(user_count, min_rating, avg_each, avg_all)
    rating_100 = count_avg.sort_values('weighted_rating',ascending = False ).head(100).drop(columns=['user_count', 'rating_avg'])
    rating_100 = spark.createDataFrame(rating_100)
    rating_100.write.parquet('hdfs:/user/yy3754/movie_100_large.parquet')


    
if __name__ == "__main__":
    spark = SparkSession.builder.appName('popularity').getOrCreate()
    
    main(spark)
