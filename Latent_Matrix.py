#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Import command line arguments and helper functions(if necessary)
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql import *
from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import  RankingMetrics
from time import time
from pyspark import SparkContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, col
import numpy as np
import pandas as pd
from pyspark.sql.functions import collect_list
from pyspark.ml.recommendation import ALS
import math

def roundrate(x):
    return round(0.5 * round(float(x)/0.5),1)
  
def main(spark):
    training = spark.read.parquet(f'hdfs:/user/yy3754/train_final_large.parquet', header = True)
    training.createOrReplaceTempView("training")
    
    train = training.select(col('userId').cast('integer'),col('movieId').cast('integer'), col('rating').cast('float'))
    
    als = ALS(rank = 150, maxIter=15, regParam=0.05, userCol="userId", itemCol="movieId", ratingCol="rating",nonnegative = True, coldStartStrategy="drop")
    model= als.fit(train)
    
    latent_items = model.itemFactors
    latent_items.createOrReplaceTempView("latent_items")
    latent_items = latent_items.toPandas()
    latent_items.columns = ['movieId','latent']
    
    latent_items_matrix = np.zeros([latent_items.shape[0], len(list(latent_items.latent[1]))])
    
    for i in range(0, len(latent_items.latent)):
        for j in range(0, len(latent_items.latent[i])):
            lat = list(latent_items.latent[i])[j]
            latent_items_matrix[i][j]=lat

    latent_items_matrix = pd.DataFrame(latent_items_matrix)
    latent_items_matrix['movieId'] = latent_items['movieId']
    train_pd = train.toPandas()
    movie_rating_dic = {i:[] for i in train_pd.movieId.unique()}
    for i, movie in enumerate(train_pd['movieId']):
        movie_rating_dic[movie] += [train_pd['rating'][i]]
    for key in movie_rating_dic:
        movie_rating_dic[key] = sum(movie_rating_dic[key]) / len(movie_rating_dic[key])

    movie_id = np.copy(latent_items_matrix['movieId'])
    latent_items_matrix['avg_rating'] = movie_id
    avg_rat = latent_items_matrix['avg_rating'].map(movie_rating_dic)
    latent_items_matrix['avg_rating'] = avg_rat 
    latent_items_matrix['round_rate'] = latent_items_matrix['avg_rating'].map(roundrate)
    latent_items_matrix = spark.createDataFrame(latent_items_matrix)
    latent_items_matrix.write.option('header','true').csv('hdfs:/user/yy3754/latent_items_matrix_large.csv')

if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('Latent_matrix').getOrCreate()

    main(spark)
    
