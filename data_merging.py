#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Python script to run benchmark on a query with a file path.
Usage:
    $ spark-submit pq_avg_income.py <file_path>
'''

import getpass
# Import command line arguments and helper functions
import sys
import numpy as np

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import row_number


def data_merging(spark, netID):
    '''
    Parameters
    ----------
    spark : spark session object

    netID
    '''

    # Read data
    movies = spark.read.parquet(f'hdfs:/user/{netID}/movies_small.parquet', header = True)
    tags = spark.read.parquet(f'hdfs:/user/{netID}/tags_small.parquet', header = True)
    ratings = spark.read.parquet(f'hdfs:/user/{netID}/ratings_small.parquet', header = True)
    links = spark.read.parquet (f'hdfs:/user/{netID}/links_small.parquet', header = True)
    movies.createOrReplaceTempView("movies")
    tags.createOrReplaceTempView("tags")
    ratings.createOrReplaceTempView("ratings")
    links.createOrReplaceTempView("links")
    
    # remove rows where rating = 0
    ratings_filtered = spark.sql('SELECT* FROM ratings where rating != 0')
    ratings_filtered.createOrReplaceTempView("ratings_filtered")
    
    # merge data from ratings, tags, movies and links
    data_merged_all = spark.sql(
    """SELECT ratings_filtered.userId, ratings_filtered.movieId, ratings_filtered.rating, tags.tag, ratings_filtered.timestamp, movies.title, movies.genres, links.imdbId, links.tmdbId
       FROM ratings_filtered 
       LEFT JOIN tags 
       ON ratings_filtered.movieId = tags.movieId and ratings_filtered.userId = tags.userId
       LEFT join movies
       ON ratings_filtered.movieId = movies.movieId
       LEFT join links
       ON ratings_filtered.movieId = links.movieId""")
    data_merged_all.createOrReplaceTempView("data_merged_all")
    
    # extract userId who have rated at least 20 movies 
    userId_df = spark.sql(
    """SELECT userId, count(distinct movieId) as counts
       FROM data_merged_all 
       GROUP BY userId 
       HAVING counts>= 20""").drop('counts')
    userId_df.createOrReplaceTempView('userId_df')
    
    # random split userId for training, validation, testing
    train_user, validation_user, test_user = userId_df.randomSplit([0.6,0.2,0.2],seed = 2022)
    train_user.createOrReplaceTempView('train_user')
    validation_user.createOrReplaceTempView('validation_user')
    test_user.createOrReplaceTempView('test_user')
    
    # prepare training data, validation data, and testing data by splited userId
    train_data_part_1 = spark.sql('SELECT * FROM data_merged_all WHERE userId IN (SELECT userId FROM train_user)')
    validation_data = spark.sql('SELECT * FROM data_merged_all WHERE userId IN (SELECT userId FROM validation_user)')
    test_data = spark.sql('SELECT * FROM data_merged_all WHERE userId IN (SELECT userId FROM test_user)')
    
    # add index for test validation data and test data
    validation_idx = validation_data.select("*", F.row_number().over(Window.partitionBy().orderBy(validation_data['userId'])).alias("row_num"))
    test_idx = test_data.select("*", F.row_number().over(Window.partitionBy().orderBy(test_data['userId'])).alias("row_num"))
    test_idx.createOrReplaceTempView('test_idx')
    validation_idx.createOrReplaceTempView('validation_idx')
    
    # split validation data and test data
    test_group_1 = spark.sql('SELECT * FROM test_idx WHERE row_num%2 = 0').drop('row_num')
    test_group_2 = spark.sql('SELECT * FROM test_idx WHERE row_num%2 =1').drop('row_num')
    validation_group_1 = spark.sql('SELECT * FROM validation_idx WHERE row_num % 2 = 0').drop('row_num')
    validation_group_2 = spark.sql('SELECT * FROM validation_idx WHERE row_num % 2 = 1').drop('row_num')
     
    train_data_part_1.createOrReplaceTempView('train_data_part_1')
    validation_group_1.createOrReplaceTempView('validation_group_1')
    validation_group_2.createOrReplaceTempView('validation_group_2')
    test_group_1.createOrReplaceTempView('test_group_1')
    test_group_2.createOrReplaceTempView('test_group_2')
    
    # prepare final train, validation, test data
    train_final = spark.sql('SELECT * FROM train_data_part_1 UNION ALL SELECT * FROM validation_group_1 UNION ALL SELECT * FROM test_group_1')
    validation_final = validation_group_2
    test_final = test_group_2
    
    return train_final, validation_final, test_final
   


def main(spark,netID ):
    '''
    Main routine for Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID
    '''
    
    train_final, validation_final, test_final = data_merging (spark, netID)
    train_final.write.parquet('hdfs:/user/yy3754/train_final_small.parquet')
    validation_final.write.parquet('hdfs:/user/yy3754/validation_final_small.parquet')
    test_final.write.parquet('hdfs:/user/yy3754/test_final_small.parquet')



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('data_merge_small').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    main(spark, netID)