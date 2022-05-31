#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Template script to connect to Active Spark Session
Usage:
    $ spark-submit lab_3_storage_template_code.py <any arguments you wish to add>
'''

import getpass
# Import command line arguments and helper functions(if necessary)
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql import *



def main(spark, netID):
    '''Main routine for run for Storage optimization template.
    Parameters
    ----------
    spark : SparkSession object

    '''
   
    
    moives =spark.read.csv(f'hdfs:/user/{netID}/movies.csv', header = True, schema='movieId INT, title STRING, genres STRING')
    links = spark.read.csv(f'hdfs:/user/{netID}/links.csv', header = True,schema = 'movieId INT, imdbId STRING, tmdbId STRING')
    tags = spark.read.csv(f'hdfs:/user/{netID}/tags.csv', header = True, schema='userId INT, movieId INT, tag STRING, timestamp STRING')
    
    ratings = spark.read.csv(f'hdfs:/user/{netID}/ratings.csv', header = True,  schema = 'userId INT, movieId INT, rating FLOAT, timestamp INT')
    
    moives.write.parquet(f'hdfs:/user/{netID}/movies_small.parquet')
    ratings.write.parquet(f'hdfs:/user/{netID}/ratings_small.parquet')
    tags.write.parquet(f'hdfs:/user/{netID}/tags_small.parquet')
    links.write.parquet(f'hdfs:/user/{netID}/links_small.parquet')



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('final_small').getOrCreate()

    
    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)