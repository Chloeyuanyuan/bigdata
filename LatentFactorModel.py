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

def main(spark):
    #load training set
    #train = spark.read.parquet('hdfs:/user/yy3754/train_final_small.parquet')
    train = spark.read.parquet('hdfs:/user/yy3754/train_medium.parquet')
    train.createOrReplaceTempView("train")
    train = train.select(col('userId').cast('integer'),
                            col('movieId').cast('integer'),
                            col('rating').cast('float'))
    #load validation set
    #val = spark.read.parquet('hdfs:/user/yy3754/validation_final_small.parquet')
    val = spark.read.parquet('hdfs:/user/yg1434/validation_medium.parquet')
    val.createOrReplaceTempView("val")
    val_new = spark.sql('select INT(userId) as userId, INT(movieId) as movieId, FLOAT(rating) as rating from val sort by rating DESC')
    val_new.createOrReplaceTempView("val_new")
    val_final = val_new.groupBy('userId').agg(collect_list('movieId').alias('movieId'))
    val_final.createOrReplaceTempView("val_final")

    #medium reg0.01，maxiter 20，rank 50 100 150 200
    maxIters = [20]
    regParams = [0.01]
    ranks = [100]
  
    for i in maxIters:
        for j in regParams:
            for k in ranks:
                init = time()
                als = ALS(rank = k, maxIter=i, regParam=j, userCol="userId", itemCol="movieId", ratingCol="rating",nonnegative = True, coldStartStrategy="drop")
                model= als.fit(train)
                predictions = model.recommendForAllUsers(100)
                predictions.createOrReplaceTempView('predictions')
                merged_result = spark.sql('select predictions.recommendations, val_final.movieId from val_final join predictions on predictions.userId= val_final.userId')
                merged_result = merged_result.collect()
                predictionAndLabels = []
                for row in merged_result:
                    truth = row[1] if len(row[1])<= 100 else row[1][0:100] #this is the list of true movies that in the top 100 of each user
                    pred = [i.movieId for i in row[0]]#this is the list of movies our recommender recommend to our user
                    predictionAndLabels.append((pred, truth))
                final_predictionAndLabels = sc.parallelize(predictionAndLabels)
                metrics = RankingMetrics(final_predictionAndLabels)
                total = time() - init
                
                #rmse
                predictions = model.transform(val_new)
                evaluator = RegressionEvaluator(metricName = "rmse", labelCol = "rating", predictionCol = "prediction")
                rmse = evaluator.evaluate(predictions)
                
                print("the result of maxIter = {}, regParam = {}, rank = {} : ".format(i, j, k))
                print("the Rmse is: ", rmse)
                print("the meanAP is : " , metrics.meanAveragePrecision)
                print("the ndcgAt 100 is : ", metrics.ndcgAt(100))
                print("the prec_at _k is : " , metrics.precisionAt(100))
                print("Running time : ", total)


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('ALS').getOrCreate()
    sc=spark.sparkContext

    main(spark)
    
