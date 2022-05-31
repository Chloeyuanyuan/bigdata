import getpass
import sys
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.sql.functions import col

def main(spark):

    training = spark.read.parquet(f'hdfs:/user/yy3754/train_final_small.parquet', header = True)
    validation = spark.read.parquet(f'hdfs:/user/yy3754/validation_final_small.parquet', header = True)
    testing = spark.read.parquet(f'hdfs:/user/yy3754/test_final_small.parquet', header = True)

    training.createOrReplaceTempView("training")
    validation.createOrReplaceTempView("validation")
    testing.createOrReplaceTempView("testing")

    # train = training.select("userId", "movieId", "rating")
    # validation = validation.select("userId", "movieId", "rating")
    # test = testing.select("userId", "movieId", "rating")

    train = training.select(col('userId').cast('integer'),
                            col('movieId').cast('integer'),
                            col('rating').cast('float'))

    test = training.select(col('userId').cast('integer'),
                            col('movieId').cast('integer'),
                            col('rating').cast('float'))
    validation = training.select(col('userId').cast('integer'),
                                col('movieId').cast('integer'),
                                col('rating').cast('float'))

    als = ALS(userCol = 'userId', itemCol = "movieId", ratingCol = 'rating', coldStartStrategy="drop", nonnegative=True)

    param_grid = ParamGridBuilder().addGrid(als.rank, [5,10,15]).addGrid(als.maxIter, [1,5,10]).addGrid(als.regParam, [0.01, 0.1,0.9]).build()

    evaluator = RegressionEvaluator(metricName = "rmse", labelCol = "rating", predictionCol = "prediction")

    tvs = TrainValidationSplit(estimator = als, estimatorParamMaps=param_grid, evaluator = evaluator)

    model = tvs.fit(train)
    best_model = model.bestModel
    predictions = best_model.transform(validation)
    rmse = evaluator.evaluate(predictions)

    print("rmse = " + str(rmse))
    print("**best model**")
    print("  Rank:", best_model.rank)
    print("  MaxIter", best_model._java_obj.parent().getMaxIter())
    print("  RegParam:", best_model._java_obj.parent().getRegParam())


if __name__ == "__main__":
    spark = SparkSession.builder.appName('als').getOrCreate()
    
    main(spark)