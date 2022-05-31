# Collaborative-filter Based Recommender System

## The data set

In this project, we'll use the [MovieLens](https://grouplens.org/datasets/movielens/latest/) datasets collected by 
> F. Maxwell Harper and Joseph A. Konstan. 2015. 
> The MovieLens Datasets: History and Context. 
> ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

The data is hosted in NYU's HPC environment under `/scratch/work/courses/DSGA1004-2021/movielens`.

Two versions of the dataset are provided: a small sample (`ml-latest-small`, 9000 movies and 600 users) and a larger sample (`ml-latest`, 58000 movies and 280000 users).
Each version of the data contains rating and tag interactions, and the larger sample includes "tag genome" data for each movie, which you may consider as additional features beyond the collaborative filter.
Each version of the data includes a README.txt file which explains the contents and structure of the data which are stored in CSV files.

## Basic recommender system 

1.  Partition the rating data into training, validation, and test samples as discussed in lecture.

2.  Evaluate popularity baseline model before implementing a sophisticated model.

3.  Use Spark's alternating least squares (ALS) method to learn latent factor representations for users and items.
    [pyspark.ml.recommendation module](https://spark.apache.org/docs/3.0.1/ml-collaborative-filtering.html).
    This model has some hyper-parameters that you should tune to optimize performance on the validation set, notably: 
      - the *rank* (dimension) of the latent factors, and
      - the regularization parameter.

### Evaluation

Evaluations is based on predictions of the top 100 items for each user, and report the ranking metrics provided by spark.
Reference: [ranking metrics](https://spark.apache.org/docs/3.0.1/mllib-evaluation-metrics.html#ranking-systems) 


### Using the cluster

Peel cluster.

## Extensions 

  - *Comparison to single-machine implementations*: compare Spark's parallel ALS model to a single-machine implementation, e.g. [lightfm](https://github.com/lyst/lightfm) or [lenskit](https://github.com/lenskit/lkpy).  Measure both efficiency (model fitting time as a function of data set size) and resulting accuracy.
 
  - *Qualitative error analysis*: using your best-performing latent factor model, investigate the mistakes that it makes.  This can be done in a number of ways, including (but not limited to):
    - investigating the trends and genres of the users who produce the lowest-scoring predictions
    - visualizing the learned item representation via dimensionality reduction techniques (e.g. T-SNE or UMAP) with additional data for color-coding (genre tags, popularity, etc)
    - clustering users by their learned representations and identifying common trends in each cluster's consumption behavior

