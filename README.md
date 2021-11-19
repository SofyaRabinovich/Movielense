<h1 align="center">Movielense</h1>

# Content 

1. [Introduction](#intr)
2. [Aims and Objectives](#aim)
3. [Data Download](#data)
4. [Data Visualisation](#visual)
5. [Methods](#methods)
6. [Linear Model](#linear)
7. [Regularisation](#reg)
8. [Matrix Factorisation](#matrix)
9. [Results](#res)
10. [Conclusion](#conc)
11. [References](#ref)

<a name="intr"></a>

# Introduction 

Recommendation systems are an essential part of the machine learning algorithms and help offer users suggestions according to their preferences and selection of movies/products. Companies such as LinkedIn, Amazon, Netflix, etc., are using recommender systems to satisfy and ease their customers' search. 
In this project, we will look at the Netflix Prize open competition introduced on the 2nd of October 2006. The competition aimed to provide the best collaborative filtering algorithm to predict the user rating to the film based on the previous ratings using limited information. By June 2007, over 20 000 teams had registered for the competition, and in September 2009, the team called  "BellKor's Pragmatic Chaos" won the prize and achieved RMSE = 0.8567. The grand prize was US$1,000,000. 

<a name="aim"></a>

# Aims and Objectives
The aim of the project is to train different linear models to achieve the most accurate RMSE result using the provided training set (edx) and test set (validation). 

## Objectives: 

1. Explore the data; 
2. Look for the correlation between different parameters;
3. Preprocess the data by removing any NAs and zero variance parameters; 
4. Train different linear models and compare the RMSE results.

<a name="data"></a>

# Data Download

```
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")
# Validation set will be 10% of MovieLens data
set.seed(1) 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1,
                                  list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)

```

<a name="visual"></a>

# Data Visualisation

| userId               |  movieId      | rating        | timestamp         | title            | genres           |
| -------------------- | ------------- | ------------- | ----------------- | ---------------- | ---------------- |
| Min.   :    1        | Min.   :    1 | Min.   :0.500 | Min.   :7.897e+08 | Length:9000055   | Length:9000055 
| 1st Qu.:18124        | 1st Qu.:  648 | 1st Qu.:3.000 | 1st Qu.:9.468e+08 | Class :character | Class :character |
| Median :35738        | Median : 1834 | Median :4.000 | Median :1.035e+09 | Mode  :character | Mode  :character | 
| Mean   :35870        | Mean   : 4122 | Mean   :3.512 | Mean   :1.033e+09 |
| 3rd Qu.:53607        | 3rd Qu.: 3626 | 3rd Qu.:4.000 | 3rd Qu.:1.127e+09 |
| Max.   :71567        | Max.   :65133 | Max.   :5.000 | Max.   :1.231e+09 |

We can see that there are 71567 userIds and 65133 movieIds,however, we should define the nuber of unique userIds and movieIds
| userId | movieId |
| ------ | ------- |
| 69878  | 10677   |

The minimal rating is 0.5, and the maximum is 5.0. 

![rating skewness](images/rating.png)

It is obvious that some people preferred one movie genre to another, therefore, it is worth to investigate the effect of the genre on the movie rating. We will look at top 20 genres in the data set. We can see that the top genre is Drama followed by Comedy and Action. 

![standard error by genres](images/se-genres.png)

As we established there are 10677 movies in the data set and using the logic we can say that some movies are rated watched more than others and therefore are rated more frequently. Whereas, others can be rated only few times. We can see that some movies were rated only one time, and some were rated for 
more than 10'000 time.

![rating per movies](images/rating-movie.png)

As we defined previously there is 69878 and we can check the number of ratings given by them. We can see that some users have rated less than 30 movies, and this will underestimate our models.

![rating-user](images/rating-user.png)

We also know that more recent movies are tend to be rated more frequently than older ones. Therefore we can convert the timestamp column of the edx data set 
into a date of the rating was given. Afterwards, we will be able to explore the relationship of the date and rating. Looking at the graph can suggest some relationship between the time and the rating, however, there is no strong correlation. 

![rating-time](images/rating-time.png)

<a name="methods"></a>

# Methods

<a name="linear"></a>

## Linear Model 

The start model assumes the same prediction for all users and movies, explaining difference by random variation. Here ![mu](https://latex.codecogs.com/gif.latex?%5Cmu) represents the "true" rating for all movies. ![epsilon](https://latex.codecogs.com/gif.latex?%5Cepsilon%7B%5Ccolor%7BRed%7D%7D) is an independent errors sampled from the same distribution, that is centered at zero. The first equation to use will be: 
![equation1](https://latex.codecogs.com/gif.latex?%5Chat%7BY%7D_%7Bu%2Ci%7D%3D%5Cmu%20&plus;%20%5Cepsilon_%7Bu%2Ci%7D)

The ![hatY](https://latex.codecogs.com/gif.latex?%5Chat%7BY%7D) is the predicted rating. Any additional value will increase the  root mean squared error (RMSE).
![equation2](https://latex.codecogs.com/gif.latex?RMSE%20%3D%20%5Csqrt%7B%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bu%2Ci%7D%28%5Chat%7By%7D_%7Bu%2Ci%7D-y_%7Bu%2Ci%7D%29%5E2%7D)

Therefore, we will add an additional variable ![b_i](https://latex.codecogs.com/gif.latex?b_i) that represents the average ranking for the movie i, improving our RMSE model: 

![equation3](https://latex.codecogs.com/gif.latex?b_i%20%3D%20mean%28%5Chat%7BY%7D_%7Bu%2Ci%7D-%5Cmu%29)

![equation4](https://latex.codecogs.com/gif.latex?%5Chat%7BY%7D_%7Bu%2Ci%7D%3D%5Cmu&plus;b_i&plus;%5Cepsilon_%7Bu%2Ci%7D)

Knowing that some users ranked more movies than other users brings in place another bias, user-specific effect. We will call this variable ![bu](https://latex.codecogs.com/gif.latex?b_u), and therefore, increase the RMSE: 
![equation5](https://latex.codecogs.com/gif.latex?%5Chat%7BY%7D_%7Bu%2Ci%7D%3D%5Cmu&plus;b_i&plus;b_u&plus;%5Cepsilon_%7Bu%2Ci%7D)
From the data exploration we defined that there is some effect of the time on the rating, therefore we can imply time-specific effect to the RMSE model. However, the correlation was not significant and may cause in decrease of the RMSE, so we would not use time-specific effect in our analysis. 

<a name="reg"></a>

## Regularisation 
The linear model ![equation6](https://latex.codecogs.com/gif.latex?%5Chat%7BY%7D_%7Bu%2Ci%7D%3D%5Cmu&plus;b_i&plus;b_u&plus;%5Cepsilon_%7Bu%2Ci%7D) will provide a good estimation of ratings, however it will not penalize large estimates that come from small samples. For example, movies that were rated few times or users that gave ratings for very few movies. Not accounting this will lead to the large estimated errors. 
Therefore, the estimated value will be improved by applying the penalty term. The b's estimate now will be: 

![equation7](https://latex.codecogs.com/gif.latex?%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bu%2Ci%7D%28y_%7Bu%2Ci%7D-%5Cmu-b_i%29%5E2&plus;%5Clambda%5Csum_ib_i%5E2)

The ![equation8](https://latex.codecogs.com/gif.latex?%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bu%2Ci%7D%28y_%7Bu%2Ci%7D-%5Cmu-b_i%29%5E2) is the mean square error, and the ![penalty term](https://latex.codecogs.com/gif.latex?%5Clambda%5Csum_ib_i%5E2) is a penalty term. Note that when b is getting large the penalty term increases. Now we can calculate the movie-specific and the user-specific effects using regularization: 

1. ![b_i](https://latex.codecogs.com/gif.latex?%5Chat%7Bb_i%7D%20%3D%20%5Cfrac%7B1%7D%7B%5Clambda%20&plus;%20n_i%7D%5Csum%5E%7Bn_i%7D_%7Bu%3D1%7D%28y_%7Bu%2Ci%7D-%5Chat%7B%5Cmu%7D%29)
2. ![b_u](https://latex.codecogs.com/gif.latex?%5Chat%7Bb_u%7D%20%3D%20%5Cfrac%7B1%7D%7B%5Clambda%20&plus;%20n_u%7D%5Csum%5E%7Bn_u%7D_%7Bi%3D1%7D%28y_%7Bu%2Ci%7D-%5Chat%7Bb_i%7D-%5Chat%7B%5Cmu%7D%29)

Here ![lambda](https://latex.codecogs.com/gif.latex?%5Clambda) is a tuning parameter and we can use cross-validation to choose the minimum one that gives the most accurate RMSE.

<a name="matrix"></a>

## Matrix Factorisation 

Data can be converted into matrix to study the same rating patterns in the movies and users groups. Each user gets a row and each movie gets a column. The aim is to approximate the large matrix ![](https://latex.codecogs.com/gif.latex?R_%7Bm%5Ctimes%20n%7D) into two smaller vectors ![](https://latex.codecogs.com/gif.latex?P_%7Bk%5Ctimes%20m%7D) and ![](https://latex.codecogs.com/gif.latex?Q_%7Bk%5Ctimes%20m%7D), such that : ![](https://latex.codecogs.com/gif.latex?R%20%5Capprox%20P%27Q). ![](https://latex.codecogs.com/gif.latex?p_u) is the u-th row of P, and ![](https://latex.codecogs.com/gif.latex?q_i) is the v-th row of Q. Therefore, the the rating given by the user ![](https://latex.codecogs.com/gif.latex?u) for the movie ![](https://latex.codecogs.com/gif.latex?i) is ![](https://latex.codecogs.com/gif.latex?p_uq_i%27).This allows us to apply more variance in the original RMSE model: 

![equation9](https://latex.codecogs.com/gif.latex?%5Chat%7BY%7D_%7Bu%2Ci%7D%3D%5Cmu&plus;b_i&plus;b_u&plus;p_uq_i&plus;%5Cepsilon_%7Bi%2Cj%7D)

We can use Singular value composition (SVD) that finds the vectors p and q that permit us to write the matrix of residuals r with m rows and n columns. By using principal components analysis (PCA), matrix factorization can capture structure in the data determined by user opinions about movies. 
For Matrix Factorisation we will user the Recosystem package. 

<a name="res"></a>

# Results

| Method                                                |  RMSE         | 
| ------------------------------------------------------| ------------- | 
| Goal RMSE                                             | 0.8649000     | 
| Monte Carlo simulation                                | 1.4998974     |
| Just the average                                      | 1.0603313     | 
| Movie Effect Model                                    | 0.9439087     | 
| Movie + User Effect Model                             | 0.8653488     | 
| Regularised Movie + User Effect Model                 | 0.8648170     | 
| Matrix Factorisation	                                | 0.7823751     |

<a name="conc"></a>

# Conclusion

Using the training set and validation set we have successfully trained several linear regression models, which we studied in the previous courses of the HarvardX Data Science programme. We identified that the linear regression model using regularised user and movie effect model and matrix factorisation gave the desired result of the RMSE < 0.8649000. The matrix factorisation produced the RMSE = 0.7823751, and we achieved this using the recosystem package and model from the (https://www.kdnuggets.com/2019/09/machine-learning-recommender-systems.html). 

<a name="ref"></a>

# References

1. Irizarry, R. (2021). Introduction to Data Science. Retrieved 25 October 2021, from https://rafalab.github.io/dsbook/
2. Netflix Prize - Wikipedia. (2021). Retrieved 25 October 2021, from https://en.wikipedia.org/wiki/Netflix_Prize
3. Qiu, Y. (2021). recosystem: Recommender System Using Parallel Matrix Factorization. Retrieved 25 October 2021, from https://cran.r-project.org/web/packages/recosystem/vignettes/introduction.html
4. Seif, G. (2021). An Easy Introduction to Machine Learning Recommender Systems - KDnuggets. Retrieved 25 October 2021, from https://www.kdnuggets.com/2019/09/machine-learning-recommender-systems.html
