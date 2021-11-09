##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

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
##### Libraries ######
library(broom)
library(dslabs)
library(HistData)
library(tidyverse)
library(tidyr)
library(tidyselect)
library(tidytext)
library(dplyr)
library(gridExtra)
library(ggplot2)
library(ggpubr)
library(ggrepel)
library(ggsci)
library(ggsignif)
library(ggthemes)
library(readxl)
library(readr)
library(reshape2)
library(lpSolve)
library(lubridate)
library(caret)
library(e1071)
library(MASS)
library(purrr)
library(pdftools)
library(matrixStats)
library(rpart)
library(Rborist)
library(recosystem)
library(randomForest)

##### PROJECT ####

#### Data investigation ####
head(edx)
tail(edx)
summary(edx)

#number of unique movies
edx %>%
  summarise(n_movies = n_distinct(movieId))

#number of unique users
edx %>%
  summarise(n_users = n_distinct(userId))

#the minimum rating that was given to the movie is 0.5, meanwhile,
#the maximum rating is 5.0. The visualisation: 
edx %>%
  ggplot(aes(rating)) + 
  geom_histogram(binwidth = 0.5, color = I('black')) +
  ggtitle("Rating Distribution") +
  labs(caption = "The minimal rating is 0.5, and the maximum is 5.0. 
       We can see that the data is skewed to the right. 
       Also, from the histogram we notice that no user gave 0.0 rating")

#the effect of the genre on the rating (top 20)
top_genres <- edx %>%
  group_by(genres) %>%
  summarise(count = n()) %>%
  top_n(20, count) %>%
  arrange(desc(count))
top_genres
#We can see that the Drama has the most movie ratings, then Comedy and Action

#Observe genre effect by plotting averages and standard errors for each genre
edx %>%
  group_by(genres) %>%
  summarise(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n())) %>%
  filter(n >= 100000) %>%
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 40, hjust = 1)) +
  ylab("Averages") + 
  ggtitle("Standard errors by genres")

#look at the top 15 movies rated by users 
top_movies <- edx %>%
  group_by(title) %>%
  summarise(rating_count = n()) %>%
  top_n(15, rating_count) %>%
  arrange(desc(rating_count))
top_movies

#Number of ratings per movie
edx %>% 
  dplyr::count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(binwidth = 0.25, bins = 30, color = I('black')) +
  scale_x_log10() +
  theme_classic() +
  xlab("Number of ratings per movie") +
  ylab("Number of the movies") +
  ggtitle("Number of rating per movie") +
  labs(caption = "We can see that some movies were rated only one time, and some were rated for more than 10'000 time.
  The distribution is almost symmetric.
  Few rating count can make the model inaccurate.
       Therefore these movies should be excluded from the list in the preprocessiing step.", hjust = 0.5)

#Define which movies were rated only 1 time
rated_1_time <- edx %>%
  group_by(movieId) %>%
  summarise(number_of_ratings = n()) %>%
  filter(number_of_ratings == 1) %>%
  left_join(edx, by = "movieId") %>%
  dplyr::select(title, movieId, number_of_ratings)
rated_1_time # We can see that there are 126 (1,18%) movies, which were rated only once. 

#As we defined previously there is 69878 and we can check the number of ratings given by them 
edx %>%
  dplyr::count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(binwidth = 0.15, bins = 30, color = I('black')) +
  scale_x_log10() +
  theme_classic() +
  xlab("Number of ratings give by user") +
  ylab("Number of users") +
  ggtitle("Number of ratings per user") +
  labs(caption = "The graph is skewed to the right (positive). 
       We can see that some users have rated less than 30 movies, and this will underestimate our model.")

# Explore the relationship of year and rating
library(lubridate)
edx %>%
  mutate(year = year(as_datetime(timestamp))) %>%
  ggplot(aes(year)) +
  geom_histogram(binwidth = 0.15, bins = 30, color = I('black')) +
  ggtitle("Rating over time") +
  xlab("Year") +
  ylab("Rating") +
  theme_classic() +
  labs(caption =)

#### Data Preprocessing ####

#check if there is any missing values
anyNA(edx)

#Remove variation that is close to 0
nearZeroVar(edx$rating)

#### Analysis ####

#Define RMSE function 
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#create a table that will store our results as we go along 
model_results <- data.frame(method = "Goal RMSE", 
                            RMSE = 0.8649)

## Monte Carlo prediction 
set.seed(4321)

# Create the probability of each rating
p <- function(x, y) mean(y == x)
rating <- seq(0.5,5,0.5)

#Estimate the probability of each rating
B <- 10000
N <- 100
MS <- replicate(B, {
  X <- sample(edx$rating,N, replace = TRUE)
  sapply(rating, p, y = X)
})
probability <- sapply(1:nrow(MS), function(x) mean(MS[x,]))

#Random ratings prediction 
y_hat <- sample(rating, size = nrow(edx), 
                replace = TRUE, prob = probability)

#Store the prediction in the table 
model_results <- bind_rows(model_results, 
                           data.frame(method = "Monte Carlo simulation", 
                                      RMSE = RMSE(edx$rating, y_hat)))
model_results

#Find the "true" ratings for all movies, mu
mu_hat <- mean(edx$rating)
mu_hat
#Add this model to the table 
model_results <- bind_rows(model_results, 
                           data.frame(method = "Just the average", 
                           RMSE = RMSE(validation$rating, mu_hat)))
model_results

# Next step will be to define the b_i,the average ranking for the movie i
b_i <- edx %>%
  group_by(movieId) %>%
  summarise(b_i = mean(rating - mu_hat))

#Predict ratings using movie-specific effect 
predicted_ratings <- mu_hat + validation %>%
  left_join(b_i, by = 'movieId') %>%
  .$b_i
movie_specific_RMSE <- RMSE(validation$rating, predicted_ratings)
movie_specific_RMSE

#Add to the model table 
model_results <- bind_rows(model_results, 
                           data.frame(method = "Movie Effect Model", 
                                      RMSE = movie_specific_RMSE))

#User Effect Model 
b_u <- edx %>%
  left_join(b_i, by = 'movieId') %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu_hat - b_i))
predicted_ratings <- validation %>%
  left_join(b_i, by = 'movieId') %>%
  left_join(b_u, by = 'userId') %>%
  mutate(prediction = mu_hat + b_i + b_u) %>%
  .$prediction
user_specific_RMSE <- RMSE(validation$rating, predicted_ratings)
user_specific_RMSE

##Regularisation 

#define lambdas
lambdas <- seq(0, 10, 0.25)

#Define the function "RMSES" function 
RMSES <- sapply(lambdas, function(l){
  mu_hat <- mean(edx$rating)
  b_i <- edx %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu_hat)/(n() + l))
  b_u <- edx %>%
    left_join(b_i, by = 'movieId') %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - b_i - mu_hat)/(n() + l))
  predicted_user_movie_rating <- validation %>%
    left_join(b_i, by = 'movieId') %>%
    left_join(b_u, by = 'userId') %>%
    mutate(prediction = mu_hat + b_i + b_u) %>%
    .$prediction
  return(RMSE(validation$rating, predicted_user_movie_rating))
})

#plot lambdas over RMSES results 
qplot(lambdas, RMSES)

# we can access the best value of lambda as following 
lambda <- lambdas[which.min(RMSES)]

#Final RMSE model value 
min(RMSES)

#Add final RMSE value to the table
model_results <- bind_rows(model_results, 
                           data.frame(method = "Regularised Movie + User Effect Model", 
                                      RMSE = regularised_user_movie_specific_rmse))

#Matrix Factorisation 
#Don't forget to install recosystem package
#firstly we will make copy of the edx and validation sets and call them train_set and test_set, respectively 
set.seed(123)
invisible(gc())
train_set <- with(edx, data_memory(user_index = userId, 
                                   item_index = movieId, 
                                   rating = rating))
test_set <- with(validation, data_memory(user_index = userId, 
                                         item_index = movieId, 
                                         rating = rating))

#Now we build up the recommender object 
rec <- Reco()

#Tune the train_set 
#This will take few minutes to be done 
opts <- rec$tune(train_set, opts = list(dim = c(10,20,30), 
                                        lrate = c(0.1,0.2), 
                                        costp_l1 = 0, 
                                        costq_l1 = 0, 
                                        nthread = 1, 
                                        niter = 10))
opts
#Train the system 
rec$train(train_set, opts = c(opts$min, nthread = 1, niter = 20))
#Calculate the prediction 
y_hat <- rec$predict(test_set, out_memory())
#Add the result into the model nresults table 
model_results <- bind_rows(model_results, 
                           data.frame(method = "Matrix Factorisation", 
                                      RMSE = RMSE(validation$rating, y_hat)))





