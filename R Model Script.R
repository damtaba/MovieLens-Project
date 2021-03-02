##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################
# Given, with minor changes with commentaries


if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org") 
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem") # For matrix factorization
if(!require(tinytex)) install.packages("tinytex") 

# For making the pdf
if(!require(rmarkdown)) install.packages("rmarkdown")
#if(!require(proTeXt)) install.packages("proTeXt")
if(!require(formatR)) install.packages("formatR")
if(!require(latexpdf)) install.packages("latexpdf") 
if(!require(xfun)) install.packages("xfun") 
#if(!require(proTeXt)) install.packages("MikTek") 
#tinytex::install_tinytex()
options(pillar.sigfig=7)

library(tidyverse)
library(caret)
library(data.table)
library(recommenderlab)
library(recosystem)
library(ggthemes)
#library(MikTek)
library(latexpdf)
library(xfun)
library(tinytex)
library(rmarkdown)
library(formatR)
#library(proTeXt)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
edx1 <- edx # to not use the original edx
rm(dl, ratings, movies, test_index, temp, movielens, removed)


# Making edx_clean with year_title. It has the year of the movie 
edx_clean <- 
  edx1 %>%
  mutate(title = str_trim(title)) %>% # to not have problems with white spaces
  mutate(year_title_temp = str_extract_all(title,pattern="\\((\\d{4})\\)"),year_title = as.integer(str_extract(year_title_temp,pattern = "\\d{4}"))) %>%
  select(-year_title_temp) # First it will catch the 4 numbers that are between parethesis. Then, it will remove the parethesis.



##########################################################
#################### | Partitions | #######################
##########################################################



#Creating the partition of the edx_clean dataset. We'll use 90% for the training of the model and 10% for testing. 
#This ratio is possible thanks to the large amount of data
test_index <- createDataPartition(edx_clean$rating,p=0.1,list=FALSE)


train_set <- edx_clean[-test_index,] 
test_set_temp <- edx_clean[test_index,]
validation_temp <- validation

# Matching the dataset variables for the test set
test_set <- test_set_temp %>%
  semi_join(train_set,by="movieId") %>%
  semi_join(train_set,by="userId") %>%
  semi_join(train_set,by="genres") %>%
  semi_join(train_set,by="year_title")

# Matching the dataset variables for the validation set
validation_final <- 
  validation_temp %>%
  # select(title) %>%
  #  head(n = 1) %>%
  mutate(title = str_trim(title)) %>%
  mutate(year_title_temp = str_extract_all(title,pattern="\\((\\d{4})\\)"),year_title = as.integer(str_extract(year_title_temp,pattern = "\\d{4}"))) %>%
  select(-year_title_temp) %>%
  semi_join(train_set,by="movieId") %>%
  semi_join(train_set,by="userId") %>%
  semi_join(train_set,by="genres") %>%
  semi_join(train_set,by="year_title")



##########################################################
#################### | Averages | #######################
##########################################################


# What regularization does is penalize the information based on small samples by incorporating the lambda variable. 
# We will reduce this random variation, explaining part of this by the effect of the film, the user, the genre and the year of the film

#the base of the model, de average score of every rate made
mu_hat <-mean(train_set$rating) 

#making the averages for every movie
movie_avgs <- train_set %>%
  group_by(movieId) %>% 
  summarize(b_movie = mean(rating - mu_hat)) # substracting the part explained by the total average

#making the averages for every user
user_avgs <- train_set %>% 
  left_join(movie_avgs,by='movieId') %>%
  group_by(userId) %>% 
  summarize(b_user = mean(rating - mu_hat - b_movie)) # substracting the part explained by the total average and movie effect

#making the averages for every combined genre
genre_avg_more_than_one_avg <- train_set %>%
  filter(str_detect(genres,"\\|")==TRUE) %>% #movies with more than one genre
  left_join(movie_avgs,by='movieId') %>%
  left_join(user_avgs,by='userId') %>%
  group_by(genres) %>% 
  summarize(b_genre = mean(rating - mu_hat - b_user - b_movie)) # substracting the part explained by the total average and movie and user effect

#making the averages for every genre
genre_avg_per_genre_avgs <- train_set %>% 
  left_join(movie_avgs,by='movieId') %>%
  left_join(user_avgs,by='userId') %>%
  separate_rows(genres, sep = "\\|") %>% # The separator for every genre is "|" 
  group_by(genres) %>% 
  summarize(b_genre = mean(rating - mu_hat - b_user - b_movie)) # substracting the part explained by the total average and movie and user effect

genre_avgs <- bind_rows(genre_avg_more_than_one_avg,genre_avg_per_genre_avgs) 

#making the averages per year
year_avgs <- train_set %>% 
  left_join(movie_avgs,by='movieId') %>%
  left_join(user_avgs,by='userId') %>%
  left_join(genre_avgs,by='genres') %>%
  group_by(year_title) %>% 
  summarize(b_year = mean(rating - mu_hat - b_movie - b_user - b_genre)) # substracting the part explained by the total average and movie, user and year effect

# Making predictions
predicted_ratings_avg <- test_set %>%
  left_join(movie_avgs,by='movieId') %>%
  left_join(user_avgs,by='userId') %>%
  left_join(genre_avgs,by='genres') %>%
  left_join(year_avgs,by='year_title') %>%
  mutate(prediction = mu_hat+b_movie+b_user+b_genre+b_year) %>%
  .$prediction

RMSEs_Table <- tibble(Method = "Goal", RMSE = 0.8649)

RMSE_avg_movie <- RMSE(predicted_ratings_avg,test_set$rating)

RMSEs_Table <- bind_rows(RMSEs_Table, 
                         tibble(Method = "Avg + movie, user, genres and year effect",RMSE=RMSE_avg_movie))



##########################################################
#################### | Regularization | #######################
##########################################################



#What regularization does is penalize the information based on small samples by means of the lambda variable. 
# By iterating through different lambda scenarios, the lambda that minimizes the RMSE is selected

lambdas1 <- seq(1,10,1)

# Iterating the lambdas
rmses_lambdas_reg1 <- sapply(lambdas1,function(x){
  #Using the terms that previously were done, but with some changes doing the regularization
  mu <- mean(train_set$rating)
  
  movie_avgs_reg1 <- train_set %>%
    group_by(movieId) %>% 
    summarize(b_movie = sum(rating - mu)/(n()+x))
  
  user_avgs_reg1 <- train_set %>% 
    left_join(movie_avgs_reg1,by='movieId') %>%
    group_by(userId) %>% 
    summarize(b_user = sum(rating - mu - b_movie)/(n()+x))   
  
  b_g_1_reg1 <- train_set %>%
    filter(str_detect(genres,"\\|")==TRUE) %>%
    left_join(movie_avgs_reg1,by='movieId') %>%
    left_join(user_avgs_reg1,by='userId') %>%
    group_by(genres) %>%
    summarize(b_genres = sum(rating - mu - b_user - b_movie)/(n()+x))  
  
  b_g_2_reg1 <- train_set %>%
    left_join(movie_avgs_reg1,by='movieId') %>%
    left_join(user_avgs_reg1,by='userId') %>%    
    separate_rows(genres, sep = "\\|") %>%    
    group_by(genres) %>%
    summarize(b_genres = sum(rating - mu - b_user - b_movie)/(n()))
  
  b_g_reg1 <- bind_rows(b_g_1_reg1,b_g_2_reg1)
  
  b_y_reg1 <- train_set %>%
    left_join(movie_avgs_reg1,by='movieId') %>%
    left_join(user_avgs_reg1,by='userId') %>%
    left_join(b_g_reg1,by='genres') %>%
    group_by(year_title) %>%
    summarize(b_years = sum(rating - mu - b_user - b_movie-b_genres)/(n()+x))
  
  predicted_ratings <- test_set %>%
    left_join(movie_avgs_reg1,by='movieId') %>%
    left_join(user_avgs_reg1,by='userId') %>%
    left_join(b_g_reg1,by='genres') %>%
    left_join(b_y_reg1,by='year_title') %>%
    mutate(pred = mu + b_movie + b_user + b_genres + b_years) %>%
    .$pred
  
  return(RMSE(predicted_ratings,test_set$rating))
  
}
)

#qplot(lambdas1,rmses_lambdas_reg1)
lambda1 <- lambdas1[which.min(rmses_lambdas_reg1)]

RMSEs_Table <- bind_rows(RMSEs_Table, 
                         tibble(Method = "Regularization",RMSE=min(rmses_lambdas_reg1)))



##########################################################
#################### | Matrix Factorization | ############
##########################################################



# Matrix factorization algorithms work by decomposing the user-item interaction matrix into the product of two lower dimensionality rectangular matrices

#creating the test and train dataset for the matrix factorization
train_data_mf <-  with(train_set, data_memory(user_index = userId, 
                                           item_index = movieId, 
                                           rating     = rating))

test_data_mf  <-  with(test_set,  data_memory(user_index = userId, 
                                           item_index = movieId, 
    
                                                                                  rating     = rating))
# object of class "RecoSys" that can be used to construct recommender model and conduct prediction
r <-  recosystem::Reco()

# Training the data
r$train(train_data_mf)

# Making predictions
y_hat_reco <-  r$predict(test_data_mf, out_memory())


RMSEs_Table <- bind_rows(RMSEs_Table, 
                         tibble(Method = "Matrix Factorization",RMSE=RMSE(test_set$rating, y_hat_reco)))


##########################################################
#################### | Final Validation | ################
##########################################################



# Final validation using the best model

# Creating validation test for matrix recomendation
validation_data_mf  <-  with(validation_final,  data_memory(user_index = userId, 
                                              item_index = movieId, 
                                              rating     = rating))

r <-  recosystem::Reco()

r$train(train_data_mf)

y_hat_reco_validation <-  r$predict(validation_data_mf, out_memory())

RMSEs_Table <- bind_rows(RMSEs_Table, 
                         tibble(Method = "Final Validation",RMSE=RMSE(validation_final$rating, y_hat_reco_validation)))


