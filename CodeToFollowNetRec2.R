if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http.://cran.us.r-project.org")

library(recosystem)
library(tidyverse)
library(caret)
library(dplyr)
options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

#Training and Test Sets
test_index2 <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
Training <- edx[-test_index2,]
Temp2 <- edx[test_index2,]
Testing <- Temp2 %>% 
  semi_join(Training, by = "movieId") %>%
  semi_join(Training, by = "userId")
removed2 <- anti_join(Temp2, Testing)
Training <- rbind(Training, removed2)
rm(temp)
rm(Temp2)
rm(edx)

# Let's explore the data a bit. For our training set, let's measure the mean, median and standard deviation of the ratings. 
# Let's also count up unique users and movies.
n_rows <- nrow(Training)
cat("Number of rows in the dataset:", n_rows, "\n")
mean_rating <- mean(Training$rating, na.rm = TRUE)
median_rating <- median(Training$rating, na.rm = TRUE)
sd_rating <- sd(Training$rating, na.rm = TRUE)
cat("Mean of 'rating':", mean_rating, "\n")
cat("Median of 'rating':", median_rating, "\n")
cat("Standard Deviation of 'rating':", sd_rating, "\n")
unique_users <- n_distinct(Training$userId)
unique_movies <- n_distinct(Training$movieId)
cat("Number of unique users:", unique_users, "\n")
cat("Number of unique movies:", unique_movies, "\n")
rating_distribution <- Training %>%
  group_by(rating) %>%
  summarize(count = n())
print(rating_distribution)

# Let's graph the distribution of the ratings.
ggplot(Training, aes(x = rating)) +
  geom_histogram(fill = "dodgerblue", bins = 9) +
  labs(
    title = "Distribution of Ratings",
    x = "Rating",
    y = "Count"
  )

#Analysis starts here
#Accounting for user and movie effects...
y <- select(Training, movieId, userId, rating) |>
  pivot_wider(names_from = movieId, values_from = rating) 
rnames <- y$userId
y <- as.matrix(y[,-1])
rownames(y) <- rnames
mu <- mean(y, na.rm = TRUE)
b_i <- colMeans(y - mu, na.rm = TRUE)
fit_movies <- data.frame(movieId = as.integer(colnames(y)), 
                         mu = mu, b_i = b_i)
left_join(Testing, fit_movies, by = "movieId") |> 
  mutate(pred = mu + b_i) |> 
  summarize(rmse = RMSE(rating, pred))
# rmse
# 0.9441568
b_u <- rowMeans(sweep(y - mu, 2, b_i), na.rm = TRUE)
fit_users <- data.frame(userId = as.integer(rownames(y)), b_u = b_u)
left_join(Testing, fit_movies, by = "movieId") |> 
  left_join(fit_users, by = "userId") |> 
  mutate(pred = mu + b_i + b_u) |> 
  summarize(rmse = RMSE(rating, pred))
# rmse
# 0.8659736

#Now let's "penalize," to avoid overtraining 
n <-  colSums(!is.na(y))
lambdas <- seq(0, 10, 0.1)
sums <- colSums(y - mu, na.rm = TRUE)
rmses <- sapply(lambdas, function(lambda){
  b_i <-  sums / (n + lambda)
  fit_movies$b_i <- b_i
  left_join(Testing, fit_movies, by = "movieId") |> mutate(pred = mu + b_i) |> 
    summarize(rmse = RMSE(rating, pred)) |>
    pull(rmse)
})
qplot(lambdas, rmses, geom = "line")
lambda <- lambdas[which.min(rmses)]
print(lambda)
fit_movies$b_i_reg <- colSums(y - mu, na.rm = TRUE) / (n + lambda)
fit_users$b_u <- rowMeans(sweep(y - mu, 2, b_i), na.rm = TRUE)
left_join(Testing, fit_movies, by = "movieId") |> 
  left_join(fit_users, by = "userId") |> 
  mutate(pred = mu + b_i_reg + b_u) |> 
  summarize(rmse = RMSE(rating, pred))
# rmse
# 0.8658692

#Let's try to incorporate genres. To start we'll split the genre column into a matrix of columns.
Training$genre_list <- str_split(Training$genres, "\\|")
UniqueGenres <- unique(unlist(Training$genre_list))
matrix <- matrix(0, nrow = nrow(Training), ncol = length(UniqueGenres))
colnames(matrix) <- UniqueGenres
for (i in 1:nrow(Training)) {
  for (j in Training$genre_list[[i]]) {
    matrix[i,j] <- 1
  }
}
Training2 <- cbind(Training, matrix)
Training2 <- select(Training2, -timestamp, -title, -genre_list, -genres)
#And again for Testing data
Testing$genre_list <- str_split(Testing$genres, "\\|")
UniqueGenres <- unique(unlist(Testing$genre_list))
matrix <- matrix(0, nrow = nrow(Testing), ncol = length(UniqueGenres))
colnames(matrix) <- UniqueGenres
for (i in 1:nrow(Testing)) {
  for (j in Testing$genre_list[[i]]) {
    matrix[i,j] <- 1
  }
}
Testing2 <- cbind(Testing, matrix)
Testing2 <- select(Testing2, -timestamp, -title, -genre_list, -genres)
rm(matrix)

#Let's try to find genre effects.
genre_list <- c("Action", "Adventure", "Sci-Fi", "Children", "Comedy", "Fantasy",
                "Drama", "Romance", "War", "Western", "Thriller", "Crime", "Mystery",
                "Film-Noir", "Horror", "Animation", "Musical", "Documentary", "IMAX")
mean_ratings <- Training2 %>%
  summarize(
    across(all_of(genre_list), ~ mean(rating[. > 0]), .names = "mean_{.col}_rating")
  )
genre_effects <- Training2 %>%
  summarize(
    across(all_of(genre_list), ~ mean(rating[. > 0]), .names = "mean_{.col}_rating")
  ) %>%
  mutate(across(starts_with("mean"), ~ . - mu, .names = "effect_{.col}"))

genre_effects_movies <- Training2 %>%
  group_by(movieId) %>%
  summarize(across(all_of(genre_list), ~ mean(rating[!is.na(.)]), .names = "mean_{.col}_rating")) %>%
  mutate(across(starts_with("mean"), ~ . - mu, .names = "effect_{.col}"))
genre_effects_users <- Training2 %>%
  group_by(userId) %>%
  summarize(across(all_of(genre_list), ~ mean(rating[!is.na(.)]), .names = "mean_{.col}_rating")) %>%
  mutate(across(starts_with("mean"), ~ . - mu, .names = "effect_{.col}"))

fit_movies2 <- left_join(fit_movies, genre_effects_movies, by = "movieId")
fit_users2 <- left_join(fit_users, genre_effects_users, by = "userId")

# Generate predictions incorporating genre effects
Testing2 <- left_join(Testing2, fit_movies2, by = "movieId") %>%
  left_join(fit_users2, by = "userId") %>%
  mutate(
    pred = mu + b_i + b_u + rowMeans(across(starts_with("effect_")), na.rm = TRUE, dims = 2)
  )
Testing2 %>% summarize(rmse = RMSE(rating, pred))
# rmse
# 0.9352433

# Since that RMSE wasn't what we hoped for, perhaps it'd be better to temper our genre effects' influence on our predictions
Testing3 <- left_join(Testing2, fit_movies2, by = "movieId") %>%
  left_join(fit_users2, by = "userId") %>%
  mutate(
    pred = mu + b_i + b_u + .5 * rowMeans(across(starts_with("effect_")), na.rm = TRUE, dims = 2)
  )
Testing3 %>% summarize(rmse = RMSE(rating, pred))
# rmse
# 0.8844345
# That still doesn't help us, unfortunately. 

# Let's try applying matrix factorization using the recosystem package
reco_train <- with(Training, data_memory(user_index = userId, item_index = movieId, rating = rating))
reco_test <- with(Testing, data_memory(user_index = userId, item_index = movieId, rating = rating))
reco <- Reco()
reco_tuning <- reco$tune(reco_train, opts = list(dim = c(20, 30), costp_12 = c(0.01, 0.1), costq_12 = c(0.01, 0.1), lrate = c(.005,.05), nthread = 4, niter = 10))

# I started with a large number of dimensions to help identify complex patterns in this huge dataset. However, I chose to reduce to speed up the process for replication.
# When selecting regularization parameters, I tried to balance the need to prevent over-fitting with the restraints of my laptop. 
# With the lrate, I similarly wanted to balance the time it would take to run with accuracy.
reco$train(reco_train, opts = c(reco_tuning$min, nthread = 4, niter = 30))
reco_prediction <- reco$predict(reco_test, out_memory())
factor_rmse <- RMSE(reco_prediction, Testing$rating)
factor_rmse
# 0.7896491
# that's great! finally a much lower rmse! 
# Let's apply it to the final holdout set.
reco_holdout <- with(final_holdout_test, data_memory(user_index = movieId, item_index = userId, rating = rating))
reco_holdout_prediction <- reco$predict(reco_holdout, out_memory())
factor_rmse_holdout <- RMSE(reco_holdout_prediction, final_holdout_test$rating)
factor_rmse_holdout
# 0.7896262