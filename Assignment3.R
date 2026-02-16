library(FNN)
library(dplyr)
library(tidyverse)
library(kableExtra)
library(naivebayes)

# 1 
dataRecid <- read.csv("NIJ_s_Recidivism_Challenge_Full_Dataset.csv")
head(dataRecid)

# 2 
# Tidy missing values and create indicators for missing continuous features

dataRecid <- dataRecid |>
  mutate(
    Prison_Offense = case_match(Prison_Offense, "" ~ "Missing", .default = Prison_Offense),
    Supervision_Risk_Score_NA = as.numeric(is.na(Supervision_Risk_Score_First)),
    Supervision_Risk_Score_First = coalesce(Supervision_Risk_Score_First, 0)
  )

# Defining features 

features_chosen <- c(
  "Gender", "Race", "Age_at_Release", "Education_Level", "Dependents",
  "Prison_Offense", "Prison_Years", "Prior_Arrest_Episodes_Felony", 
  "Prior_Arrest_Episodes_Misd", "Prior_Arrest_Episodes_Violent", 
  "Prior_Arrest_Episodes_Property", "Prior_Arrest_Episodes_Drug",
  "Prior_Conviction_Episodes_Felony", "Prior_Conviction_Episodes_Misd",
  "Prior_Conviction_Episodes_Viol", "Prior_Conviction_Episodes_Prop",
  "Prior_Conviction_Episodes_Drug", "Condition_MH_SA", "Condition_Cog_Ed",
  "Delinquency_Reports", "Program_Attendances", "Supervision_Risk_Score_First"
)

# Converting categorical features to 0/1 indicators

formula_str <- paste("~", paste(c(features_chosen, "Supervision_Risk_Score_NA"), collapse = " + "))

X_mat <- model.matrix(as.formula(formula_str), data = dataRecid)

# Dropping (Intercept)
X <- X_mat[, -1] 

# 3 
# Defining Outcome Variable

y <- as.numeric(dataRecid$Recidivism_Within_3years == "true")

# Splitting into Training and Validation

train_idx <- which(dataRecid$Training_Sample == 1)
valid_idx <- which(dataRecid$Training_Sample == 0)

X_train <- X[train_idx, ]
y_train <- y[train_idx]
X_valid <- X[valid_idx, ]
y_valid <- y[valid_idx]

# 4 
# Cross-validation

k_list <- seq(1, 101, by = 10) 

results <- data.frame(k = rep(0, length(k_list)), logLikelihood = rep(0.0, length(k_list)))
for(i in 1:length(k_list)) {
  k_val <- k_list[i]
  knn_cv <- FNN::knn.cv(train = X_train, cl = y_train, k = k_val, prob = TRUE)
  # Laplace Correction
  prob_win <- attr(knn_cv, "prob")
  p <- ifelse(knn_cv == 1, prob_win, 1 - prob_win)
  p_adj <- (p * k_val + 1) / (k_val + 2)
  results$k[i] <- k_val
  results$logLikelihood[i] <- mean(ifelse(y_train == 1, log(p_adj), log(1 - p_adj)))
}

best_k <- results$k[which.max(results$logLikelihood)]

# 5 
# Plot 

plot(results$k, results$logLikelihood, type="b", pch=19, col="red",
     main="Question 5: k-NN Optimization", xlab="k", ylab="Avg Log-Likelihood")

# 6 
# Evaluating k-NN on Validation Data

final_knn <- FNN::knn(train = X_train, test = X_valid, cl = y_train, k = best_k, prob = TRUE)

p_v <- ifelse(final_knn == 1, attr(final_knn, "prob"), 1 - attr(final_knn, "prob"))

p_adj_v <- (p_v * best_k + 1) / (best_k + 2)

knn_loglik_val <- mean(ifelse(y_valid == 1, log(p_adj_v), log(1 - p_adj_v)))

# 7
# Comparing with Naive Bayes

nb_model <- naivebayes::naive_bayes(x = as.data.frame(X_train), 
                                    y = as.factor(y_train), 
                                    laplace = 1)


nb_prob <- predict(nb_model, newdata = as.data.frame(X_valid), type = "prob")[, 2]

# Laplace smoothing for the log-likelihood calculation
# Preventing probabilities of exactly 0 or 1 which caused NaNs in log() earlier
eps <- 1e-10
nb_prob_safe <- pmax(pmin(nb_prob, 1 - eps), eps)

# Calculating Average Bernoulli Log-Likelihood
nb_loglik_val <- mean(ifelse(y_valid == 1, log(nb_prob_safe), log(1 - nb_prob_safe)))

cat("Naive Bayes Validation Log-Likelihood:", nb_loglik_val, "\n")

# 8 
# Normalised knn scaling 

X_scaled <- scale(X)
X_train_s <- X_scaled[train_idx, ]
X_valid_s <- X_scaled[valid_idx, ]

# Rerunning for scaled data

scaled_knn <- FNN::knn(train = X_train_s, test = X_valid_s, cl = y_train, k = best_k, prob = TRUE)
p_s <- ifelse(scaled_knn == 1, attr(scaled_knn, "prob"), 1 - attr(scaled_knn, "prob"))
p_adj_s <- (p_s * best_k + 1) / (best_k + 2)
knn_scaled_loglik <- mean(ifelse(y_valid == 1, log(p_adj_s), log(1 - p_adj_s)))

# Final Table Comparison

comparison <- data.frame(
  Method = c("k-NN (Unscaled)", "Naive Bayes", "k-NN (Scaled)"),
  Avg_Log_Likelihood = c(knn_loglik_val, nb_loglik_val, knn_scaled_loglik)
)
print(comparison)

# 9 
# Some reasons why the models perform differently:

# When I ran the first k-NN, I realized that features with larger ranges—like
# Prior_Arrest_Episodes or Supervision_Risk_Score—were basically manipulating
# the binary features. Because k-NN just calculates distances between points, 
# a difference of 5 arrests looks bigger to the model than the difference 
# between being male or female (which is only 0 or 1). Once I scaled the data 
# in question 8, every feature got to have an equal say, which is why the 
# scaled version usually performs better.

# Naive Bayes also assumes that every piece of information is independent. 
# But in real life, things like being Gang_Affiliated and having a 
# high Supervision_Risk_Score are obviously linked. 
# Naive Bayes naively treats them as separate votes, which can lead to 
# overconfident or slightly off probabilities. k-NN is actually smarter here 
# because it looks at the whole person (the "neighbor") and captures how those 
# traits overlap naturally.

