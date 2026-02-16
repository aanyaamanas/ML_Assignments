# Assignment 
library(dplyr)
library(naivebayes)
library(tidyr)
library(kableExtra)
library(pROC)
library(classInt)

dataRecid <- read.csv("NIJ_s_Recidivism_Challenge_Full_Dataset.csv")

#Data cleaning 

dataRecid <- dataRecid |>
  mutate(
    Prison_Offense = ifelse(Prison_Offense == "", "NA", Prison_Offense),
    Gang_Affiliated = ifelse(Gang_Affiliated == "", "NA", Gang_Affiliated),
    Supervision_Level_First = ifelse(Supervision_Level_First == "", "NA", Supervision_Level_First)
  )

# A) Fitting model 
# Adding Gang_Affiliated and Supervision_Level_First as new features 
# because social ties and supervision intensity are often predictive

nb_model <- naive_bayes(
  (Recidivism_Within_3years == "true") ~ Gender + Age_at_Release + Education_Level +
    Prior_Conviction_Episodes_Viol + Prison_Offense + Prison_Years +
    Gang_Affiliated + Supervision_Level_First,
  data = subset(dataRecid, Training_Sample == 1),
  laplace = 1
)

summary(nb_model)

# Making predictions on the test data 
# probabilities for the whole dataset 

dataRecid$pNB <- predict(nb_model, newdata = dataRecid, type = "prob")[,2]

# B) Calculating performance metrics for test data only 

#misclassification rate (threshold = 0.5)

misclass <- dataRecid |>
  filter(Training_Sample == 0) |>
  summarise(misclass = mean((Recidivism_Within_3years == "false" & pNB > 0.5) | 
              (Recidivism_Within_3years == "true" & pNB <0.5 )))

round(misclass$misclass, 4)

#0.3662 

# false positive rate (type 1 error)

fpr <- dataRecid |>
  filter(Recidivism_Within_3years == "false" & Training_Sample == 0) |>
  summarise(falsePos = mean(pNB > 0.5))

round(fpr$falsePos, 4)

# 0.5208

# false negative rate (type 2 error)

fnr <- dataRecid |>
  filter(Recidivism_Within_3years == "true" & Training_Sample == 0) |>
  summarise(falseNeg = mean(pNB < 0.5))

round(fnr$falseNeg, 4)

# 0.2516

# C) ROC curve and AUC using test data only 

nbROC <- roc((Recidivism_Within_3years == "true") ~ pNB,
             data = subset(dataRecid, Training_Sample == 0))
plot(nbROC, main = "ROC Curve")
#AUC
round(nbROC$auc, 4)

# 0.6738 

# D) Whether probabilities are well-calibrated 

dataRecid |>
  filter(Training_Sample == 0) |>
  mutate(pCat = cut(pNB, breaks = (0:10)/10)) |> 
  filter(!is.na(pCat)) |>
  group_by(pCat) |>
  summarise(phat = mean(Recidivism_Within_3years == "true"),
            p_mean = mean(pNB)) |>
  plot(phat ~ p_mean, data = _, pch = 16, col = "red",
       xlim = 0:1, ylim = 0:1,
       xlab = "Predicted Probability",
       ylab = "Actual Probability",
       main = "Calibration Plot")

abline(0,1)  


# E) Evidence balance sheet for one parolee

#selecting random parolee from the test set 

target_id <- dataRecid$ID[dataRecid$Training_Sample == 0][1]

target_parolee <- dataRecid |>
  filter(ID == target_id)



# helper fn to get WoE from the NB object tables 

woe_model <- function(model, var, val) {
  tbl <- model$tables[[var]]
  prob_false <- tbl[val, "FALSE"]
  prob_true <- tbl[val, "TRUE"]
  return(log(prob_true/prob_false))
}



vars <- c("Gender", "Age_at_Release", "Education_Level", 
          "Prior_Conviction_Episodes_Viol", "Prison_Offense", 
          "Prison_Years", "Gang_Affiliated", "Supervision_Level_First")  

# prior woe

prior_prob <- nb_model$prior
w0 <- log(prior_prob["TRUE"] / prior_prob["FALSE"])

# ebs dataframe 

ebs_data <- data.frame(feature = "Prior", woe = w0)

for(v in vars) {
  val <- as.character(target_parolee[[v]])
  woe_val <- woe_model(nb_model, v, val)
  ebs_data <- rbind(ebs_data, data.frame(feature = paste0(v, "=", val), woe = woe_val))
}

ebs_data$woe <- round(100 * ebs_data$woe)
print(ebs_data)



# final probability from total WoE

total_woe <- sum(ebs_data$woe)
prob_final <- 1/(1 + exp(-total_woe/100))
print(paste("Predicted Probability for ID", target_id, ":", round(prob_final,2)))

#table

ebs <- ebs_data |>
  arrange(feature != "Prior", desc(abs(woe)))   # keep Prior on top

posEvidence <- ebs |> filter(woe > 0)
negEvidence <- ebs |> filter(woe <= 0)

maxRows <- max(nrow(posEvidence), nrow(negEvidence))

tab <- data.frame(
  posVar = rep(NA_character_, maxRows+3),
  woeP   = rep(NA_real_, maxRows+3),
  negVar = rep(NA_character_, maxRows+3),
  woeN   = rep(NA_real_, maxRows+3)
)

total_pos <- sum(posEvidence$woe, na.rm = TRUE)  
total_neg <- sum(negEvidence$woe, na.rm = TRUE)  
total_woe_calc <- total_pos + total_neg          
prob_calc <- round(1 / (1 + exp(-total_woe_calc / 100)), 2) 

# 2. Filling table features and individual WoEs
tab[1:nrow(posEvidence), 1:2] <- posEvidence
tab[1:nrow(negEvidence), 3:4] <- negEvidence

# 3. Insert the summary rows using the variables 
tab[maxRows + 1, 1] <- "Total positive weight"
tab[maxRows + 1, 2] <- total_pos
tab[maxRows + 1, 3] <- "Total negative weight"
tab[maxRows + 1, 4] <- total_neg

tab[maxRows + 2, 3] <- "Total weight of evidence"
tab[maxRows + 2, 4] <- total_woe_calc           

tab[maxRows + 3, 3] <- "Probability ="
tab[maxRows + 3, 4] <- prob_calc                 

# 4. Clean formatting (NAs to empty strings)
tab[is.na(tab)] <- ""

# 5. Display with headers matching Balance Sheet.png
kbl(tab,
    col.names = c("Feature (Evidence For)", "WoE", 
                  "Feature (Evidence Against)", "WoE"),
    row.names = FALSE, align = "lrlr",
    caption = "Evidence Balance Sheet") |>
  kable_styling(bootstrap_options = c("striped", "condensed"), 
                full_width = FALSE) |>
  row_spec((maxRows+1):(maxRows+3), bold = TRUE)

str(tab)
print(total_woe)
p = 1/(1+exp(0.28))

# 2 
# Comparison by race
# white parolees 

roc_white <- roc((Recidivism_Within_3years == "true") ~ pNB, 
                 data = subset(dataRecid, Training_Sample == 0 & Race == "WHITE"))
roc_white$auc
# Area under the curve: 0.6809

# black parolees 
roc_black <- roc((Recidivism_Within_3years == "true") ~ pNB, 
                 data = subset(dataRecid, Training_Sample == 0 & Race == "BLACK"))
roc_black$auc
# Area under the curve: 0.6688

# Plotting to compare 

plot(roc_white, col = "steelblue", main = "ROC Comparison: White (Blue) vs Black (Pink)")
lines(roc_black, col = "pink")


# 2A: Discussion for second question 

# The NB classifier demonstrates a slight disparity in predictive performance 
# across racial groups, achieving an AUC of 0.6809 for White parolees compared to 
# 0.6688 for Black parolees. This gap indicates that the model is somewhat more effective at 
# correctly ranking risk for White individuals than for Black individuals, due to variability in 
# the underlying features.
# Even this slight difference is significant because the scores generated for Black parolees are 
# less reliable, leading to higher rates of misclassification. 
