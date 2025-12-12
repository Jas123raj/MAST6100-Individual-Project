# Load necessary libraries
library(tidyverse)
library(patchwork)
library(caret)
library(randomForest)
library(xgboost)
library(pROC)
library(keras)

# Load and prepare dataset
df <- read.csv("Final dataset Attrition.csv", stringsAsFactors = TRUE)

# Ensure the target variable is a factor which is Attrition and set to "No" as a reference level
df$Attrition <- as.factor(df$Attrition)
df$Attrition <- relevel(df$Attrition, ref = "No")

#Check on Target variables
str(df$Attrition)
table(df$Attrition)



# Fit GLM (logistic regression) model
glm_model_full <- glm(
  Attrition ~ Age + DistanceFromHome +
    JobInvolvement + JobLevel + JobSatisfaction + MaritalStatus + MonthlyIncome +
    NumCompaniesWorked + OverTime + TotalWorkingYears + YearsAtCompany +
    YearsSinceLastPromotion + YearsWithCurrManager,
  data = df,
  family = binomial(link = "logit")
)

# Summary of the model
summary(glm_model_full)

# Extract fitted values and residuals
df$fitted <- fitted(glm_model_full)
df$residuals <- residuals(glm_model_full, type = "deviance")

# Plot Residuals vs Fitted
ggplot(df, aes(x = fitted, y = residuals)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "loess", se = FALSE, color = "blue") +
  labs(
    title = "Residuals vs Fitted Values (GLM - Logistic Regression)",
    x = "Fitted Values (Predicted Probability of Attrition)",
    y = "Residuals (Deviance)"
  ) +
  theme_minimal()


#Effect of Age on predicted Attrition(GLM)
x_var <- "Age"
#Creates a sequence of Age values across its range
x_seq <- seq(min(df[[x_var]], na.rm = TRUE),
             max(df[[x_var]], na.rm = TRUE),
             length.out = 300)
#Builds a profile with medians of common categorises
newdata <- df %>%
  summarise(
    DistanceFromHome = median(DistanceFromHome, na.rm = TRUE),
    JobInvolvement = median(JobInvolvement, na.rm = TRUE),
    JobLevel = median(JobLevel, na.rm = TRUE),
    JobSatisfaction = median(JobSatisfaction, na.rm = TRUE),
    MaritalStatus = names(sort(table(MaritalStatus), decreasing = TRUE))[1],
    MonthlyIncome = median(MonthlyIncome, na.rm = TRUE),
    NumCompaniesWorked = median(NumCompaniesWorked, na.rm = TRUE),
    OverTime = names(sort(table(OverTime),decreasing = TRUE))[1],
    TotalWorkingYears = median(TotalWorkingYears, na.rm = TRUE),
    YearsAtCompany = median(YearsAtCompany, na.rm = TRUE),
    YearsSinceLastPromotion = median(YearsSinceLastPromotion, na.rm = TRUE),
    YearsWithCurrManager = median(YearsWithCurrManager, na.rm = TRUE)
  ) %>%
  slice(rep(1,length(x_seq))) %>%
  mutate(!!x_var := x_seq)
#Stores predicted probabilities of attrition for each Age value
newdata$pred_prob <- predict(glm_model, newdata = newdata, type = "response")

#Plots fitted probabilities against Age
ggplot() + 
  geom_point(data = df, aes_string(x = x_var, y = "fitted"),
                                   alpha = 0.4, color = "black") + 
  geom_line(data = newdata, aes_string(x = x_var, y = "pred_prob"),
            color = "blue", size = 1.2) + 
  labs(
    title = paste("GLM Predicted Probability of Attrition vs", x_var),
    x = x_var,
    y = "Predicted Probability (from GLM)"
  ) + 
  theme_minimal()


#Effect of Monthly Income

#Converts Attrition to 0/1 to be set up for plotting
df$Attrition01 <- ifelse(df$Attrition == "Yes", 1, 0)

ggplot(df, aes(x = MonthlyIncome, y = Attrition01)) + 
  geom_point(alpha = 0.6) + 
  geom_smooth(
    method = "glm", 
    method.args = list(family = binomial(link = "logit")),
    se = TRUE
  ) + 
  scale_y_continuous(limits = c(0,1)) +
  labs(
    title = "Effect of Monthly Income on Probability of Attrition",
    x = "Monthly Income",
    y = "Probability of Attrition"
  ) + 
  theme_minimal()

#Effect of Distance from Home 
#Sequence of Distances 
dist_seq <- seq(min(df$DistanceFromHome),
                max(df$DistanceFromHome),
                length.out = 100)

#Build employee and vary DistanceFromHome
new_dist <- data.frame(
  DistanceFromHome = dist_seq,
  TotalWorkingYears = median(df$TotalWorkingYears, na.rm = TRUE),
  Age = median(df$Age, na.rm = TRUE),
  JobInvolvement = median(df$JobInvolvement, na.rm = TRUE),
  JobLevel = median(df$JobLevel, na.rm = TRUE),
  JobSatisfaction = median(df$JobSatisfaction, na.rm = TRUE),
  MaritalStatus = names(sort(table(df$MaritalStatus), decreasing = TRUE))[1],
  MonthlyIncome = median(df$MonthlyIncome, na.rm = TRUE),
  NumCompaniesWorked = median(df$NumCompaniesWorked, na.rm = TRUE),
  OverTime = names(sort(table(df$OverTime), decreasing = TRUE))[1],
  YearsAtCompany = median(df$YearsAtCompany, na.rm = TRUE), 
  YearsSinceLastPromotion = median(df$YearsSinceLastPromotion, na.rm = TRUE),
  YearsWithCurrManager = median(df$YearsWithCurrManager, na.rm = TRUE)
)

#Predicted Probability vs DistanceFromHome
new_dist$pred_prob <- predict(glm_model_full, newdata = new_dist, type = "response")

ggplot(new_dist, aes(x = DistanceFromHome, y = pred_prob)) + 
  geom_line(size = 1) + 
  labs(
    title = "Predicted Probability of Attrition Vs Distance From Home",
    x = "Distance From Home", 
    y = "Predicted Probability of Attrition"
  ) + 
  theme_minimal()

#Effect of Total Working Years 
twy_seq <- seq(min(df$TotalWorkingYears),
               max(df$TotalWorkingYears),
               length.out = 100)

new_twy <- new_dist
new_twy$TotalWorkingYears <- twy_seq

new_twy$pred_prob<- predict(glm_model_full, newdata = new_twy, type = "response")

ggplot( new_twy, aes(x = TotalWorkingYears, y = pred_prob)) + 
  geom_line(size = 1) + 
  labs(
    title = "Predicted Probability of Attrition vs Total Working Years",
    x = "Total Working Years",
    y = "Predicted Probability of Attrition"
  ) + 
  theme_minimal()

# GLM Curve function that plots both age and monthly income 
glm_curve <- function(var){
  
  x_seq <- seq(
    min(df[[var]], na.rm = TRUE),
    max(df[[var]], na.rm = TRUE),
    length.out = 300
    )
  
  base <- df %>% 
    summarise(
      Age = median(Age),
      DistanceFromHome = median(DistanceFromHome),
      JobInvolvement = median(JobInvolvement),
      JobLevel = median(JobLevel),
      JobSatisfaction = median(JobSatisfaction),
      MaritalStatus = names(sort(table(MaritalStatus), decreasing = TRUE))[1],
      MonthlyIncome = median(MonthlyIncome),
      NumCompaniesWorked = median(NumCompaniesWorked),
      OverTime = names(sort(table(OverTime), decreasing = TRUE))[1],
      TotalWorkingYears = median(TotalWorkingYears),
      YearsAtCompany = median(YearsAtCompany),
      YearsSinceLastPromotion = median(YearsSinceLastPromotion),
      YearsWithCurrManager = median(YearsWithCurrManager)
    )
  newdata <- base[rep(1, length(x_seq)), ]
  newdata[[var]] <- x_seq
  
  newdata$pred <- predict(glm_model, newdata, type = "response")
  
  ggplot(newdata, aes(x = .data[[var]], y = pred)) + 
    geom_line(color = "blue", size = 1.2) + 
    geom_ribbon(aes(ymin = pred -0.03, ymax = pred + 0.03),
                alpha = 0.1, fill = "blue") + 
    labs(
      title = paste("Effect of", var, "on Attrition (GLM)"),
      x = var,
      y = "Predicted Probability of Attrition"
    ) +
    theme_minimal()
}

p_age <- glm_curve("Age")
p_income <- glm_curve("MonthlyIncome")

p_age + p_income

#Train/ test split for machine learning models
set.seed(123)
idx <- createDataPartition(df$Attrition, p = 0.7, list = FALSE)
train <- df[idx, ]
test <- df[-idx, ]

#Sets up cross-validation control
ctrl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 2,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)


#GLM 
set.seed(123)
glm_fit <- train (
  Attrition ~ Age + DistanceFromHome + JobInvolvement + JobLevel + 
    JobSatisfaction + MaritalStatus + MonthlyIncome + 
    NumCompaniesWorked + OverTime + TotalWorkingYears + 
    YearsAtCompany + YearsSinceLastPromotion + YearsWithCurrManager,
  data = train,
  method = "glm", 
  family = binomial, 
  metric = "ROC",
  trControl = ctrl
)

glm.fit

glm_prob <- predict(glm_fit, newdata = test, type = "prob")[, "Yes"]
glm_class <- ifelse(glm_prob > 0.5, "Yes", "No") %>% factor(levels = c("No", "Yes"))

confusionMatrix(glm_class, test$Attrition)
roc_glm <- roc(test$Attrition, glm_prob); auc(roc_glm)


# Random Forest Model
set.seed(123)
rf_fit <- train(
  Attrition ~ Age + DistanceFromHome + JobInvolvement + JobLevel + 
    JobSatisfaction + MaritalStatus + MonthlyIncome + 
    NumCompaniesWorked + OverTime + TotalWorkingYears + 
    YearsAtCompany + YearsSinceLastPromotion + YearsWithCurrManager,
  data = train,
  method = "rf",
  metric = "ROC", 
  trControl = ctrl
)

rf_fit

rf_imp <- varImp(rf_fit)$importance

rf_imp_df <- rf_imp %>%
  rownames_to_column(var = "Variable") %>%
  arrange(desc(Overall))

print(rf_imp_df)

#Plots variable importance
ggplot(rf_imp_df, aes(x = reorder(Variable, Overall), y = Overall)) + 
  geom_bar(stat = "identity", fill = "purple") + 
  coord_flip() + 
  labs(
    title = "Random Forest Variable Importance (Attrition)",
    x = "Predictor",
    y = "Importance (Gini Decrease)"
  ) + 
  theme_minimal(base_size = 14)

rf_prob <- predict(rf_fit, newdata = test, type = "prob")[, "Yes"]
rf_class <- ifelse(rf_prob > 0.5, "Yes", "No") %>% factor(levels = c("No", "Yes"))

confusionMatrix(rf_class, test$Attrition)
roc_rf <- roc(test$Attrition, rf_prob); auc(roc_rf)



#XGBoost Model 
set.seed(123)

xgb_fit <- train(
  Attrition ~ Age + DistanceFromHome + JobInvolvement + JobLevel + 
    JobSatisfaction + MaritalStatus + MonthlyIncome + 
    NumCompaniesWorked + OverTime + TotalWorkingYears + 
    YearsAtCompany + YearsSinceLastPromotion + YearsWithCurrManager,
  data = train,
  method = "xgbTree",
  metric = "ROC",
  trControl = ctrl
)

xgb_fit

xgb_prob <- predict(xgb_fit, newdata = test, type = "prob")[, "Yes"]
xgb_class <- ifelse(xgb_prob > 0.5, "Yes", "No") %>% factor(levels = c("No", "Yes"))

confusionMatrix(xgb_class, test$Attrition)
roc_xgb <- roc(test$Attrition, xgb_prob); auc(roc_xgb)

#Deep Learning model via Keras Neural Network 
#Prepares design matrices for Keras
x_train <- model.matrix(Attrition ~ Age + DistanceFromHome + JobInvolvement + 
                            JobLevel + JobSatisfaction + MaritalStatus + 
                            MonthlyIncome + NumCompaniesWorked + OverTime +
                            TotalWorkingYears + YearsAtCompany +
                            YearsSinceLastPromotion + YearsWithCurrManager,
                          data = train)[, -1]

x_test <- model.matrix(Attrition ~ Age + DistanceFromHome + JobInvolvement + 
                         JobLevel + JobSatisfaction + MaritalStatus + 
                         MonthlyIncome + NumCompaniesWorked + OverTime +
                         TotalWorkingYears + YearsAtCompany +
                         YearsSinceLastPromotion + YearsWithCurrManager,
                       data = test)[, -1]

#Binary targets for Keras
y_train <- ifelse(train$Attrition == "Yes", 1, 0)
y_test <- ifelse(test$Attirtion == "Yes", 1, 0)

#Standardise predictors
x_train <- scale(x_train)
x_test <- scale(x_test,
                center = attr(x_train, "scaled:center"),
                scale = attr(x_train, "scaled:scale"))

input_dim <- ncol(x_train)

#Defines neural network architecture
keras_model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = input_dim) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 1, activation = "sigmoid")

#Compiles model
keras_model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics = c("accuracy")
)

#Train neural network
history <- keras_model %>% fit(
  x = x_train, y = y_train,
  epochs = 30,
  batch_size = 32, 
  validation_split = 0.1,
  verbose = 0
)

#Plots training vs validation loss to check for overfitting
plot(history$metrics$loss, type = "l",
     xlab= "Epoch", ylab = "Loss", main = "Training vs Validation Loss")
lines(history$metrics$val_loss, col = "red")
legend("topright", legend = c("Train loss", "Val loss"),
       col = c("black", "red"), lty = 1, bty = "n")

#Predictions and perfromance on test data set
d1_prob <- keras_model %>% predict(x_test) %>% as.vector()
d1_class <- ifelse(d1_prob > 0.5, "Yes", "No") %>%
  factor(levels = c("No", "Yes"))

confusionMatrix(d1_class, test$Attrition)
roc_d1 <- roc(test$Attrition, d1_prob); auc(roc_d1)

#Compares models by AUC on a ROC curve plot
data.frame(
  Model = c("GLM", "Random Forest", "XGBoost","Deep Learning"),
  AUC = c(auc(roc_glm), auc(roc_rf), auc(roc_xgb), auc(roc_d1))
) %>%
  arrange(desc(AUC))


roc_list <- list(
  GLM           = roc_glm,
  RandomForest  = roc_rf,
  XGBoost       = roc_xgb,
  DeepLearning  = roc_d1
)

ggroc(roc_list) +
  labs(
    title = "ROC Curves for All Models",
    x = "1 - Specificity",
    y = "Sensitivity"
  ) +
  theme_minimal()








