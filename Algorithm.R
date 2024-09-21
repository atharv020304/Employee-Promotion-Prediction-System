
data <- read.csv("Dataset.csv")
library(ROSE)
library(randomForest)
library(caret)
library(e1071)
library(neuralnet)
library(rpart)
library(dplyr)
library(xgboost)
library(irr)

########################## Random forest #####################

cat("\n                  RESULTS FOR RANDOM FOREST")
cat("\n")



library(ROSE)
library(randomForest)
library(caret)
library(e1071)

sample <- read.csv("Dataset.csv")

sample$is_promoted <- as.factor(sample$is_promoted)

set.seed(123)

ind <- sample(2, nrow(sample), replace = TRUE, prob = c(0.7, 0.3))
train <- sample[ind == 1, ]
test <- sample[ind == 2, ]

cat("Class Distribution for Original Data:\n")
table(sample$is_promoted)

rf_train <- randomForest(is_promoted ~ ., data = train)

confusion_matrix_original <- confusionMatrix(predict(rf_train, newdata = test), test$is_promoted, positive = '1')

cat("\nOriginal Model Metrics:\n")
print(confusion_matrix_original)

over_sampled_data <- ovun.sample(is_promoted ~ ., data = train, method = "over", N = 70600)$data

cat("\nClass Distribution for Oversampled Data:\n")
table(over_sampled_data$is_promoted)

rf_over <- randomForest(is_promoted ~ ., data = over_sampled_data, ntree = 100, nodesize = 2)

confusion_matrix_oversampled <- confusionMatrix(predict(rf_over, newdata = test), test$is_promoted, positive = '1')

cat("\nOversampled Model Metrics:\n")
print(confusion_matrix_oversampled)

precision_oversampled <- confusion_matrix_oversampled$byClass['Precision']
f1_score_oversampled <- confusion_matrix_oversampled$byClass['F1']
recall_oversampled <- confusion_matrix_oversampled$byClass['Sensitivity']

cat("Oversampled Model Precision:", precision_oversampled, "\n")
cat("Oversampled Model F1 Score:", f1_score_oversampled, "\n")
cat("Oversampled Model Recall:", recall_oversampled, "\n")

k <- 5
folds <- createFolds(train$is_promoted, k = k)

accuracy_values <- vector("numeric", length = k)
precision_values <- vector("numeric", length = k)
recall_values <- vector("numeric", length = k)
f1_score_values <- vector("numeric", length = k)
specificity_values <- vector("numeric", length = k)
sensitivity_values <- vector("numeric", length = k)

for (i in 1:k) {
  validation_indices <- folds[[i]]
  train_fold <- train[-validation_indices, ]
  validation_fold <- train[validation_indices, ]
  
  rf_fold <- randomForest(is_promoted ~ ., data = train_fold)
  
  rf_predictions <- predict(rf_fold, newdata = validation_fold)
  
  confusion_matrix_fold <- confusionMatrix(rf_predictions, validation_fold$is_promoted, positive = '1')
  accuracy_values[i] <- confusion_matrix_fold$overall['Accuracy']
  precision_values[i] <- confusion_matrix_fold$byClass['Precision']
  recall_values[i] <- confusion_matrix_fold$byClass['Sensitivity']
  f1_score_values[i] <- confusion_matrix_fold$byClass['F1']
  specificity_values[i] <- confusion_matrix_fold$byClass['Specificity']
  sensitivity_values[i] <- confusion_matrix_fold$byClass['Sensitivity']
}

mean_accuracy <- mean(accuracy_values)
mean_precision <- mean(precision_values)
mean_recall <- mean(recall_values)
mean_f1_score <- mean(f1_score_values)
mean_specificity <- mean(specificity_values)
mean_sensitivity <- mean(sensitivity_values)

cat("\nMean Cross-validated Metrics:\n")
cat("Mean Accuracy:", mean_accuracy, "\n")
cat("Mean Precision:", mean_precision, "\n")
cat("Mean Recall:", mean_recall, "\n")
cat("Mean F1 Score:", mean_f1_score, "\n")
cat("Mean Specificity:", mean_specificity, "\n")
cat("Mean Sensitivity:", mean_sensitivity, "\n")




######################## SVM ##########################
cat("\n                      RESULTS FOR SVM")
cat("\n")

set.seed(42)
data_size <- floor(0.8 * nrow(data))
train_indices <- sample(1:nrow(data), size = data_size)  
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

oversampled_data <- ovun.sample(is_promoted ~ ., data = train_data, method = "over", 
                                N = 70600, seed = 123)$data

k <- 5

folds <- createFolds(oversampled_data$is_promoted, k = k)

accuracy_values <- vector("numeric", length = k)
precision_values <- vector("numeric", length = k)
recall_values <- vector("numeric", length = k)
f1_score_values <- vector("numeric", length = k)
specificity_values <- vector("numeric", length = k)
sensitivity_values <- vector("numeric", length = k)

for (i in 1:k) {
  
  validation_indices <- folds[[i]]
  train_fold <- oversampled_data[-validation_indices, ]
  validation_fold <- oversampled_data[validation_indices, ]
  
  svm_model <- svm(is_promoted ~ ., data = train_fold, type = "C-classification", kernel = "linear", cost = 1, gamma = 0.1, scale = TRUE)
  
  svm_predictions <- predict(svm_model, validation_fold)
  
  confusion_matrix <- table(Actual = validation_fold$is_promoted, Predicted = svm_predictions)
  accuracy_values[i] <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
  precision_values[i] <- confusion_matrix[2, 2] / sum(confusion_matrix[, 2])
  recall_values[i] <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
  f1_score_values[i] <- 2 * (precision_values[i] * recall_values[i]) / (precision_values[i] + recall_values[i])
  specificity_values[i] <- confusion_matrix[1, 1] / sum(confusion_matrix[1, ])
  sensitivity_values[i] <- recall_values[i]
}

mean_accuracy <- mean(accuracy_values)
mean_precision <- mean(precision_values)
mean_recall <- mean(recall_values)
mean_f1_score <- mean(f1_score_values)
mean_specificity <- mean(specificity_values)
mean_sensitivity <- mean(sensitivity_values)

cat("\nMEAN METRICS ACROSS ALL FOLDS\n")
cat("Mean Accuracy:", mean_accuracy, "\n")
cat("Mean Precision:", mean_precision, "\n")
cat("Mean Recall:", mean_recall, "\n")
cat("Mean F1 Score:", mean_f1_score, "\n")
cat("Mean Specificity:", mean_specificity, "\n")
cat("Mean Sensitivity:", mean_sensitivity, "\n")

############################## ANN ##################################
cat("\n                     RESULTS FOR ANN")
cat("\n ")

data$is_promoted <- factor(data$is_promoted, levels = c("0", "1"))


preProcessDesc <- preProcess(data[, c("previous_year_rating", "length_of_service")], method = c("center", "scale"))
data[, c("previous_year_rating", "length_of_service")] <- predict(preProcessDesc, data[, c("previous_year_rating", "length_of_service")])


formula <- as.formula("is_promoted ~ previous_year_rating + length_of_service")


set.seed(123)
ind <- sample(2, nrow(data), replace = TRUE, prob = c(0.7, 0.3))
train_data <- data[ind == 1, ]
test_data <- data[ind == 2, ]


ctrl <- trainControl(method = "cv", 
                     number = 5,     
                     verboseIter = TRUE)


tuning_grid <- expand.grid(size = c(15, 20, 25), decay = c(0.01, 0.001, 0.0001))


oversampled_train_data <- ovun.sample(is_promoted ~ ., data = train_data, method = "over", N = 70600, seed = 123)$data


set.seed(123)
sink("training_output.txt")
model <- train(formula,
               data = oversampled_train_data,  
               method = "nnet",               
               trControl = ctrl,
               preProcess = c("center", "scale"),
               tuneGrid = tuning_grid,         
               linout = FALSE,                 
               lifesign = "full",
               stepmax = 100000)
sink()


conf_matrix <- confusionMatrix(predict(model, newdata = test_data), test_data$is_promoted)
cat("\nConfusion Matrix:\n")
print(conf_matrix$table)

accuracy <- conf_matrix$overall["Accuracy"]
precision <- conf_matrix$byClass["Precision"]
recall <- conf_matrix$byClass["Sensitivity"]
f1 <- conf_matrix$byClass["F1"]
sensitivity <- conf_matrix$byClass["Sensitivity"]
specificity <- conf_matrix$byClass["Specificity"]

cat("\nMEAN METRICS ACROSS ALL FOLDS\n")
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1, "\n")
cat("Sensitivity:", sensitivity, "\n")
cat("Specificity:", specificity, "\n")

###########################decision tree###############################
cat("\n")
cat("\n                        RESULTS FOR DECISION TREE")
cat("\n")
set.seed(123)
data$is_promoted <- as.factor(data$is_promoted)
k <- 5
results <- list()

for (i in 1:k) {
  folds <- createFolds(data$is_promoted, k = k, list = TRUE)
  train_indices <- unlist(folds[-i])
  test_indices <- unlist(folds[i])
  train_data <- data[train_indices, ]
  test_data <- data[test_indices, ]
  
  oversampled_data <- ovun.sample(is_promoted ~ ., data = train_data, method = "over", N = 70600)$data
  
  dt_model_after_oversampling <- rpart(is_promoted ~ ., data = oversampled_data, method = "class", control = rpart.control(minsplit = 10, minbucket = 5))
  
  dt_predictions_after_oversampling <- as.factor(predict(dt_model_after_oversampling, test_data, type = "class"))
  
  dt_confusion_matrix_after_oversampling <- confusionMatrix(data = dt_predictions_after_oversampling, reference = test_data$is_promoted)
  
  results[[i]] <- dt_confusion_matrix_after_oversampling
}
cat("\nConfusion Matrix:\n")
print(dt_confusion_matrix_after_oversampling$table)

mean_accuracy <- mean(sapply(results, function(result) result$overall["Accuracy"]))
mean_precision <- mean(sapply(results, function(result) result$byClass["Precision"]))
mean_f1_score <- mean(sapply(results, function(result) {
  precision <- result$byClass["Precision"]
  recall <- result$byClass["Sensitivity"]
  2 * (precision * recall) / (precision + recall)
}))
mean_recall <- mean(sapply(results, function(result) result$byClass["Sensitivity"]))

cat("\nMEAN METRICS ACROSS ALL FOLDS\n")
cat("Mean Accuracy: ", mean_accuracy, "\n")
cat("Mean Precision: ", mean_precision, "\n")
cat("Mean F1 Score: ", mean_f1_score, "\n")
cat("Mean Recall: ", mean_recall, "\n")

#######################Logistics Regression #######################################
cat("\n")
cat("\n                       RESULTS FOR LOGISTICS REGRESSION")
cat("\n")

char_columns <- c("department", "education", "gender", "recruitment_channel", "region")
for (col in char_columns) {
  data[[col]] <- as.numeric(factor(data[[col]], levels = unique(data[[col]])))
}

num_folds <- 5

fold_metrics_before <- list()

oversample_target <- 70600

oversampled_data <- ROSE(is_promoted ~ ., data = data, N = oversample_target)$data

fold_metrics_after <- list()

for (fold in 1:num_folds) {
  train_indices <- createDataPartition(oversampled_data$is_promoted, p = 0.75, list = FALSE)
  train_data <- oversampled_data[train_indices, ]
  test_data <- oversampled_data[-train_indices, ]
  
  logistic_model <- glm(is_promoted ~ ., data = train_data, family = "binomial")
  
  predictions <- predict(logistic_model, newdata = test_data, type = "response")
  threshold <- 0.5
  binary_predictions <- ifelse(predictions > threshold, 1, 0)
  
  confusion_matrix <- table(Actual = test_data$is_promoted, Predicted = binary_predictions)
  
  TP <- confusion_matrix[2, 2]
  TN <- confusion_matrix[1, 1]
  FP <- confusion_matrix[1, 2]
  FN <- confusion_matrix[2, 1]
  
  accuracy <- (TP + TN) / sum(confusion_matrix)
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  f1_score <- 2 * (precision * recall) / (precision + recall)
  
  fold_metrics_after[[fold]] <- list(
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    f1_score = f1_score
  )
}
cat("\nConfusion Matrix:\n")
print(confusion_matrix)
avg_metrics_after <- sapply(fold_metrics_after, function(fold) c(fold$accuracy, fold$precision, fold$recall, fold$f1_score))
avg_metrics_after <- colMeans(avg_metrics_after)

cat("\nMEAN METRICS ACROSS ALL FOLDS\n")
cat("Average Accuracy:", avg_metrics_after[1], "\n")
cat("Average Precision:", avg_metrics_after[2], "\n")
cat("Average Recall:", avg_metrics_after[3], "\n")
cat("Average F1 Score:", avg_metrics_after[4], "\n")

######################### xgboost #####################################
cat("\n                     RESULTS FOR XGBoost")
cat("\n")

character_columns_to_convert <- c("employee_id", "department", "region", "education", "gender", "recruitment_channel")

data[character_columns_to_convert] <- lapply(data[character_columns_to_convert], as.factor)
data[character_columns_to_convert] <- lapply(data[character_columns_to_convert], as.numeric)

set.seed(123)

xgb_params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.01,
  max_depth = 6,
  nrounds = 0,
  early_stopping_rounds = 0
)

response_variable <- "is_promoted"

calculate_metrics <- function(predictions, actual) {
  accuracy <- sum(predictions == actual) / length(actual)
  conf_matrix <- table(Actual = actual, Predicted = predictions)
  precision <- posPredValue(conf_matrix)
  recall <- sensitivity(conf_matrix)
  f1_score <- 2 * (precision * recall) / (precision + recall)
  kappa_value <- kappa2(as.matrix(conf_matrix))
  return(list(accuracy = accuracy, precision = precision, recall = recall, 
              f1_score = f1_score, kappa = kappa_value, conf_matrix = conf_matrix))
}

train_model_and_evaluate <- function(train_data, test_data, xgb_params, response_variable) {
  x_train <- as.matrix(train_data[, !names(train_data) %in% response_variable])
  y_train <- as.numeric(train_data[, response_variable])
  x_test <- as.matrix(test_data[, !names(test_data) %in% response_variable])
  y_test <- as.numeric(test_data[, response_variable])
  dtrain <- xgb.DMatrix(data = x_train, label = y_train)
  xgb_model <- xgboost(data = dtrain, params = xgb_params, nthread = -1, nrounds = xgb_params$nrounds, verbose = 0)
  dtest <- xgb.DMatrix(data = x_test)
  xgb_predictions <- predict(xgb_model, newdata = dtest)
  xgb_predictions_binary <- ifelse(xgb_predictions > 0.5, 1, 0)
  metrics <- calculate_metrics(xgb_predictions_binary, y_test)
  return(metrics)
}

num_folds <- 5
fold_metrics <- list()

mean_confusion_matrix <- matrix(0, nrow = 2, ncol = 2)
mean_accuracy <- 0
mean_precision <- 0
mean_recall <- 0
mean_f1_score <- 0

for (fold in 1:num_folds) {
  fold_indices <- createDataPartition(data$'is_promoted', p = 0.7, list = FALSE)
  train_data <- data[fold_indices, ]
  test_data <- data[-fold_indices, ]
  oversampled_data <- ovun.sample(is_promoted ~ ., data = train_data, method = "over", N = 70600)$data
  fold_metrics[[fold]] <- train_model_and_evaluate(oversampled_data, test_data, xgb_params, response_variable)
  
  mean_confusion_matrix <- mean_confusion_matrix + fold_metrics[[fold]]$conf_matrix
  
  mean_accuracy <- mean_accuracy + fold_metrics[[fold]]$accuracy
  mean_precision <- mean_precision + fold_metrics[[fold]]$precision
  mean_recall <- mean_recall + fold_metrics[[fold]]$recall
  mean_f1_score <- mean_f1_score + fold_metrics[[fold]]$f1_score
}

mean_confusion_matrix <- mean_confusion_matrix / num_folds
cat("\nConfusion Matrix:\n")
print(mean_confusion_matrix)

mean_accuracy <- mean_accuracy / num_folds
mean_precision <- mean_precision / num_folds
mean_recall <- mean_recall / num_folds
mean_f1_score <- mean_f1_score / num_folds

cat("\nMEAN METRICS ACROSS ALL FOLDS\n")
cat("Mean Accuracy:", mean_accuracy, "\n")
cat("Mean Precision:", mean_precision, "\n")
cat("Mean Recall:", mean_recall, "\n")
cat("Mean F1 Score:", mean_f1_score, "\n")
