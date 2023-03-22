# install necessary libraries for the analysis
.packages = c("GGally", 
              "ggcorrplot",
              "ggplot2",
              "rsq",
              "Metrics",
              "kableExtra",
              "caret",
              "caTools",
              "e1071",
              "tinytex",
              "randomForest",
              "dplyr"
)
# Install CRAN packages (if not already installed)
.inst <- .packages %in% installed.packages()
if(length(.packages[!.inst]) > 0) install.packages(.packages[!.inst])
# Load packages into session 
lapply(.packages, require, character.only=TRUE)

library(GGally) # for visualization
library(ggplot2) # for visualization
library(ggcorrplot) # for visualization
library(dplyr)
library(rsq) # for evaluation
library(Metrics) # for evaluation
library(caret)
library(e1071) # for SVR algo
library(randomForest) # for RF algo
library(caTools)
library(kableExtra) # display kable tables
library(tinytex)

# Data from the Kaggle and same uploaded to my GIT HUB
heartdataset <- read.csv("https://raw.githubusercontent.com/VamsiK51/CapStoneCYO/main/HeartDataset.csv")

# Removing X column from the dataset which is not relavant to dataset algorithm
heart_multi <- subset(heartdataset, select = -c(X))

# rename the column name from 'heart.disease' to 'heart_disease'
heart_multi <- rename(heart_multi, heart_disease = heart.disease)

# check the first 5 rows of the dataset
head(heart_multi, n=5)

# check the column names
colnames(heart_multi)

# check structure of the dataset
str(heart_multi) # here we have 498 observations with numeric datatype

# check missing values:
sum(is.na(heart_multi))
# there is no missing values in the data

# check summary
summary(heart_multi)

# check for outliers using boxplot
Q1 <- apply(heart_multi, 2, quantile, probs=0.25, na.rm=TRUE)
Q3 <- apply(heart_multi, 2, quantile, probs=0.75, na.rm=TRUE)
IQR <- Q3 - Q1

outliers <- apply(heart_multi, 2, function(x) 
  sum(x < (Q1 - (1.5 * IQR)) | x > (Q3 + (1.5 * IQR)), na.rm = TRUE))

outliers

# using boxplot method
boxplot(heart_multi)

# in this data set there are no outliers

# step 2: Visualizing the dataset

# create histogram
hist(heart_multi$biking,
     main = "Distribution of Biking",
     xlab = "Values of biking",
     ylab = "Frequency",
     xlim = range(0, 80),
     ylim = range(0, 100),
     labels = TRUE)


hist(heart_multi$smoking,
     main = "Distribution of Smoking",
     xlab = "Values of Smoking",
     ylab = "Frequency",
     xlim = range(0, 30),
     ylim = range(0, 140),
     labels = TRUE)


hist(heart_multi$heart_disease,
     main = "Distribution of Heart Disease",
     xlab = "Values of Heart Disease",
     ylab = "Frequency",
     xlim = range(0, 25),
     ylim = range(0, 90),
     labels = TRUE)


# plotting
ggpairs(heart_multi)

# Visualize the relationship between the features 
# and the response using scatterplots

# relationship with biking
ggplot(heart_multi, aes(x = biking, y = heart_disease)) + 
  geom_point() +
  labs(title = "Scatter plot of Biking vs. Heart Disease")


# relationship with Smoking
ggplot(heart_multi, aes(x = smoking, y = heart_disease)) + geom_point() +
  labs(title = "Scatter plot of Smoking vs. Heart Disease")

# create a correlation matrix
cor_matrix <- cor(heart_multi)
ggcorrplot(cor_matrix, lab = TRUE, lab_size = 4)

# rescale the features
normalize <- function(x) {
  return ((x - mean(x)) / (max(x) - min(x)))
}

heart_multi <- apply(heart_multi, 2, normalize)
heart_multi <- as.data.frame(heart_multi)

# Step 3: Split the data into train, test and validation 

# Set the seed for reproducibility
set.seed(100)

split <- sample.split(heart_multi, SplitRatio = 0.8) 
train_df <- subset(heart_multi, split == "TRUE") 
validation <- subset(heart_multi, split == "FALSE") 

split <- sample.split(train_df, SplitRatio = 0.8) 
train <- subset(train_df, split == "TRUE") 
test <- subset(train_df, split == "FALSE") 

dim(train) # dimension/shape of train dataset
print(head(train))   
dim(test)  # dimension/shape of test dataset
print(head(test))

# Step 4: Build Regression model

###########################################################################
# Multiple linear Regression
###########################################################################

# build linear regression model
model <- lm(heart_disease ~ biking + smoking, data = train)

# summary
summary(model) 

# confidence interval
confint(model)

# print intercept and coefficient
coefficients <- coef(model)
print(coefficients)
# From the above result we may infer that if smoking value 
# increases by 1 unit it will affect heart disease by 0.25 unit.

# Similarly, From the above result we may also infer that if biking value 
# increases by 1 unit it will affect heart disease by -0.74 unit.

# Step 5: Predictions
pred <- predict(model, test) 
# predictions are made on the testing data set using predict()

pred # predicted values 

# Step 6: Model evaluation
lm_rsq <- R2(pred, test$heart_disease)
print(paste("R-squared value for MLR:", 
            round(lm_rsq, 4))) 

lm_rmse <- sqrt(mean(pred-test$heart_disease)^2)
print(paste("RMSE value for MLR:", round(lm_rmse, 4)))  

# step 7: plot actual vs predicted
df <- data.frame(actual = test$heart_disease, predicted = pred)
ggplot(df, aes(x = actual, y = predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red") +
  labs(x = "Actual", y = "Predicted", 
       title = "Actual vs Predicted values of Linear Regression Model")

# calculate residuals
resid <- test$heart_disease - pred

# plot error terms
plot(resid, main = "Residuals vs Fitted Values", xlab = "Fitted Values", 
     ylab = "Residuals")

# plot distribution of residuals
hist(resid, main = "Distribution of Residuals", xlab = "Residuals")


# Store accuracy
accuracy_algorithm = tibble(Model = "Multiple linear regression", Accuracy = lm_rsq, RMSE = lm_rmse)


###########################################################################
# Support Vector Regression
###########################################################################

# load required package

# build support vector regression model
svm_model <- svm(heart_disease ~ biking + smoking, data = train, 
                 kernel = "linear")

# predict on test set
svm_pred <- predict(svm_model, newdata = test)

# Evaluate model
svm_rmse <- sqrt(mean(svm_pred-test$heart_disease)^2)
print(paste("RMSE value for SVM (regression) model:", 
            round(svm_rmse, 4)))  

# calculate R-squared using caret package
svm_rsq <- R2(svm_pred, test$heart_disease)
print(paste("R-Square value for SVM (regression):", 
            round(svm_rsq, 4)))  

accuracy_algorithm <- bind_rows(accuracy_algorithm,
                                tibble(Model = "Support Vector Regression",
                                       Accuracy = svm_rsq , RMSE = svm_rmse))

###########################################################################
# Random Forest
###########################################################################

library(randomForest)

# Specify the independent variables (biking and smoking) 
# and the dependent variable (heart_disease) for the model:
x_train <- train[, c("biking", "smoking")]
y_train <- train$heart_disease

# Build the random forest model using the randomForest function
set.seed(100)
# fit the random forest model
rf_model <- randomForest(heart_disease ~ biking + smoking, 
                         data = train, importance = TRUE, ntree = 500)

# predict on test data
rf_pred <- predict(rf_model, newdata = test)

# Evaluate the model
rf_rmse <- sqrt(mean(rf_pred-test$heart_disease)^2)
print(paste("RMSE value for Random Forest:", 
            round(rf_rmse, 4)))  

# calculate R-squared using caret package
rf_rsq <- R2(rf_pred, test$heart_disease)
print(paste("R-Square value for Random Forest:", 
            round(rf_rsq, 4))) 

accuracy_algorithm <- bind_rows(accuracy_algorithm,
                                tibble(Model = "Random Forest",
                                       Accuracy = rf_rsq , RMSE = rf_rmse))

###############################################
# compare models
###############################################

models <- c("Multiple linear regression", 
            "Support Vector Regression", 
            "Random Forest")

RMSE <- c(round(lm_rmse, 4), round(svm_rmse, 4), round(rf_rmse, 4))
R_squared <- c(round(lm_rsq, 4), round(svm_rsq, 4), round(rf_rsq, 4))

comparison_df <- data.frame(models, RMSE, R_squared)
comparison_df

# visualizing the comparison of models and BAR chart for RMSE
ggplot(comparison_df, aes(x = models, y = RMSE)) +
  geom_bar(stat = "identity", fill = "lightgreen") +
  geom_text(aes(label = round(RMSE, 4)), 
            position = position_dodge(width = 0.9),
            vjust = -0.5) +
  labs(title = "Comparison of Regression models",
       x = "Models",
       y = "RMSE")

# Comparison of Regression models and BAR chart for R Square prediction

ggplot(comparison_df, aes(x = models, y = R_squared)) +
  geom_bar(stat = "identity", fill = "lightgreen") +
  geom_text(aes(label = round(R_squared, 4)),
            position = position_stack(vjust = 0.9),
            size = 4) +
  labs(title = "Comparison of Regression models",
       x = "Models",
       y = "R-squared")


#####################################################

# since the SVM model is giving better result, we can check & implement SVM for 
#predicting on validation data. Also it will help to check if the model is over fitting or not

# predict on test set
svm_pred <- predict(svm_model, newdata = validation)

# Evaluate model
rmse_val <- sqrt(mean(svm_pred-validation$heart_disease)^2)
print(paste("RMSE value for SVM (regression) model:", 
            round(svm_rmse, 4)))  

# calculate R-squared using caret package
svm_rsq <- R2(svm_pred, validation$heart_disease)
print(paste("R-Square value for SVM (regression):", 
            round(svm_rsq, 4)))  

################################
# Conclusion
################################

accuracy_algorithm %>%
  arrange(desc(Accuracy)) %>%
  knitr::kable() %>%
  kable_styling()

# Support Vector Regression RMSE AND R SQAURE VALUES are with in range and 
# hence model is not overfitted and we can this model for the given dataset
