#=================================================================
#load libraries 
#=================================================================
library(readr) 
library(mlbench)
library(caret)
library(tidyverse)
library(ggplot2)
library(doSNOW)
library(gbm)
library(C50)


#=================================================================
#load data 
#=================================================================
m <- file.choose()
df_pb <-read.csv(m)
View(df_pb)


#=================================================================
#Preprocessing 
#=================================================================
is.na(df_pb)
attributes(df_pb)
summary(df_pb) 
str(df_pb)
names(df_pb)
sum(is.na(df_pb)) #no na 


#=================================================================
#finding correlation 
#=================================================================
correlationMatrix <- cor(df_pb)
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
# print indexes of highly correlated attributes
print(highlyCorrelated)


#=================================================================
#Rank Feature by importance 
#=================================================================
set.seed(7)
# prepare training scheme
control1 <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model1 <- train(brand~., data=df_pb, method="rf", preProcess="scale", trControl=control1)
# estimate variable importance
importance_bp <- varImp(model1, scale=FALSE)
# summarize importance
print(importance_bp)
# plot importance
plot(importance_bp)


#=================================================================
#converting brand to factor 
#=================================================================
df_pb$brand<- as.factor(df_pb$brand) 

#=================================================================
# plotting using ggplot2
#=================================================================
ggplot(data = df_pb) + 
  geom_point(mapping = aes(x = brand, y = salary))
ggplot(data = df_pb) + 
  geom_point(mapping = aes(x = brand, y = age))
ggplot(data = df_pb) + 
  geom_bar(mapping = aes(x = brand))
ggplot(data = df_pb, mapping = aes(x = brand, y = age)) + 
  geom_boxplot()
ggplot(data = df_pb, mapping = aes(x = brand, y = salary)) + 
  geom_boxplot()
ggplot(data = df_pb, mapping = aes(x = brand, y = zipcode)) + 
  geom_boxplot()
ggplot(data = df_pb) +
  geom_histogram(mapping = aes(x = salary), binwidth = 0.5)
#=================================================================
#partition 75% and 25%
#=================================================================
bp_intraining <- createDataPartition(df_pb$brand, p = .75, list = FALSE)
bp_training <- df_pb[bp_intraining,]
bp_testing <- df_pb[-bp_intraining,]
View(bp_testing)
View(bp_training)
#=================================================================
#Examine the proportions of the brand
#=================================================================
prop.table(table(df_pb$brand))
prop.table(table(bp_training$brand))
prop.table(table(bp_testing$brand))

#=================================================================
# Train Model
#=================================================================

train.control <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 3,
                              search = "grid")

#=================================================================
#Manual Tune Grid 
#=================================================================
tune.grid <- expand.grid(eta = c(0.05, 0.075, 0.1),
                         nrounds = c(50, 75, 100),
                         max_depth = 6:8,
                         min_child_weight = c(2.0, 2.25, 2.5),
                         colsample_bytree = c(0.3, 0.4, 0.5),
                         gamma = 0,
                         subsample = 1)
View(tune.grid)


# Use the doSNOW package to enable caret to train in parallel.
cl <- makeCluster(2, type = "SOCK")

# Register cluster so that caret will know to train in parallel.
registerDoSNOW(cl)

# Train the xgboost model using 10-fold CV repeated 3 times 
#Both xgboost and gbm follows the principle of gradient boosting. There are however, the difference in modeling details.
#Specifically, xgboost used a more regularized model formalization to control over-fitting, which gives it better performance.
#=================================================================
#xgbTree
#=================================================================
caret.bp <- train(brand ~ ., 
                  data = bp_training,
                  method = "xgbTree",
                  tuneGrid = tune.grid,
                  trControl = train.control)

stopCluster(cl)
# Examine caret's processing results
caret.bp
plot(caret.bp)
#=================================================================
#variable importance  - XgbTree
#=================================================================
gbmImp_xg <- varImp(caret.bp, scale = FALSE)
gbmImp_xg

#=================================================================
#Automatic tuning grid 
#=================================================================
set.seed(123)
gbm.bp <- train(brand ~ ., 
                  data = bp_training,
                  method = "gbm",
                  trControl = train.control)
gbm.bp
plot(gbm.bp)
#=================================================================
#variable importance - GBM
#=================================================================
gbmImp1 <- varImp(gbm.bp, scale = FALSE)
gbmImp1

#=================================================================
#RandomForest
#=================================================================
rfGrid_bp <- expand.grid(mtry=c(1,2,3,4,5))
set.seed(123)
caret.rf <- train(brand ~ ., 
                  data = bp_training,
                  method = "rf",
                  tuneGrid = rfGrid_bp,
                  trControl = train.control)
caret.rf
plot(caret.rf)
rfImp <- varImp(caret.rf, scale = FALSE)
rfImp

#=================================================================
C5
#=================================================================
fitControl_c5 <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 3)

grid_c5 <- expand.grid( .winnow = FALSE, .trials=c(1,2,3,4,5), .model="tree" )



x <- bp_training[c("age","salary","car","credit","zipcode","elevel")]
y <- bp_training$brand
set.seed(123)
bp.c5 <- train(x=x,y=y,tuneGrid=grid_c5,trControl=fitControl_c5 ,method="C5.0",verbose=FALSE)
bp.c5 
plot(bp.c5)
c5Imp <- varImp(bp.c5, scale = FALSE)
c5Imp

gbmImp_xg
gbmImp1
rfImp
c5Imp
# -------------------------------------------------------------------
# calculate resamples
resample_results <- resamples(list(GBM = gbm.bp,RF = caret.rf, C5.0=bp.c5))

summary(resample_results)

#=================================================================
#Prediction 
#=================================================================

#GBM 

preds_gbm <- predict(gbm.bp, bp_testing)

#RF

preds_rf <- predict(caret.rf, bp_testing)

#C5
preds_c5 <- predict(bp.c5, bp_testing)

#=================================================================
#Confusion Matrix
#=================================================================
confusionMatrix(preds_gbm, bp_testing$brand)
confusionMatrix(preds_rf, bp_testing$brand)
confusionMatrix(preds_c5, bp_testing$brand)

#=================================================================
#Loading Incomplete Survey csv 
#=================================================================

n <- file.choose()
df_ic_bp <-read.csv(n)
View(df_ic_bp)

is.na(df_ic_bp)
attributes(df_ic_bp)
summary(df_ic_bp) 
str(df_ic_bp)
names(df_ic_bp)
sum(is.na(df_ic_bp))

df_ic_bp$brand<- as.factor(df_ic_bp$brand) 

ggplot(data = df_ic_bp) + 
  geom_bar(mapping = aes(x = brand))

#predict 
preds_final_ic_rf <- predict(caret.rf, df_ic_bp)
confusionMatrix(preds_final_ic_rf, df_ic_bp$brand)

resample_results <- resamples(list(GBM = gbm.bp,RF = caret.rf, C5.0=bp.c5))
summary(resample_results)

postResample(preds_rf, bp_testing$brand)
postResample(preds_final_ic_rf, df_ic_bp$brand)

plot(preds_final_ic_rf)

View(preds_final_ic_rf)