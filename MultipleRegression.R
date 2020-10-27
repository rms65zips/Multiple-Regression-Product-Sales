# Title: Predicting Sales Volume

# Last update: 2-13-2020

# File/project name: 2020 Task3MultipleRegression.R
# RStudio Project name: Task 3 - Task3MultReg.Rproj

###############
# Project Notes
###############

# Analysis Purpose:
# Predicting sales of four different product types: PC, Laptops, Netbooks and Smartphones
# Assessing the impact services reviews and customer reviews have on sales of different product types

# Assignment "<-" short-cut: 
#   Win [Alt]+[-] 


# Comment multiple lines
# WIN: CMD + SHIFT + C


###############
# Housekeeping
###############

# Clear objects if necessary
rm(list = ls())


getwd()

setwd("F:/UT Data Analytics/Course 2 - Predicting Customer Preferences/Task3 - Multiple Regression in R/Task3MultReg")
dir()

# set a value for seed (to be used in the set.seed function)
seed <- 123


################
# Load packages
################

# install.packages("caret")
# install.packages("corrplot")
# install.packages("readr")
# install.packages("corrplot")
install.packages("kernlab")
library(corrplot)
library(caret)
library(corrplot)
library(doMC)
library(doParallel)
library(mlbench)
library(readr)
library(kernlab)


#####################
# Parallel processing
#####################


#--- for WIN ---#
# install.packages("doParallel") # install in 'Load packages' section above
#library(doParallel)  # load in the 'Load Packages' section above
detectCores()  # detect number of cores
cl <- makeCluster(2)  # select number of cores; 2 in this example
registerDoParallel(cl) # register cluster
getDoParWorkers()  # confirm number of cores being used by RStudio
# Stop Cluster. After performing your tasks, make sure to stop your cluster. 
stopCluster(cl)


##############
# Import data
##############

#### --- Load raw datasets --- ####

# --- Load Train/Existing data (Dataset 1) --- #
existingProductAttributes <- read.csv("existingproductattributes2017.csv", stringsAsFactors = FALSE)
class(existingProductAttributes)  # "data.frame"
str(existingProductAttributes)


# --- Load Predict/New data (Dataset 2) --- #

newProductAttributes <- read.csv("newproductattributes2017.csv", stringsAsFactors = FALSE)
class(newProductAttributes)  # "data.frame"
str(newProductAttributes)



#### --- Load preprocessed datasets --- ####

#dummify data
existingProductAttributes_dummy <- dummyVars("~.", data = existingProductAttributes)
readyExistingData <- data.frame(predict(existingProductAttributes_dummy,newdata = existingProductAttributes))

newProductAttributes_dummy <- dummyVars("~.",data = newProductAttributes)
readyNewData <- data.frame(predict(newProductAttributes_dummy,newdata = newProductAttributes))


################
# Evaluate data
################

#--- Dataset 1 ---#
str(readyExistingData)  # 
names(readyExistingData)
summary(readyExistingData)
head(readyExistingData)
tail(readyExistingData)

#--- Dataset 2 ---#
str(readyNewData)  # 
names(readyNewData)
summary(readyNewData)
head(readyNewData)
tail(readyNewData)

#--- Dataset 1 ---#
# plot
hist(readyExistingData$Volume)
plot(readyExistingData$x5StarReviews, readyExistingData$Volume)
qqnorm(readyExistingData$Volume)
# check for missing values 
anyNA(readyExistingData)
is.na(readyExistingData)

#--- Dataset 2 ---#
hist(readyNewData$Volume)
plot(readyNewData$x5StarReviews, readyNewData$Volume)
qqnorm(readyNewData$Volume)
# check for missing values 
anyNA(readyNewData)
is.na(readyNewData)


#############
# Preprocess
#############

#--- Dataset 1 ---#

# change data types
#DatasetName$ColumnName <- as.typeofdata(DatasetName$ColumnName)

# rename a column
#names(DatasetName)<-c("ColumnName","ColumnName","ColumnName") 

# check for missin values (NAs)
any(is.na(readyExistingData)) 

# handle missing values 
# remove obvious features (e.g., ID, other)
readyExistingData27v <- readyExistingData   # make a copy 
readyExistingData27v$BestSellersRank <- NULL
readyExistingData27v$ProductNum <- NULL
str(readyExistingData27v) #27 Variables

# save preprocessed dataset
write.csv(readyExistingData27v, "readyExistingData27v.csv")


#--- Dataset 2 ---#
# change data types
#DatasetName$ColumnName <- as.typeofdata(DatasetName$ColumnName)

# rename a column
#names(DatasetName)<-c("ColumnName","ColumnName","ColumnName") 

# check for missin values (NAs)
any(is.na(readyNewData)) 

# handle missing values 
# remove obvious features (e.g., ID, other)
readyNewData27v <- readyNewData   # make a copy 
readyNewData27v$BestSellersRank <- NULL
readyNewData27v$ProductNum <- NULL
str(readyNewData27v) #27 Variables

# save preprocessed dataset
write.csv(readyNewData27v, "readyNewData27v.csv")


################
# Feature Selection
################

## ---- Corr analysis ----- ###

readyExisting27vCorr <- cor(readyExistingData27v)
readyExisting27vCorr


corrplot(readyExisting27vCorr)

###Remove x5Star/ x3Star / x1Star Dataset #1
readyExistingData24v <- readyExistingData27v
readyExistingData24v$x5StarReviews  <- NULL
readyExistingData24v$x3StarReviews <- NULL
readyExistingData24v$x1StarReviews <- NULL
str(readyExistingData24v)
write.csv(readyExistingData24v,"readyExistingData24v.csv")

###Remove x5Star/ x3Star / x1Star Dataset #2
readyNewData24v <- readyNewData27v
readyNewData24v$x5StarReviews  <- NULL
readyNewData24v$x3StarReviews <- NULL
readyNewData24v$x1StarReviews <- NULL
str(readyNewData24v)
write.csv(readyNewData24v,"readyNewData24v.csv")

################
# Sampling
################



##################
# Train/test sets
##################

# create the training partition that is 75% of total obs
set.seed(seed) # set random seed
inTraining <- createDataPartition(readyExistingData24v$Volume, p=0.75, list=FALSE)
# create training/testing dataset
trainSet <- readyExistingData24v[inTraining,]   
testSet <- readyExistingData24v[-inTraining,]   
# verify number of obs 
nrow(trainSet)  
nrow(testSet)   


################
# Train control
################

# set 10 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)


##############
# Train model
##############

?modelLookup()
modelLookup("rf")


## ------- LM ------- ##

# LM train/fit
set.seed(seed)
#lmFit1 <- train(SolarRad~., data=trainSet, method="leapSeq", trControl=fitControl)
lmFit1 <- train(Volume~., data=trainSet, method="lm", trControl=fitControl)
lmFit1  
# RMSE      Rsquared   MAE     
# 801.8518  0.7780317  489.2355


# make predictions
lmPred1 <- predict(lmFit1, testSet)
lmPred1
# performace measurment
postResample(lmPred1, testSet$Volume)
# Note results

## ------- RF ------- ##

# RF train/fit
set.seed(seed)
system.time(rfFit1 <- train(Volume~., data=trainSet, method="rf", importance=T, trControl=fitControl)) #importance is needed for varImp
rfFit1

# mtry  RMSE      Rsquared   MAE     
# 2    885.6485  0.8169762  513.2943
# 12    809.4328  0.8896861  382.8019
# 23    754.4303  0.9035842  350.8508
# 
# RMSE was used to select the optimal model using the smallest value.
# The final value used for the model was mtry = 23.

plot(rfFit1)

rfVarFit1 <- varImp(rfFit1)
rfVarFit1
# Overall
# PositiveServiceReview       100.000
# x4StarReviews                44.563
# x2StarReviews                24.152
# ProductWidth                 13.579
# ProductHeight                13.316
# ProductDepth                 11.126
# ProductTypePC                11.105
# ProductTypeExtendedWarranty  10.940
# ProductTypePrinter           10.613
# ProductTypeGameConsole       10.148
# ProductTypePrinterSupplies    8.846
# NegativeServiceReview         8.277
# Price                         6.379
# Recommendproduct              5.776
# ProductTypeLaptop             5.551
# ShippingWeight                5.530
# ProductTypeSoftware           5.218
# ProductTypeAccessories        4.545
# ProductTypeDisplay            4.393
# ProfitMargin                  3.920


## ------- SVM ------- ##

# SVM train/fit
set.seed(seed)
svmFit1 <- train(Volume~., data=trainSet, method="svmLinear", trControl=fitControl)
svmFit1

# RMSE      Rsquared   MAE     
# 999.2661  0.7822891  552.0302
# 
# Tuning parameter 'C' was held constant at a value of 1


#################
# Evaluate models
#################

##--- Compare models ---##

# use resamples to compare model performance
ModelFitResults <- resamples(list(lm=lmFit1, rf=rfFit1, svm=svmFit1))
# output summary metrics for tuned models 
summary(ModelFitResults)

### Random Forest is the best performing model. ###
# mtry  RMSE      Rsquared   MAE     
# 2    885.6485  0.8169762  513.2943
# 12    809.4328  0.8896861  382.8019
# 23    754.4303  0.9035842  350.8508
# 
# RMSE was used to select the optimal model using the smallest value.
# The final value used for the model was mtry = 23.

##--- Conclusion ---##

# Random Forest will be chosen because it performed better than the other algorithms in terms of R-Squared and RMSE.


########################
# Validate top model
########################

# make predictions

rfPred1 <- predict(rfFit1, testSet)

# performace measurment
postResample(rfPred1, testSet$Volume)
# RMSE    Rsquared         MAE 
# 972.5775407   0.6178754 338.3700000 

# plot predicted verses actual
plot(rfPred1,testSet$Volume)

rfPred1


########################
# Predict with top model
########################

# make predictions
rfPred_readyNewData <- predict(rfFit1, readyNewData24v)
rfPred_readyNewData
# 1          2          3          4          5          6          7          8          9         10         11         12         13 
# 494.44667  182.60840  218.78733   39.43200   15.65333   56.65600 1223.00707  149.66840   22.98093 1155.21733 7451.18853  383.83520  637.87867 
# 14         15         16         17         18         19         20         21         22         23         24 
# 81.27707  143.12733 1290.65187   21.99787   38.05667   60.94400  120.46227   78.46387   19.68987   13.59507 2881.50253 

write.csv(rfPred_readyNewData,"rfPred_readyNewData.csv")


########################
# Save validated model
########################



# save model 
saveRDS(rfFit1,file="RFfitMultReg.rds")  

# load and name model to make predictions with new data
RFfit1 <- readRDS(file="RFfitMultReg") # Q: What type of object does readRDS create?





