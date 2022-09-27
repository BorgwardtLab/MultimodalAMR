library(data.table)
library(reticulate)
library(dplyr)
library(caret)
#library(PRROC)
library(randomForest)
library(adabag)
library(caret) 

np <- import("numpy")

args=(commandArgs(TRUE))
print(args)

setwd(paste("/home/gallia/scratch/u230399/DRIAMS-", args[1], "/", args[2], "/", sep=""))
#setwd("/home/gallia/scratch/u230399/DRIAMS-B/Cefepime")

eval <- function (CM){
 if(ncol(CM)==2 & nrow(CM)==2){
  TP <- CM[1,1]
  TN <- CM[2,2]
  FP <- CM[1,2]
  FN <- CM[2,1]
  precision =(TP)/(TP+FP)
  recall =(TP)/(TP+FN)
  mcc_num <- (TP*TN - FP*FN)
  mcc_den <- as.double((TP+FP))*as.double((TP+FN))*as.double((TN+FP))*as.double((TN+FN))
  mcc <- mcc_num/sqrt(mcc_den)
  F1=(2*TP)/(2*TP+FP+FN)
  accuracy=(TP+TN)/(TP+TN+FP+FN)
  return(c(precision, recall, mcc, F1, accuracy))
 } else {
  return(c(NA, NA, NA, NA, NA))
 }
}

########################################
# Train models from siamese embeddings #
########################################
mat <- np$load("embeddings_zero_shot_train.npy", allow_pickle=T)
data=fread("zero_shot_spect_drugEmbFingerPrint_Resist_train.csv")
data=data.frame(data$response, mat)
data=data[c(TRUE, lapply(data[-1], var, na.rm = TRUE) != 0)] #remove constant variables
tmp <- cor(data)
tmp[!lower.tri(tmp)] <- 0
data <- data[, !apply(tmp, 2, function(x) any(abs(x) > 0.99, na.rm = TRUE))]
VarToKeep=colnames(data)

#Logistic regression
print("Training logistic regression")
model_logReg <- glm( data.response ~., data = data, family = binomial)
probabilities_logReg <- model_logReg %>% predict(data, type = "response")
predicted.classes_logReg <- ifelse(probabilities_logReg > 0.5, 1, 0)
CM_logReg = table(as.factor(data$data.response), as.factor(as.integer(predicted.classes_logReg)))
performance_train_logReg =data.frame(metric=c("precision", "recall", "mcc", "F1", "accuracy"), perf=eval(CM_logReg), type=rep("logReg", 5))

#rf
print("Training random forest")
data$data.response=as.factor(data$data.response)
model_rf <- randomForest(data.response ~ ., data = data, ntree = 500, na.action = na.omit)
probabilities <- predict(model_rf, data)
CM= table(as.factor(data$data.response),as.factor(probabilities))
performance_train_rf=data.frame(metric=c("precision", "recall", "mcc", "F1", "accuracy"), perf=eval(CM), type=rep("rf", 5))

#Ababoost
print("Training Adaboost")
data$data.response=as.factor(data$data.response)
model_adaboost <- boosting(data.response~., data=data, boos=TRUE, mfinal=50)
probabilities <- predict(model_adaboost, data)
CM= probabilities$confusion
performance_train_ada=data.frame(metric=c("precision", "recall", "mcc", "F1", "accuracy"), perf=eval(CM), type=rep("adaboost", 5))

################################
# Evaluation on validation set #
################################
mat <- np$load("embeddings_zero_shot_val.npy")
data=fread("zero_shot_spect_drugEmbFingerPrint_Resist_val.csv")
data=data.frame(data$response, mat)
data=data %>% select(VarToKeep)

#Logistic regression
probabilities <- model_logReg %>% predict(data, type = "response")
predicted.classes <- ifelse(probabilities> 0.5, 1, 0)
CM= table(as.factor(data$data.response), as.factor(as.integer(predicted.classes)))
performance_val_logReg=data.frame(metric=c("precision", "recall", "mcc", "F1", "accuracy"), perf=eval(CM), type=rep("logReg", 5))

#rf
data$data.response=as.factor(data$data.response)
probabilities <- predict(model_rf, data)
CM= table(as.factor(data$data.response),as.factor(probabilities))
performance_val_rf=data.frame(metric=c("precision", "recall", "mcc", "F1", "accuracy"), perf=eval(CM), type=rep("rf", 5))

#adaboost
data$data.response=as.factor(data$data.response)
probabilities <- predict(model_adaboost, data)
CM= probabilities$confusion
performance_val_ada=data.frame(metric=c("precision", "recall", "mcc", "F1", "accuracy"), perf=eval(CM), type=rep("rf", 5))


#############################
# Evaluation on testing set #
#############################
mat <- np$load("embeddings_zero_shot_test.npy")
data=fread("zero_shot_spect_drugEmbFingerPrint_Resist_test.csv")
data=data.frame(data$response, mat)
data=data %>% select(VarToKeep)

#Logistic regression
probabilities <- model_logReg %>% predict(data, type = "response")
predicted.classes <- ifelse(probabilities > 0.5, 1, 0)
CM= table(as.factor(data$data.response), as.factor(as.integer(predicted.classes)))
performance_test_logReg=data.frame(metric=c("precision", "recall", "mcc", "F1", "accuracy"), perf=eval(CM), type=rep("logReg", 5))

#rf
data$data.response=as.factor(data$data.response)
probabilities <- predict(model_rf, data)
CM= table(as.factor(data$data.response),as.factor(probabilities))
performance_test_rf=data.frame(metric=c("precision", "recall", "mcc", "F1", "accuracy"), perf=eval(CM), type=rep("rf", 5))

#adaboost
data$data.response=as.factor(data$data.response)
probabilities <- predict(model_adaboost, data)
CM= probabilities$confusion
performance_test_ada=data.frame(metric=c("precision", "recall", "mcc", "F1", "accuracy"), perf=eval(CM), type=rep("rf", 5))

##########
# Export #
##########
output_train=rbind(performance_train_logReg, performance_train_rf, performance_train_ada)
output_val=rbind(performance_val_logReg, performance_val_rf, performance_val_ada)
output_test=rbind(performance_test_logReg, performance_test_rf, performance_test_ada)

fwrite(output_train, paste("/home/gallia/scratch/u230399/DRIAMS-", args[1], "/", args[2], "/performance_train.txt", sep=""))
fwrite(output_val, paste("/home/gallia/scratch/u230399/DRIAMS-", args[1], "/", args[2], "/performance_val.txt", sep=""))
fwrite(output_test, paste("/home/gallia/scratch/u230399/DRIAMS-", args[1], "/", args[2], "/performance_test.txt", sep=""))

