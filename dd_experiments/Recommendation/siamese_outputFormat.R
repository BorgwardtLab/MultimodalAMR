library(data.table)
library(reticulate)
library(dplyr)
library(caret)
library(PRROC)
library(randomForest)
library(adabag)
library(caret) 

np <- import("numpy")

################
# Training set #
################

#Load data
mat <- np$load("/home/gallia/scratch/u230399/DRIAMS-B/recommendation/embeddings_train.npy", allow_pickle=T)
data=fread("/home/gallia/scratch/u230399/DRIAMS-B/recommendation/spect_drugEmbFingerPrint_Resist_train.csv")

#Format data
data=data.frame(data$sample_id, mat)
data=data[c(TRUE, lapply(data[-1], var, na.rm = TRUE) != 0)] #remove constant variables
tmp <- cor(data[,2:ncol(data)])
tmp[!lower.tri(tmp)] <- 0
train <- data[, !apply(tmp, 2, function(x) any(abs(x) > 0.99, na.rm = TRUE))]
VarToKeep=colnames(train)
colnames(train)=c("sample_id", colnames(train[,2:ncol(train)]))

#Export
fwrite(train, "/home/gallia/scratch/u230399/DRIAMS-B/recommendation/embeddings_train_reduced.csv")

###############
# Testing set #
###############

#Load data
mat <- np$load("/home/gallia/scratch/u230399/DRIAMS-B/recommendation/embeddings_test.npy")
data=fread("/home/gallia/scratch/u230399/DRIAMS-B/recommendation/spect_drugEmbFingerPrint_Resist_test.csv")

#Format data
data=data.frame(data$sample_id, mat)
data=data %>% select(VarToKeep)
colnames(data)=c("sample_id", colnames(data[,2:ncol(data)]))

#Export
fwrite(data, "/home/gallia/scratch/u230399/DRIAMS-B/recommendation/embeddings_test_reduced.csv")