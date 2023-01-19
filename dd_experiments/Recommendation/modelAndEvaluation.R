library(data.table)
library(reticulate)
library(dplyr)
library(caret)
library(PRROC)
library(randomForest)
library(adabag)
library(caret) 

np <- import("numpy")
path="/home/"
outputpath="/home/"
setwd(path)

########################################
# Train models from siamese embeddings #
########################################

#Load data
mat <- np$load("/home/embeddings_train.npy", allow_pickle=T)
data=fread("/home/spect_drugEmbFingerPrint_Resist_train.csv")

#Format data
data=data.frame(data$response, mat)
data=data[c(TRUE, lapply(data[-1], var, na.rm = TRUE) != 0)] #remove constant variables
tmp <- cor(data)
tmp[!lower.tri(tmp)] <- 0
data <- data[, !apply(tmp, 2, function(x) any(abs(x) > 0.99, na.rm = TRUE))]
VarToKeep=colnames(data)

#Logistic regression
print("Training logistic regression")
model_logReg <- glm( data.response ~., data = data, family = binomial)
#save(model_logReg, file="model_logReg.Rdata")

#############################
# Evaluation on testing set #
#############################
mat <- np$load("/home/embeddings_test.npy")
data=fread("/home/spect_drugEmbFingerPrint_Resist_test.csv")
info=data[,c(1,3,4)]
data=data.frame(data$response, mat)
data=data %>% select(VarToKeep)

  #########################
  # Precision at cutoff j #
  #########################

probabilities <- model_logReg %>% predict(data, type = "response")
recommended=data.frame(info, probabilities)
test_id=unique(recommended$sample_id)
result_logReg_sensitivity=matrix(nrow=length(test_id), ncol=5)
result_logReg_resistance=matrix(nrow=length(test_id), ncol=5)
for(i in 1:length(test_id)){ #for each sample in the test set
  id=test_id[i]
  tmp=recommended[recommended$sample_id==id,]
  tmp=tmp[order(tmp$probabilities),]
  for(j in 1:5){
    if(nrow(tmp)>=j){ #if we have tested more than i drugs with know values for sample j
    result_logReg_sensitivity[i,j]=(j-sum(tmp[1:j,"response"]))/j
    result_logReg_resistance[i,j]=sum(tmp[(nrow(tmp)-j+1):nrow(tmp),"response"])/j
    } else { 
             result_logReg_sensitivity[i,j]=NA      
             result_logReg_resistance[i,j]=NA 
           }
  }
}
#Export
fwrite(result_logReg_sensitivity, paste(outputpath, "/result_logReg_sensitivity.csv", sep=""))
fwrite(result_logReg_resistance, paste(outputpath, "/result_logReg_resistance.csv", sep=""))


  ###################################
  # Truncated precision at cutoff j #
  ###################################

probabilities <- model_logReg %>% predict(data, type = "response")
recommended=data.frame(info, probabilities)
head(recommended)
test_id=unique(recommended$sample_id)
result_logReg_sensitivity=matrix(nrow=length(test_id), ncol=5)
result_logReg_resistance=matrix(nrow=length(test_id), ncol=5)
for(i in 1:length(test_id)){ #for each sample in the test set
  id=test_id[i]
  tmp=recommended[recommended$sample_id==id,]
  tmp=tmp[order(tmp$probabilities),]
  #Precicion at cutoff j
  for(j in 1:5){
    if(nrow(tmp)>=j){ #if we have tested more than i drugs with know values for sample j
    result_logReg_sensitivity[i,j]=(j-sum(tmp[1:j,"response"]))/(min(j, nrow(tmp[tmp$response==0,]))) #truncated precision
    result_logReg_resistance[i,j]=sum(tmp[(nrow(tmp)-j+1):nrow(tmp),"response"])/(min(j, nrow(tmp[tmp$response==1,]))) #truncated precision
    } else { 
             result_logReg_sensitivity[i,j]=NA      
             result_logReg_resistance[i,j]=NA 
           }
  }
}
#Export
fwrite(result_logReg_sensitivity, paste(outputpath, "/truncatedPrecision_logReg_sensitivity.csv", sep=""))
fwrite(result_logReg_resistance, paste(outputpath, "/truncatedPrecision_logReg_resistance.csv", sep=""))
