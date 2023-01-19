library(data.table)
library(plyr)
library(tidyverse)

setwd("/home/")
output_path="/home/"
train=fread("df_train_B.csv")
test=fread("df_test_B.csv")
train=train[,c(1,2,3,4)]
test=test[,c(1,2,3,4)]
id_test=unique(test$sample_id)

#######################################################################
# Evolution of the precision according to the top number of neighbors #
#######################################################################

total_precision=list()
total_nb_drugs=list()
precision=data.frame(NA,NA,NA,NA)
nb_drugs=data.frame(NA,NA,NA,NA)
top_k=c(1,5,10,30,50,75,100) #nb of neighbors corresponding to same species

for(m in 1:100){ #for 100 replicates
 print(m)
 for(i in 1:length(id_test)){ #for each individual the test set
   tmp_test=test[test$sample_id==id_test[i],]
   tmp_train1=train[train$species==unique(tmp_test$species),]
   id_train=unique(tmp_train1$sample_id)
   for(k in 1:length(top_k)){ 
    if(length(unique(tmp_train1$sample_id))>top_k[k]){ #if enough neighbors are available
     tmp_train=tmp_train1 %>% filter(sample_id %in% sample(id_train, top_k[k])) #select neighbors corresponding to same species
     tmp_train=tmp_train[,3:4]
     tmp_train$response=ifelse(tmp_train$response==0,1,0)
     tmp_train=aggregate(tmp_train$response, by=list(Category=tmp_train$drug), FUN=sum) #compute the drug sensititivy and resistance occurrence
     tmp_train=tmp_train[order(-tmp_train$x),]
     colnames(tmp_train)=c("drug", "count")
     tmp_train_test=merge(tmp_test, tmp_train, "drug")
     tmp_train_test=tmp_train_test[order(-count), ]
     if(nrow(tmp_train_test)==0){ #if there is no drug in I_sr
        precision[i,k]=0
     } else {
     tmp_train_test=tmp_train_test[tmp_train_test$count==max(tmp_train_test$count),] #select the drug(s) with the highest occurrence
     nb_drugs[i,k]=nrow(tmp_train_test)
     precision[i,k]= 1-(sum(tmp_train_test$response)/nrow(tmp_train_test)) }
   } else { precision[i,k]=NA }
  }
 }
 total_precision[[m]]=precision
 total_nb_drugs[[m]]=nb_drugs
}

total_precision=do.call(rbind, total_precision)
total_nb_drugs=do.call(rbind, total_nb_drugs)

fwrite(total_precision, paste(output_path, "baseline_species_total.csv", sep=""))
fwrite(total_nb_drugs, paste(output_path, "baseline_nb_drugs_total_nb_drugs.csv", sep=""))

####################################################################################
# Precision at cutoff n=1..5 for                                                   #
# option 1: k=30, if less than k-neighbors available, do not compute the precision #
####################################################################################

k=30
total_sensitivity=list()
total_resistance=list()
result_sensitivity=matrix(ncol=5, nrow=length(id_test))
result_resistance=matrix(ncol=5, nrow=length(id_test))

for(m in 1:100){ #for 100 replicates
 print(m)
 for(i in 1:length(id_test)){ #for each individual the test set
   tmp_test=test[test$sample_id==id_test[i],]
   tmp_train1=train[train$species==unique(tmp_test$species),]
   id_train=unique(tmp_train1$sample_id)
   if(length(unique(tmp_train1$sample_id))>k){
    #select the top-k closest neighbors
    tmp_train=tmp_train1 %>% filter(sample_id %in% sample(id_train, k))
    tmp_train=tmp_train[,3:4]
    tmp_train$response=ifelse(tmp_train$response==0,1,0)
    tmp_train=aggregate(tmp_train$response, by=list(Category=tmp_train$drug), FUN=sum)
    tmp_train=tmp_train[order(-tmp_train$x),]
    colnames(tmp_train)=c("drug", "count")
    tmp=merge(tmp_test, tmp_train, by="drug")
    tmp=tmp[order(-count), ]
    for(j in 1:5){
     if(nrow(tmp)>=j){ #if we have tested more than i drugs with know values for sample j
      result_sensitivity[i,j]=(j-sum(tmp[1:j,"response"]))/j
      result_resistance[i,j]=sum(tmp[(nrow(tmp)-j+1):nrow(tmp),"response"])/j
     } else { 
             result_sensitivity[i,j]=NA      
             result_resistance[i,j]=NA 
    }}} else { result_sensitivity[i,]=NA 
    result_resistance[i,]=NA }
 }
 total_sensitivity[[m]]=result_sensitivity
 total_resistance[[m]]=result_resistance
}

total_sensitivity=do.call(rbind, total_sensitivity)
total_resistance=do.call(rbind, total_resistance)

fwrite(total_sensitivity, paste(output_path, "result_sensitivity_baseline_species_100rep.csv", sep=""))
fwrite(total_resistance, paste(output_path, "result_resistance_baseline_species_100rep.csv", sep=""))


####################################################################################
# Precision at cutoff n=1..5 for                                                   #
# option 2: for each test samepl k=max available neighbors (same species) in train #
####################################################################################

result_sensitivity=matrix(ncol=5, nrow=length(id_test))
result_resistance=matrix(ncol=5, nrow=length(id_test))
total_sensitivity=list()
total_resistance=list()

for(m in 1:100){ #for 100 replicates
 print(m)
 for(i in 1:length(id_test)){ #for each individual the test set
   tmp_test=test[test$sample_id==id_test[i],]
   tmp_train1=train[train$species==unique(tmp_test$species),]
   id_train=unique(tmp_train1$sample_id)
   if(length(unique(tmp_train1$sample_id))>0){
    tmp_train=tmp_train1
    tmp_train=tmp_train[,3:4]
    tmp_train$response=ifelse(tmp_train$response==0,1,0)
    tmp_train=aggregate(tmp_train$response, by=list(Category=tmp_train$drug), FUN=sum)
    tmp_train=tmp_train[order(-tmp_train$x),]
    colnames(tmp_train)=c("drug", "count")
    tmp=merge(tmp_test, tmp_train, by="drug")
    tmp=tmp[order(-count), ]
    for(j in 1:5){
     if(nrow(tmp)>=j){ #if we have tested more than i drugs with know values for sample j
      result_sensitivity[i,j]=(j-sum(tmp[1:j,"response"]))/j
      result_resistance[i,j]=sum(tmp[(nrow(tmp)-j+1):nrow(tmp),"response"])/j
      } else { 
             result_sensitivity[i,j]=NA      
             result_resistance[i,j]=NA 
    }}} else { result_sensitivity[i,]=NA 
    result_resistance[i,]=NA }
 }
 total_sensitivity[[m]]=result_sensitivity
 total_resistance[[m]]=result_resistance
}

total_sensitivity=do.call(rbind, total_sensitivity)
total_resistance=do.call(rbind, total_resistance)

fwrite(total_sensitivity, paste(output_path, "result_sensitivity_baseline_species_maxSamples_100rep.csv", sep=""))
fwrite(total_resistance, paste(output_path, "result_resistance_baseline_species_maxSamples_100rep.csv", sep=""))

####################################################################################
# Truncated precision at cutoff n=1..5 for                                         #
# option 1: k=30, if less than k-neighbors available, do not compute the precision #
####################################################################################

total_sensitivity=list()
total_resistance=list()
k=30
result_sensitivity=matrix(ncol=5, nrow=length(id_test))
result_resistance=matrix(ncol=5, nrow=length(id_test))

for(m in 1:100){ #for 100 replicates
 print(m)
 for(i in 1:length(id_test)){ #for each individual the test set
   tmp_test=test[test$sample_id==id_test[i],]
   tmp_train1=train[train$species==unique(tmp_test$species),]
   id_train=unique(tmp_train1$sample_id)
   if(length(unique(tmp_train1$sample_id))>k){
    #print("enough similar species")
    #select the top-k closest neighbors
     tmp_train=tmp_train1 %>% filter(sample_id %in% sample(id_train, k))
     tmp_train=tmp_train[,3:4]
     tmp_train$response=ifelse(tmp_train$response==0,1,0)
     tmp_train=aggregate(tmp_train$response, by=list(Category=tmp_train$drug), FUN=sum)
     tmp_train=tmp_train[order(-tmp_train$x),]
     colnames(tmp_train)=c("drug", "count")
     tmp=merge(tmp_test, tmp_train, by="drug")
     tmp=tmp[order(-count), ]
     for(j in 1:5){
      if(nrow(tmp)>=j){ #if we have tested more than i drugs with know values for sample j
       result_sensitivity[i,j]=(j-sum(tmp[1:j,"response"]))/(min(j, nrow(tmp[tmp$response==0,]))) #truncated precision
       result_resistance[i,j]=sum(tmp[(nrow(tmp)-j+1):nrow(tmp),"response"])/(min(j, nrow(tmp[tmp$response==1,]))) #truncated precision
       } else { 
              result_sensitivity[i,j]=NA      
              result_resistance[i,j]=NA 
     }}} else { result_sensitivity[i,]=NA 
    result_resistance[i,]=NA }
 }
 total_sensitivity[[m]]=result_sensitivity
 total_resistance[[m]]=result_resistance
}

total_sensitivity=do.call(rbind, total_sensitivity)
total_resistance=do.call(rbind, total_resistance)

fwrite(total_sensitivity, paste(output_path, "truncatedResult_sensitivity_baseline_species_100rep.csv", sep=""))
fwrite(total_resistance, paste(output_path, "truncatedResult_resistance_baseline_species_100rep.csv", sep=""))

####################################################################################
# Truncated precision at cutoff n=1..5 for                                         #
# option 2: for each test samepl k=max available neighbors (same species) in train #
####################################################################################

result_sensitivity=matrix(ncol=5, nrow=length(id_test))
result_resistance=matrix(ncol=5, nrow=length(id_test))
total_sensitivity=list()
total_resistance=list()


for(m in 1:100){ #for 100 replicates
 print(m)
 for(i in 1:length(id_test)){ #for each individual the test set
   tmp_test=test[test$sample_id==id_test[i],]
   tmp_train1=train[train$species==unique(tmp_test$species),]
   id_train=unique(tmp_train1$sample_id)
   if(length(unique(tmp_train1$sample_id))>0){
    tmp_train=tmp_train1
    tmp_train=tmp_train[,3:4]
    tmp_train$response=ifelse(tmp_train$response==0,1,0)
    tmp_train=aggregate(tmp_train$response, by=list(Category=tmp_train$drug), FUN=sum)
    tmp_train=tmp_train[order(-tmp_train$x),]
    colnames(tmp_train)=c("drug", "count")
    tmp=merge(tmp_test, tmp_train, by="drug")
    tmp=tmp[order(-count), ]
    for(j in 1:5){
     if(nrow(tmp)>=j){ #if we have tested more than i drugs with know values for sample j
      result_sensitivity[i,j]=(j-sum(tmp[1:j,"response"]))/(min(j, nrow(tmp[tmp$response==0,]))) #truncated precision
      result_resistance[i,j]=sum(tmp[(nrow(tmp)-j+1):nrow(tmp),"response"])/(min(j, nrow(tmp[tmp$response==1,]))) #truncated precision
      } else { 
              result_sensitivity[i,j]=NA      
              result_resistance[i,j]=NA 
     }}} else { result_sensitivity[i,]=NA 
    result_resistance[i,]=NA }
 }
 total_sensitivity[[m]]=result_sensitivity
 total_resistance[[m]]=result_resistance
}

total_sensitivity=do.call(rbind, total_sensitivity)
total_resistance=do.call(rbind, total_resistance)

fwrite(total_sensitivity, paste(output_path, "truncatedPrecision_sensitivity_baseline_species_maxSamples_100rep.csv", sep=""))
fwrite(total_resistance, paste(output_path, "truncatedPrecision_resistance_baseline_species_maxSamples_100rep.csv", sep=""))


