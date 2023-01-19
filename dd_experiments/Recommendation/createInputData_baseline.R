library(data.table)
library(tidyr)
set.seed(123)

args=(commandArgs(TRUE))
print(args)
path=args[2]
setwd(path)

###############
# Import data #
###############
data=fread(paste("/home/DRIAMS_combined_long_table.csv"))
data=data[data$dataset==args[1],]

#########################
# Create train and test #
#########################
#all ids
id=unique(data$sample_id)

#test ids
test_id=c()
while(length(test_id)<round(0.2*length(id))){
  i=sample(seq(1,length(id)), 1)
  tmp=data[data$sample_id==id[i],]
  #samples in the test need to have at least one sensitive and one resistant drug
  if(all(mean(tmp$response)!=1 & mean(tmp$response)!=0 & id[i] %in% test_id==F)){  
	test_id=c(test_id, id[i])
 }
}
test_id=data.frame(test_id)
colnames(test_id)="sample_id"

#train ids
train_id=setdiff(id, as.character(test_id$sample_id))
train_id=data.frame(train_id)
colnames(train_id)="sample_id"

#Load all spectrums
all.files <- list.files(paste("/home/DRIAMS-", args[1], "/binned_6000/2018", sep=""), pattern="*.txt", full.names=T)

samples=data.frame(all.files, sub(".*/", "", all.files))
colnames(samples)=c("path", "samples")

listOfSamples=list()
print(nrow(samples))
for(i in 1:nrow(samples)){
   print(i)
   x=samples[i,1]
   tmp=fread(file = as.character(x))
   listOfSamples[[i]]=tmp[,2]
}
all <- as.data.frame(do.call("cbind", listOfSamples))
all=t(all)
colnames(all)=paste("binned_intensity", seq(1, ncol(all)))
all=as.data.frame(all)
all$sample_id=substring(samples$samples,1, nchar(as.character(samples$samples))-4)

spectrum_train=merge(all, train_id, by="sample_id")
spectrum_test=merge(all, test_id, by="sample_id")
df_train=merge(data,  train_id, by="sample_id")
df_test=merge(data,  test_id, by="sample_id")

##########
# Export #
##########

fwrite(spectrum_train, paste(path, "spectrum_train.csv", sep=""))
fwrite(spectrum_test, paste(path, "spectrum_test.csv", sep=""))
fwrite(data, paste(path, "df_", args[1], ".csv", sep=""))
fwrite(df_train, paste(path, "df_train_", "B", ".csv", sep=""))
fwrite(df_test, paste(path, "df_test_", "B", ".csv", sep=""))

