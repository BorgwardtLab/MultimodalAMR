library(data.table)
library(tidyr)
set.seed(123)

args=(commandArgs(TRUE))
print(args)

setwd(args[3])
#setwd("/home/gallia/scratch/u230399/DRIAMS-B/Cefepime/")

for(a in c("train", "test", "val")){

#load resistance
resistance=fread(paste(a, "_data_zero_shot.csv", sep=""))
resistance=resistance[,-1]
resistance$drug <- tolower(resistance$drug)

#Load drug fingerprint
drugFingerprint=fread("/home/gallia/scratch/u230399/drug_fingerprints.csv")
drugFingerprint$drug <- tolower(drugFingerprint$drug)
drugFingerprint=data.frame(drugFingerprint)

fingerPrint=list()
for(i in 1:(ncol(drugFingerprint)-1)){ #for each fingerprint type
 tmp=drugFingerprint[,i+1]
 splitting=unlist(strsplit(tmp[1], split = ""))
 for(j in 2:nrow(drugFingerprint)){ #for each drug
   splitting=rbind(splitting, unlist(strsplit(tmp[j], split = "")))
 }
colnames(splitting)=paste(colnames(drugFingerprint[i+1]), seq(1,ncol(splitting)), sep="_")
fingerPrint[[i]]=splitting
}
fingerPrint=as.data.frame(do.call("cbind", fingerPrint))
drugFingerprint <- data.frame(drugFingerprint$drug, fingerPrint)
colnames(drugFingerprint)=c("drug", colnames(drugFingerprint[,2:ncol(drugFingerprint)]))

#merge resistance and drug embeddings
resistance_drug=merge(resistance, drugFingerprint, by="drug")

#load all spectrums
samples=unique(resistance_drug$sample_id)
samples=paste(samples, ".txt", sep="")
length(samples)

all.files <- list.files(paste("/home/gallia/scratch/u230399/DRIAMS-", args[1], "/binned_6000/2018", sep=""),
			pattern="*.txt", full.names=T)
#all.files <- list.files(paste("/home/gallia/scratch/u230399/DRIAMS-", "B", "/binned_6000/2018", sep=""),
#			pattern="*.txt", full.names=T)

all.files=data.frame(all.files, sub(".*/", "", all.files))
colnames(all.files)=c("path", "samples")
samples=merge(data.frame(samples),all.files, by="samples")
dim(samples)

listOfSamples=list()
print(nrow(samples))
for(i in 1:nrow(samples)){
   print(i)
   x=samples[i,2]
   tmp=fread(file = as.character(x))
   listOfSamples[[i]]=tmp[,2]
}
all <- as.data.frame(do.call("cbind", listOfSamples))
all=t(all)
colnames(all)=paste("binned_intensity", seq(1, ncol(all)))
all=as.data.frame(all)
all$sample_id=substring(samples$samples,1, nchar(as.character(samples$samples))-4)

resistance_drug_spectrum=merge(resistance_drug, all, by="sample_id")
dim(resistance_drug_spectrum)

fwrite(resistance_drug_spectrum, paste("/home/gallia/scratch/u230399/DRIAMS-", args[1], "/", args[2], "/zero_shot_spect_drugEmbFingerPrint_Resist_", a, ".csv", sep=""))
}

#Create dataset with random splitting between training and validation set
random1=fread(paste("/home/gallia/scratch/u230399/DRIAMS-", args[1], "/", args[2], "/zero_shot_spect_drugEmbFingerPrint_Resist_train.csv", sep=""))
random2=fread(paste("/home/gallia/scratch/u230399/DRIAMS-", args[1], "/", args[2], "/zero_shot_spect_drugEmbFingerPrint_Resist_val.csv", sep=""))
random=rbind(random1, random2)
smp_size <- floor(0.8 * nrow(random))

train_ind <- sample(seq_len(nrow(random)), size = smp_size)
random_train <- random[train_ind, ]
random_test <- random[-train_ind, ]
fwrite(random_train, paste("/home/gallia/scratch/u230399/DRIAMS-", args[1], "/", args[2], "/random_spect_drugEmbFingerPrint_Resist_train.csv", sep=""))
fwrite(random_test, paste("/home/gallia/scratch/u230399/DRIAMS-", args[1], "/", args[2], "/random_spect_drugEmbFingerPrint_Resist_test.csv", sep=""))
