library(data.table)
library(stringr)
library(RJSONIO)

setwd("C:/Users/Diane/Documents/MLFPM/Tartu/Hackathon")
data=fromJSON("drugs_2d_structure.json")

for(i in 1:length(data)){
  tmp=data[[i]]
  tmp=data.frame(tmp[[3]]$aid1,tmp[[3]]$aid2, tmp[[3]]$order)
  colnames(tmp)=c("V1", "V2", "V3")
  
  #remove 0s
  row_sub = apply(tmp, 1, function(row) all(row !=0 )) #identify absent edges
  tmp=tmp[row_sub,1:2]  #remove them
    
  #create a number to geneId mapping
  genes=unique(c(tmp$V1,tmp$V2))
  genes=data.frame(seq(0, (length(genes)-1)), genes)
  colnames(genes)=c("nbId", "V1")
    
  #renames genes with nbId
  tmp=merge(genes, tmp, by="V1")
  colnames(genes)=c("nbId", "V2")
  tmp=merge(genes, tmp, by="V2")
  tmp=tmp[,c(2,4)]
  colnames(tmp)=c("V1", "V2")
    
  output=as.factor(paste( "(", tmp$V1, "," , tmp$V2, ")", sep=""))
  output=paste(output, collapse=",")
  output=paste("ind=[", output, "]", sep="")
    
  write(output, paste("ind_", i, ".py", sep=""))
}
