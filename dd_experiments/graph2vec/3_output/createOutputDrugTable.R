library(data.table)
library(stringr)
library(RJSONIO)

setwd("C:/Users/Diane/Documents/MLFPM/Tartu/Hackathon")
data=fromJSON("drugs_2d_structure.json")

output=matrix(nrow=54, ncol=125)
for(i in 1:54){
  drug=as.matrix((fread(paste("results", i, ".txt", sep=""))))
  drug=na.omit(c(drug))
  drug=sub("]]", "", drug, fixed = TRUE)
  drug=sub("[[", "", drug, fixed = TRUE)
  drug=as.numeric(drug)
  output[i,]=c(names(data[i]), drug)
}

fwrite(output, "drug_Graph2vec_embedding.txt")

