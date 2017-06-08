
#Remove rows with duplicated MKEY
michigan<-read.csv("michigan.csv", header = T, row.names = 1)
michigan_unique<-michigan[!michigan$MKEY %in% c(michigan$MKEY[duplicated(michigan$MKEY)]),]

#Link diagnostic codes with the main file of michigan data
diagnosticcodes= read.table(file = "diagnosticcodes", sep="|",stringsAsFactors = F)

dx<-data.frame(matrix(ncol=45,nrow=dim(michigan_unique)[1]))
colnames(dx)<-paste(rep("DX_",45),c(1:45),sep="")
for (i in 1:dim(diagnosticcodes)[1])
{
  num = diagnosticcodes[i,3]+1
  id = diagnosticcodes[i,1]
  code = diagnosticcodes[i,2]
  if (id %in% michigan_unique$MKEY ){
    dx[michigan_unique$MKEY==id,num]=code
  }
}
michigan_mapped<-cbind(michigan_unique,dx)

#Remove sensitive information such as dates and zip codes
michigan_deidentified<-michigan_mapped[,-c(1,2,3,7,9,15,16,17,20,22,29,30,36,37,38,39)]
write.csv(michigan_deidentified,file="Michigan/michigan_deidentified.csv",quote = F)
