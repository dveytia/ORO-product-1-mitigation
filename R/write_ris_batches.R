## Write articles in batches to .ris files

write_ris_batches <- function(df, batchSize, writeDir, fileNamePrefix){
  startInd <- seq(1, nrow(df), by = batchSize) # the start id of the set
  endInd <- startInd+batchSize-1 
  endInd <- ifelse(endInd > nrow(df), nrow(df), endInd)
  
  for(i in 1:length(startInd)){
    temp <- df[startInd[i]:endInd[i],]
    temp <- as.data.frame(temp)
    revtools::write_bibliography(temp, 
                                 filename = file.path(writeDir, paste0(fileNamePrefix,i,".ris")))
  }
}

