
## SET UP -----------------

## Load libraries
library(dplyr)
library(dbplyr)
library(R.utils)
library(RSQLite)
library(ggplot2)
library(tidyverse)

## Set working directory
setwd("C:/Users/deviv/R-working-folder/ORO-product-1-mitigation")

## connect to database
p1_db <- RSQLite::dbConnect(RSQLite::SQLite(),
                            here::here("data","sqlite-databases","product1.sqlite"),
                            create=FALSE)


# Collect table of unique references metadata
uniquerefs <- tbl(p1_db, "uniquerefs") %>%
  select(analysis_id, title, abstract,keywords) %>%
  collect()

# identify references relevant to mitigation
pred_OROmitigation <- tbl(p1_db, "pred_oro_any_mitigation") %>%  # mitigation 
  select(analysis_id, `oro_any.M_Renewables - mean_prediction`, `oro_any.M_Increase_efficiency - mean_prediction`,
         `oro_any.M_CO2_removal_or_storage - mean_prediction`) %>%
  collect()

# Filter to articles relevant for at least 1 mitigation ORO
pred_OROmitigation <- pred_OROmitigation %>%
  mutate(n_OROs = rowSums(.[2:4] > 0.5)) %>% 
  filter(1 <= n_OROs) %>%
  select(analysis_id) %>%
  left_join(uniquerefs, by = "analysis_id")

#sum(is.na(pred_OROmitigation$abstract)) # no NA abstracts = 0


# Create a text column to search for keywords
pred_OROmitigation$text <- apply(pred_OROmitigation[,c("title","abstract","keywords")],
                                 1, 
                                 paste,
                                 collapse = " ")

pred_OROmitigation <- pred_OROmitigation %>%
  select(-c(title, abstract, keywords))

# Disconnect from database
dbDisconnect(p1_db)



## FORMAT FOR KEYWORD SEARCH ----------------------------

# NLP functions
functionsToSource <- c("clean_string.R", "screen.R","tokenization.R","utils.R", "bool_detect.R")
for(i in 1:length(functionsToSource)){
  source(here::here("R", functionsToSource[i]))
}


## Search queries 
queries <- list(
  CCS = '(carbon AND capture) OR (CO2 AND capture) OR (stor* AND sediment)',
  CDR_BC = 'mangrove OR "salt marsh" OR "tidal marsh" OR seagrass OR (carbon AND sequest*) OR "blue carbon"',
  CDR_Cult = 'macroalg* OR microalg* OR seaweed OR kelp',
  CDR_BioPump = '"iron fertilization" OR "artificial upwell*" OR "biological carbon pump"',
  CDR_OAE = 'alkal* AND enhanc*',
  CDR_Other = '"carbon dioxide remov*" OR "CO2 remov*" OR (electrochem* AND carbon)',
  MRE_Bio = 'bioethanol AND (marine OR ocean OR kelp OR seaweed OR macroalgae OR phytoplankton OR microalgae)'
)


## Clean text from punctuation but keep numbers
clean_string2 <- function(x) {
  is_character(x)
  x <- gsub("\n", " ", x)          # Remove new lines
  x <- gsub("- ", "-", x)          # Deal with caesura
  x <- gsub("[[:punct:]]", " ", x) # Remove punctuation
  #x <- gsub("[0-9]", "", x)        # Remove numbers
  x <- gsub("\\s+", " ", x)        # Remove whitespaces
  x <- gsub("^\\s|\\s$", "", x)    # Remove whitespaces
  
  x <- tolower(x)
  x 
}



## LOOP THROUGH DOCMENTS AND QUERIES AND CONDUCT SEARCH -----------------
results = parallel::mclapply(1:nrow(pred_OROmitigation), function(i){
  
  print(paste(i, "document/", nrow(pred_OROmitigation)))
  # process text
  # group title, abstract together
  text = pred_OROmitigation$text[i]
  
  # clean string to remove extra spaces and punctuation
  text <- clean_string2(text)
  
  # screen for keywords
  screens_vec = vector('logical', length = length(queries))
  for(q in 1:length(queries)){
    screens_vec[q] <- tryCatch(
      bool_detect2(text, queries[[q]]),
      error = function(e) return(NA)
    )
  }
  screens_vec
  
  ## Bind both together
  screens_all <- data.frame(analysis_id = pred_OROmitigation$analysis_id[i],
                            ORO_type = names(queries),
                            query = unlist(queries),
                            query_match = screens_vec) 
  
  # return the dataframe
  return(screens_all)
  
}, mc.cores = 1)


## Bind results together
screens <- do.call(rbind.data.frame, results)

## write
p1_db <- RSQLite::dbConnect(RSQLite::SQLite(),
                            here::here("data","sqlite-databases","product1.sqlite"),
                            create=FALSE)
DBI::dbWriteTable(p1_db, "CCS_CDR_keywordMatches_v2", screens, append=FALSE, overwrite = TRUE)
DBI::dbDisconnect(p1_db)




