library(dplyr)
library(tidyr)
df = tmp
format_4_distilBert <- function(df, vars2pivot, idVar) {
  # Convert boolean columns to binary (1/0)
  df <- df %>%
    mutate(across(all_of(vars2pivot), ~ ifelse(. == "true", 1, ifelse(. == "false", 0, .)))) 

  # Identify multi-label columns (those that need to be split)
  multi_label_cols <- vars2pivot[vars2pivot %in% colnames(df) & sapply(df[vars2pivot], function(x) any(grepl("|", x, fixed = TRUE)))]
  df[,multi_label_cols] <- apply(df[,multi_label_cols], 1:2, function(x){return(gsub(",,,",",", gsub("\\|",",",x)))})
  
  # ensure binary columns are integer not str
  df[,vars2pivot[!(vars2pivot %in% multi_label_cols)]] <- apply(df[,vars2pivot[!(vars2pivot %in% multi_label_cols)]], 1:2, as.integer)
  
  summary(df)
  # Process multi-label columns
  for (col in multi_label_cols) {
    # Separate values into rows using correct separator
    df <- df %>%
      separate_rows(!!sym(col), sep = ",", convert = TRUE) %>%
      mutate(!!sym(col) := trimws(!!sym(col))) # Trim spaces
      # filter(!is.na(!!sym(col))) 
    # Convert to binary wide format for the specific column
    df <- df %>%
      mutate(value = 1) %>%
      pivot_wider(names_from = !!sym(col), values_from = value, 
                  names_prefix = paste0(col, "."), values_fill = list(value = 0))
  }
  summary(df)
  
  
  # Ensure binary format (convert counts to 1 where applicable)
  # df <- df %>% mutate(across(-all_of(idVar), ~ ifelse(. > 0, 1, .)))
  
  # Keep only unique idVar rows with processed binary columns
  df <- df %>% group_by(across(all_of(idVar))) %>% summarise(across(everything(), max), .groups = "drop")
  
  # Remove NA columns
  df <- df[,-grep(".NA", colnames(df))]
  
  # return only requested columns
  df <- df[,grep(paste(c(idVar, vars2pivot), collapse="|"), colnames(df))] %>%
    relocate(all_of(idVar))
  
  
  return(df)
}