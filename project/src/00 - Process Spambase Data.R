library(tidyverse)
library(jsonlite)

spam <- read_csv('project/data/original/spambase.data',
                 col_names = FALSE)
col_rename <- read_csv("project/data/processed/column_rename.csv")

# values_refactor <- fromJSON('project/data/processed/refactor_values.json')

colnames(spam) <- col_rename$new_name

spam <- mutate(spam, spam = factor(spam, c(1, 0), c('spam', 'non-spam')))

# Refactor all values from single character into more description words
# for (column in colnames(mushroom)) {
#   mushroom[[column]] <- replace(mushroom[[column]],
#                                 "?" == mushroom[[column]], NA)
#   mushroom[[column]] <- factor(mushroom[[column]],
#                                values_refactor[[column]]$og_value,
#                                values_refactor[[column]]$nw_value)
# }

# Clean up NA. In this case we remove a record if it has NA in any column
spam <- model.frame(spam ~ ., data = spam)

write_csv(spam, "project/data/processed/spam_data_cleaned.csv")

rm(list = ls())
