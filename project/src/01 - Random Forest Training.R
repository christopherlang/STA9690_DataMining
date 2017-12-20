# R Script used for training and tuning Random Forest =========================
# Christopher Lnag
# STA9690 - Advanced Data Mining, Fall 2017
# Final Project
library(tidyverse)
library(randomForest)
library(iterators)
library(foreach)
library(snow)
library(doSNOW)
library(caret)

ntrials <- 100

spam <- (
  read_csv("project/data/processed/spam_data_cleaned.csv") %>%
    mutate(spam = factor(spam, c('spam', 'non-spam'))) %>%
    select(-word_freq_3d, -word_freq_table, -word_freq_parts) %>%  # All zeros
    mutate_at(vars(-spam), funs(scale(.)))  # scale all features to z-score
)

dd_n <- nrow(spam)  # this is n, # of records/observations/units/etc.
dd_p <- ncol(spam) - 1  # this is p, # of features, -1 for response column

set.seed(12345)

# Generate a parameter space for training and testing
# For combination of:
# Training size (2p, 10p, n/2)
# mtry (1 through sqrt(p) + 2)
# ntree 10, 50, 100, 500, 1000
# replace TRUE or FALSE
# Trial 1 through 100
rf_params <- (
  list(
    training_size = c(2 * dd_p, 10 * dd_p, floor(dd_n / 2)),
    trial_i = 1:ntrials,
    rf_mtry = 1:(ceiling(sqrt(dd_p)) + 2),
    rf_ntree = c(10, 50, 100, 500, 1000),
    rf_replace = c(TRUE, FALSE)
  ) %>%
    expand.grid() %>%
    as.tbl() %>%
    mutate(trial_seed = sample(100000, n(), replace = F))
)

# Create parallel backend ====
cl <- makeCluster(7)
registerDoSNOW(cl)

# Progress bar ====
pb <- txtProgressBar(max = nrow(rf_params), style = 3, width = 80)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress, preschedule = FALSE)

# Modeling code ==============================================================
rf_results <- foreach(
  pp = iter(rf_params, by = 'row'),
  .export = c('spam', 'dd_n', 'dd_p', 'ntrials'),
  .packages = c('caret', 'tidyverse', 'randomForest'),
  .options.snow = opts) %dopar% {
    # Extract work parameters
    trial_i <- pp[['trial_i']]
    rf_ntree <- pp[['rf_ntree']]
    rf_replace <- pp[['rf_replace']]
    rf_mtry <- pp[['rf_mtry']]
    trial_seed <- pp[['trial_seed']]

    set.seed(trial_seed)

    randomized_indices <- sample(seq_len(dd_n))
    training_indices <- randomized_indices[1:pp$training_size]
    testing_indices <- randomized_indices[(pp$training_size + 1):dd_n]

    training_set <- spam[training_indices,]
    testing_set <- spam[testing_indices,]
    testing_outcome <- testing_set$spam

    rf_model <- randomForest::randomForest(spam ~ ., data = training_set,
                                           ntree = rf_ntree,
                                           replace = rf_replace,
                                           mtry = rf_mtry)

    predicted_outcome <- predict(rf_model, testing_set)

    varimpdf <- randomForest::importance(rf_model)
    varimpdf <- (
      dplyr::as_data_frame(varimpdf) %>%
        dplyr::mutate(variables = rownames(varimpdf)) %>%
        dplyr::select(variables, everything())
    )
    varimpdf <- dplyr::bind_cols(
      dplyr::slice(pp, rep(1, nrow(varimpdf))), varimpdf
    )

    trial_result <- (
      pp %>%
        dplyr::mutate(total = nrow(testing_set)) %>%
        dplyr::mutate(correct = sum(predicted_outcome == testing_outcome)) %>%
        dplyr::mutate(accuracy = correct / total) %>%
        dplyr::mutate(misclassification = 1 - accuracy) %>%
        dplyr::mutate(trial_run = pp[['trial_i']]) %>%
        dplyr::mutate(seed = trial_seed)
    )

    return(list(trial_result = trial_result, varimpdf = varimpdf))
  }

trial_results <- bind_rows(lapply(rf_results, function(x) x$trial_result))
varimp_result <- bind_rows(lapply(rf_results, function(x) x$varimpdf))

write_csv(trial_results, "project/data/processed/rf_results.csv")
write_csv(varimp_result, "project/data/processed/rf_varimp_results.csv")

stopCluster(cl)
rm(list = ls())
