# R Script used for training and tuning Support Vector Machine ================
# Christopher Lnag
# STA9690 - Advanced Data Mining, Fall 2017
# Final Project
library(tidyverse)
library(e1071)
library(iterators)
library(foreach)
library(snow)
library(doSNOW)
library(caret)
library(ggplot2)

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

# Generate a parameter space for parameter tuning using cross validation
# For combination of:
# cost (see values below)
# gamma (see values below)
svm_fold_params <- (
  list(
    training_size = c(2 * dd_p, 10 * dd_p, floor(dd_n / 2)),
    cost = c(0.01, 0.03, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 200, 500),
    gamma = c(0, 0.001, 0.002, 0.003, 0.004, 0.006, 0.007, 0.008, 0.009, 0.01,
              0.03, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100)
  ) %>%
    expand.grid() %>%
    as.tbl() %>%
    mutate(seeds = sample(100000, n(), replace = F))
)

# Create parallel backend ====
cl <- makeCluster(7)
registerDoSNOW(cl)

# Progress bar ====
pb <- txtProgressBar(max = nrow(svm_fold_params), style = 3, width = 80)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress, preschedule = FALSE)

snow::clusterEvalQ(cl, {
  library(tidyverse)
  library(e1071)
  library(caret)
})

snow::clusterExport(cl, c('spam', 'dd_n', 'dd_p'))

svm_fold_results <- foreach(
  pp = iter(svm_fold_params, by = 'row'),
  .options.snow = opts) %dopar% {
    # Extract work parameters
    tsize <- pp[['training_size']]
    svm_cost <- pp[['cost']]
    svm_gamma <- pp[['gamma']]
    seeds <- pp[['seeds']]

    set.seed(seeds)

    randomized_spam <- dplyr::sample_n(spam, tsize)

    # For each parameter pairs, assess performance with 10-fold cross validation
    fold_indices <- caret::createFolds(randomized_spam$spam, k = 10,
                                       returnTrain = F)

    total_test_result <- list()
    for(fold_name in names(fold_indices)) {
      valid_indices <- fold_indices[[fold_name]]
      learn_indices <- Filter(function(fn) fn != fold_name, names(fold_indices))
      learn_indices <- lapply(learn_indices, function(i) fold_indices[i])
      learn_indices <- unname(unlist(learn_indices))

      learn_dataset <- dplyr::slice(randomized_spam, learn_indices)
      valid_dataset <- dplyr::slice(randomized_spam, valid_indices)

      svm_model <- e1071::svm(spam ~ ., data = learn_dataset, scale = F,
                              type = 'C-classification',
                              kernel = 'radial',
                              gamma = svm_gamma, cost = svm_cost)

      valid_outcome <- valid_dataset[['spam']]
      predi_outcome <- predict(svm_model, newdata = valid_dataset)

      test_result <- (
        pp %>%
          mutate(total = length(valid_outcome),
                 correct = sum(predi_outcome == valid_outcome)) %>%
          mutate(testfold_name = fold_name)
      )

      total_test_result <- c(total_test_result, list(test_result))
    }

    total_test_result <- dplyr::bind_rows(total_test_result)

    return(total_test_result)
  }

svm_fold_results <- bind_rows(svm_fold_results)
svm_fold_results <- (
  svm_fold_results %>%
    mutate(accuracy = correct / total) %>%
    mutate(misclassification = 1 - accuracy)
)

write_csv(svm_fold_results, "project/data/processed/svm_fold_results.csv")

# From the kfold cross validation run, select the best parameters via minimum
# error method and one standard error method
# Will be used in the 100 trail sampling run to get the boxplot visuals of 
# the models stability on out-of-sample data
svm_maxacc_params <- (
  svm_fold_results %>%
    mutate(training_size = factor(training_size, ordered = T)) %>%
    group_by(training_size, cost, gamma) %>%
    summarize(misclassification_sd = sd(misclassification),
              misclassification = mean(misclassification)) %>%
    group_by(training_size) %>%
    # arrange(desc(accuracy)) %>%
    filter(misclassification == min(misclassification)) %>%
    select(-misclassification_sd) %>%
    mutate(param_type = 'minimum_error')
)

svm_1se_params <- (
  svm_fold_results %>%
    mutate(training_size = factor(training_size, ordered = T)) %>%
    group_by(training_size, cost, gamma) %>%
    summarize(misclassification_sd = sd(misclassification),
              misclassification = mean(misclassification),
              k = n())
)

svm_1se_params <- (
  svm_maxacc_params %>%
    rename(best_misclass = misclassification) %>%
    select(training_size, best_misclass) %>%
    right_join(svm_1se_params) %>%
    mutate(threshold = best_misclass + misclassification_sd / sqrt(k)) %>%
    filter(misclassification <= threshold) %>%
    top_n(1, misclassification) %>%
    select(training_size, cost, gamma, misclassification) %>%
    mutate(param_type = 'se1_error')
)


# Start trials for each combinations of training size, parameters
set.seed(67890)

svm_hyperparams <- (
  bind_rows(svm_maxacc_params, svm_1se_params) %>%
    rename(cv_error = misclassification) %>%
    ungroup() %>%
    mutate(trial_id = LETTERS[1:n()])
)

expand_params_indicies <- rep(seq_len(nrow(svm_hyperparams)),
                              rep(ntrials, nrow(svm_hyperparams)))

n_params <- nrow(svm_hyperparams)

svm_hyperparams <- (
  svm_hyperparams %>%
    slice(expand_params_indicies) %>%
    ungroup() %>%
    mutate(trial_id_num = formatC(rep(seq_len(ntrials), n_params),
                                  width = 3, flag = '0')) %>%
    mutate(trial_id = paste0(trial_id, trial_id_num)) %>%
    mutate(seeds = sample(10000, n(), replace = F)) %>%
    select(-trial_id_num, -cv_error)
)

# Progress bar for 100 trial run ====
pb <- txtProgressBar(max = nrow(svm_hyperparams), style = 3, width = 80)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress, preschedule = FALSE)

# 100 trial run execution
trial_results <- foreach(
  pp = iter(svm_hyperparams, by = 'row'),
  .options.snow = opts
) %dopar% {
  tsize <- as.integer(as.character(pp[['training_size']]))
  svm_cost <- pp[['cost']]
  svm_gamma <- pp[['gamma']]
  trial_seed <- pp[['seeds']]

  set.seed(trial_seed)

  learn_ind <- caret::createDataPartition(spam$spam, p = tsize / nrow(spam))
  learn_ind <- learn_ind[[1]]
  valid_ind <- seq_along(spam$spam)[!(seq_along(spam$spam) %in% learn_ind)]

  learn_set <- dplyr::slice(spam, learn_ind)
  valid_set <- dplyr::slice(spam, valid_ind)
  valid_outcomes <- valid_set[['spam']]

  svm_model <- e1071::svm(spam ~ ., data = learn_set, scale = F,
                          type = 'C-classification',
                          kernel = 'radial',
                          gamma = svm_gamma, cost = svm_cost)

  predi_outcome <- predict(svm_model, newdata = valid_set)

  trial_result <- (
    pp %>%
      mutate(total = length(valid_outcomes),
             correct = sum(predi_outcome == valid_outcomes))
  )

  return(trial_result)
}

trial_results <- bind_rows(trial_results)

trial_aggresults <- (
  trial_results %>%
    mutate(accuracy = correct / total) %>%
    mutate(misclassification = 1 - accuracy)
)

write_csv(trial_aggresults, 'project/data/processed/svm_results.csv')

stopCluster(cl)
rm(list = ls())
