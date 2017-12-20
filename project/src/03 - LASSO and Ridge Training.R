# R Script used for training and tuning Lasso, Ridge =========================
# Christopher Lnag
# STA9690 - Advanced Data Mining, Fall 2017
# Final Project
library(tidyverse)
library(glmnet)
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

cv_params <- (
  list(
    training_size = c(2 * dd_p, 10 * dd_p, floor(dd_n / 2))
  ) %>%
    expand.grid() %>%
    as.tbl() %>%
    mutate(model_type = 'lasso') %>%
    mutate(seeds = sample(100000, n(), replace = F))
)

cv_params2 <- (
  cv_params %>%
    mutate(seeds = sample(1000, n(), replace = F)) %>%
    mutate(model_type = 'ridge')
)

cv_params <- bind_rows(cv_params, cv_params2)

pb_counter <- 0
pb <- txtProgressBar(max = nrow(cv_params), style = 3, width = 80)

# kfold cross validation execution
# For each training size, randomly sample the data once and use glmnet's
# builtin cross validator to tune the parameters
cv_results <- foreach(pp = iter(cv_params, by = 'row')) %do% {
  tsize <- pp[['training_size']]
  seed <- pp[['seeds']]
  model_type <- pp[['model_type']]
  selection_criteria <- as.character(pp[['selection_criteria']])

  set.seed(seed)

  alpha <- ifelse(model_type == 'ridge', 0, 1)

  learn_indices <- caret::createDataPartition(spam$spam, p = tsize / dd_n)

  learn_set <- slice(spam, learn_indices[[1]])
  learn_outcomes <- learn_set$spam
  learn_set <- select(learn_set, -spam) %>% as.matrix()

  model <-  glmnet::cv.glmnet(learn_set, learn_outcomes, alpha = alpha,
                              nfolds = 10, type.measure = 'class',
                              family = 'binomial')

  cv_lambda_result <- (
    pp %>% mutate(lambda_min = model$lambda.min,
                  lambda_1se = model$lambda.1se)
  )

  n_lambda_results <- length(model$lambda)

  # Also store the CV run for the CV curves later
  cv_curve <- (
    pp %>%
      slice(rep(1, n_lambda_results)) %>%
      mutate(lambda = model$lambda) %>%
      mutate(df = unname(model$nzero)) %>%
      mutate(cvm = model$cvm) %>%
      mutate(cvup = model$cvup) %>%
      mutate(cvlo = model$cvlo) %>%
      mutate(cvsd = model$cvsd)
  )

  pb_counter <- pb_counter + 1
  setTxtProgressBar(pb, pb_counter)

  return(list(lambda_df = cv_lambda_result, cv_curve = cv_curve))
}

cv_lambda <- bind_rows(lapply(cv_results, function(x) x[['lambda_df']]))
cv_curve_data <- bind_rows(lapply(cv_results, function(x) x[['cv_curve']]))

write_csv(cv_curve_data, 'project/data/processed/lassoridge_cvcurve_result.csv')

# Code for model selection via AICC. For each training size, a random sampling
# of the original data is performed, then glmnet is allowed to select the
# lambda, with AICC calculated for all models
reg_params <- cv_params

aic_result <- foreach(
  pp = iter(reg_params, by = 'row')) %do% {
    tsize <- pp[['training_size']]
    seed <- pp[['seeds']]
    model_type <- pp[['model_type']]
    selection_criteria <- as.character(pp[['selection_criteria']])

    set.seed(seed)

    alpha <- ifelse(model_type == 'ridge', 0, 1)

    learn_indices <- caret::createDataPartition(spam$spam, p = tsize / dd_n)

    learn_set <- slice(spam, learn_indices[[1]])
    learn_outcomes <- learn_set$spam
    learn_set <- select(learn_set, -spam) %>% as.matrix()

    model <-  glmnet::glmnet(learn_set, learn_outcomes, alpha = alpha,
                             family = 'binomial')

    aicc <- model$nulldev - deviance(model)
    aicc <- (-aicc + 2 * model$df + 2 * model$df * (model$df + 1) /
               (model$nobs - model$df - 1))


    aicc_lambda_result <- (
      pp %>% mutate(lambda_aicc = model$lambda[which.min(aicc)]) %>%
        mutate(df = model$df[which.min(aicc)])
    )

    return(aicc_lambda_result)
  }

aic_result <- bind_rows(aic_result)
lambda_results <- inner_join(cv_lambda, aic_result)

write_csv(lambda_results,'project/data/processed/lassoridge_lamda_result.csv')

# Code for running AICC model selection via 100 sample trialing
# Mainly used to get data for the Lasso AICC curve
aic_cv_params <- (
  list(
    training_size = c(2 * dd_p, 10 * dd_p, floor(dd_n / 2)),
    trials = seq(1, ntrials)
  ) %>%
    expand.grid() %>%
    as.tbl()
)

set.seed(79685)
aic_curve_result <- foreach(
  pp = iter(aic_cv_params, by = 'row')) %do% {
    tsize <- pp$training_size
    trial_id <- pp$trials
    alpha <- 1

    learn_indices <- caret::createDataPartition(spam$spam, p = tsize / dd_n)

    learn_set <- slice(spam, learn_indices[[1]])
    learn_outcomes <- learn_set$spam
    learn_set <- select(learn_set, -spam) %>% as.matrix()

    model <-  glmnet::glmnet(learn_set, learn_outcomes, alpha = alpha,
                             family = 'binomial')
    aicc <- model$nulldev - deviance(model)
    aicc <- (-aicc + 2 * model$df + 2 * model$df * (model$df + 1) /
               (model$nobs - model$df - 1))


    aicc_result <- data_frame(
      training_size = tsize,
      model_type = 'lasso',
      trial_id = trial_id,
      lambda = model$lambda,
      df = model$df,
      aicc = aicc
    )

    return(aicc_result)
  }

aic_curve_result <- bind_rows(aic_curve_result)
write_csv(aic_curve_result, "project/data/processed/lasso_aic_curve_result.csv")

# Clear space, prep for 100 sampling runs to get data for boxplots
obj_keep <- ls()
obj_keep <- obj_keep[!(obj_keep %in% c('lambda_results', 'spam',
                                       'dd_p', 'dd_n', 'ntrials'))]
rm(list = obj_keep)

set.seed(67789)

trial_params <- (
  lambda_results %>%
    mutate(trial_id = LETTERS[1:n()]) %>%
    slice(rep(seq_len(n()), rep(ntrials, n()))) %>%  # Each case, expand 100
    mutate(seeds = sample(10000, n(), replace = F)) %>%
    group_by(training_size, model_type) %>%
    mutate(trial_id2 = formatC(1:n(), width = 3, flag = '0')) %>%
    mutate(trial_id = paste0(trial_id, trial_id2)) %>%
    ungroup() %>%
    select(-trial_id2) %>%
    gather(lambda_type, lambda, lambda_min, lambda_1se, lambda_aicc)
)

# Create parallel backend ====
cl <- makeCluster(7)
registerDoSNOW(cl)

# Progress bar ====
pb <- txtProgressBar(max = nrow(trial_params), style = 3, width = 80)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress, preschedule = FALSE)

# pb_counter <- 0
# pb <- txtProgressBar(max = nrow(trial_params), style = 3, width = 80)

snow::clusterEvalQ(cl, {
  library(tidyverse)
  library(glmnet)
  library(caret)
})

snow::clusterExport(cl, c('spam', 'dd_n', 'dd_p'))

trials_results <- foreach(
  pp = iter(trial_params, by = 'row'),
  .options.snow = opts,
  .verbose = F) %dopar% {
    tsize <- pp[['training_size']]
    seed <- pp[['seeds']]
    model_type <- pp[['model_type']]
    lambda_type <- pp[['lambda_type']]
    lambda_value <- pp[['lambda']]

    set.seed(seed)

    # Setting glmnet's alpha parameter
    # If the model being fitted is ridge, set alpha = 0
    # If the model being fitted is lasso, set alpha = 1
    alpha <- ifelse(model_type == 'ridge', 0, 1)

    # Generate learn/train dataset indices on each iteration
    # Since the seed is always different between trials, should be random
    # This will learn dataset size that is approximately training_size,
    # mainly due to non-integer result from nrow(spam) * (tize / dd_n)
    # It is usually off by 1
    learn_indices <- caret::createDataPartition(spam$spam, p = tsize / dd_n)

    learn_set <- dplyr::slice(spam, learn_indices[[1]])
    learn_outcomes <- learn_set[['spam']]
    learn_set <- dplyr::select(learn_set, -spam) %>% as.matrix()

    valid_indices <- seq_len(dd_n)[!(seq_len(dd_n) %in% learn_indices)]
    valid_set <- dplyr::slice(spam, valid_indices)
    valid_outcomes <- valid_set[['spam']]
    valid_set <- dplyr::select(valid_set, -spam) %>% as.matrix()

    # For training size n/2, had some convergence problems
    # For quick fix, for when there is a convergence problem for the 1st
    # lambda value (the lambda value we're interested in), set threhold looser
    # to 1e-02 (0.01), which allows the model to converge
    model_fit <- tryCatch({
      model_threshold <<- 1e-07
      m <<- glmnet::glmnet(learn_set, learn_outcomes, family = 'binomial',
                           alpha = alpha, lambda = lambda_value,
                           maxit = 400000)

      list(model = m, threshold = model_threshold)
    }, warning = function(w) {
      model_threshold <<- 1e-02

      m <<- glmnet::glmnet(learn_set, learn_outcomes, family = 'binomial',
                           alpha = alpha, lambda = lambda_value,
                           thres = model_threshold, maxit = 400000)


      list(model = m, threshold = model_threshold)
    })

    model_thres <- model_fit[['threshold']]
    model <- model_fit[['model']]

    predicted_outcomes <- predict(model, newx = valid_set, type = 'class')
    predicted_outcomes <- predicted_outcomes[,1]

    trial_result <- (
      pp %>%
        dplyr::mutate(total = length(valid_outcomes)) %>%
        dplyr::mutate(correct = sum(predicted_outcomes == valid_outcomes))
    )

    coef_result <- dplyr::data_frame(
      variables = names(coef(model)[,1]),
      coef = unname(coef(model)[,1]),
      df = model$df,
      nobs = model$nobs
    )

    coef_result <- (
      coef_result %>%
        dplyr::bind_cols(dplyr::slice(pp, rep(1, nrow(coef_result))))
    )

    return(list(perm = trial_result, coef = coef_result))
  }

coef_result <- bind_rows(lapply(trials_results, function(x) x[['coef']]))
trial_result <- bind_rows(lapply(trials_results, function(x) x[['perm']]))

trial_result <- (
  trial_result %>%
    mutate(accuracy = correct / total) %>%
    mutate(misclassification = 1 - accuracy)
)

write_csv(coef_result, 'project/data/processed/reg_coefficient_result.csv')
write_csv(trial_result, 'project/data/processed/reg_result.csv')

stopCluster(cl)
rm(list=ls())
