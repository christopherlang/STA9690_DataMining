library(tidyverse)
library(randomForest)
library(iterators)
library(foreach)
library(snow)
library(doSNOW)

ntrials <- 100
p <- 57

source("project/lib/utils.R", T)

spam <- read_csv("project/data/processed/spam_data_cleaned.csv")
spam$spam <- factor(spam$spam, c('spam', 'non-spam'))

set.seed(123)
seeds_for_runs <- sample.int(10000, size = ntrials)

params <- list(mtry = c(5:10), ntree = 500, replace = c(T, F), multi = c(2, 10),
               kfold = c('kLOOCV', "k10"))

grid_params <- expand_cv_search(params, n = ntrials)
nruns <- nrow(grid_params)
grid_params <- iter(grid_params, by = 'row')

cl <- makeCluster(7)
registerDoSNOW(cl)

pb <- txtProgressBar(max = nruns, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

r <- foreach(
  mp = grid_params, .options.snow = opts,
  .packages = c('tidyverse', 'randomForest'),
  .export = c('spam', 'kfold_split', 'train_valid_split')) %dopar% {

  trial_seed <- mp$seed
  splits <- train_valid_split(nrow(spam), spam$spam,
                              train_prop = (mp$multi * p) / nrow(spam),
                              seed = trial_seed)

  learn_indices <- splits$training_indices
  vl_indices <- splits$validation_indices

  if (mp$kfold == 'kLOOCV') {
    kfold <- length(learn_indices)
  } else {
    kfold <- as.integer(gsub('k', '', mp$kfold))
  }

  learn_dataset <- spam[learn_indices,]
  fold_splits <- kfold_split(nrow(learn_dataset), learn_dataset$spam, kfold,
                             seed = trial_seed)

  fold_indices <- fold_splits$fold_indices
  fold_selection <- fold_splits$traintest_folds

  rf_cv_results <- lapply(names(fold_selection), function(fold_name) {
    training_indices <- fold_selection[[fold_name]]
    training_indices <- lapply(training_indices, function(x) fold_indices[[x]])
    training_indices <- unlist(training_indices)
    training_dataset <- learn_dataset[training_indices,]

    testing_indices <- fold_indices[[fold_name]]
    testing_dataset <- spam[testing_indices,]

    model <- randomForest(spam ~ ., data = training_dataset, ntree = mp$ntree,
                          replace = mp$replace, mtry = mp$mtry)

    model_stats <- data_frame(
      fold = fold_name,
      mtry = mp$mtry,
      replace = mp$replace,
      ntree = mp$ntree
    )

    predictions <- unname(predict(model, newdata = testing_dataset))
    correct_freq <- table(testing_dataset$spam == predictions)
    correct <- correct_freq['TRUE']
    incorrect <- correct_freq['FALSE']
    total_predict <- sum(correct_freq)

    model_stats$correct <- correct
    model_stats$incorrect <- incorrect
    model_stats$total_predict <- total_predict
    model_stats$multi <- mp$multi

    return(model_stats)
  })

  rf_cv_results <- (
    bind_rows(rf_cv_results) %>%
      mutate_at(vars(correct, incorrect, total_predict),
                funs(replace(., is.na(.), 0))) %>%
      mutate(fold = as.integer(gsub('k', '', fold))) %>%
      arrange(fold) %>%
      mutate(fold = paste0('k', fold))
  )
  rf_cv_results$grid_id <- mp$grid_id

  return(rf_cv_results)
  }

r <- bind_rows(r)

group_by(rf_cv_results, fold, dataset_type, correct) %>%
  summarize(n = n())
