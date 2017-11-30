library(tidyverse)


train_valid_split <- function(nobs, classes, train_prop = 0.80, seed = 12345) {
  set.seed(seed)

  class_stratified <- split(seq_len(nobs), classes)

  stratified_indices <- lapply(class_stratified, function(x) {
    x <- sample(x)

    if (length(x) * train_prop != 0) {
      adjusted_splits <- length(x) * train_prop
      adjusted_splits <- floor(adjusted_splits)

      train_indices <- x[1:adjusted_splits]
      valid_indices <- x[(adjusted_splits + 1):length(x)]
    } else {
      train_indices <- x[1:(length(x) * train_prop)]
      valid_indices <- x[((length(x) * train_prop) + 1):length(x)]
    }

    return(list(train_i = train_indices, valid_i = valid_indices))
  })

  train_indices <- c(stratified_indices[['spam']][['train_i']],
                     stratified_indices[['non-spam']][['train_i']])
  valid_indices <- c(stratified_indices[['spam']][['valid_i']],
                     stratified_indices[['non-spam']][['valid_i']])

  result <- list()
  result[['training_indices']] <- train_indices
  result[['validation_indices']] <- valid_indices

  return(result)
}


kfold_split <- function(nobs, classes, k = 10, seed = 12345) {
  x <- seq_len(nobs)

  set.seed(seed)
  i_shuffled <- sample(x)
  x <- x[i_shuffled]
  classes <- classes[i_shuffled]
  class_stratified <- split(seq_len(nobs), classes)

  kfold_splitter <- function(x) {
    split_factor <- as.character(cut(seq_along(x), k, labels = FALSE))

    split(x, split_factor)
  }

  if (k >= nobs) {
    # Assume LOOCV
    fold_indices <- kfold_splitter(x)
  } else {
    stratified_indices <- lapply(class_stratified, kfold_splitter)

    fold_indices <- mapply(function(x, y) c(x, y), stratified_indices$spam,
                           stratified_indices[['non-spam']])
  }

  names(fold_indices) <- paste0("k", names(fold_indices))

  # Create a helper data object so that for a given select kX (e.g. k1, k2, etc)
  # for testing, provide the other k objects as training indices
  kfold_names <- names(fold_indices)

  train_test_folds <- list()

  for (ak in kfold_names) {
    train_test_folds[[ak]] <- kfold_names[-which(ak == kfold_names)]

  }

  result <- list()

  result[['fold_indices']] <- fold_indices
  result[['traintest_folds']] <- train_test_folds
  result[['info']] <- list(k = k, nobs = nobs, seed = seed)

  return(result)
}


expand_cv_search <- function(params, n, seed = 10000) {
  full_grid_search <- expand.grid(params)
  full_grid_search$grid_id <- seq_len(nrow(full_grid_search))
  full_grid_search$grid_id <- formatC(full_grid_search$grid_id,
                                      width = nchar(n), flag = '0')

  rep_i <- rep(seq_len(nrow(full_grid_search)), rep(n, nrow(full_grid_search)))

  full_grid_search <- as.tbl(full_grid_search[rep_i,])

  set.seed(seed)

  full_grid_search$seed <- sample(seq_len(n)) * runif(n, min = 10, max = 50)

  return(full_grid_search)
}
