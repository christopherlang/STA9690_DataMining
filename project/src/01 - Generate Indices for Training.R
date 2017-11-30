library(tidyverse)

kfold_split <- function(nobs, k = 10, learn_prop = 0.80, seed = 12345,
                          exact = TRUE) {
  if ((nobs * learn_prop / k) %% 1 != 0 & !exact) {
    learn_prop <- (round(nobs * learn_prop / k) * k) / nobs
  }

  valid_prop <- 1 - learn_prop
  learn_size <- ceiling(learn_prop * nobs)
  valid_size <- nobs - learn_size

  set.seed(seed)

  randomized_indices <- sample(seq_len(nobs))

  learn_indices <- randomized_indices[1:learn_size]
  valid_indices <- randomized_indices[(learn_size + 1):nobs]

  learn_indices <- sample(learn_indices)

  cv_ind_brk <- seq(1, learn_size, learn_size / k)
  cv_ind_brk <- c(cv_ind_brk, learn_size + 1)

  fold_indices <- lapply(seq_len(length(cv_ind_brk) - 1), function(i) {
    start_i <- cv_ind_brk[i]
    end_i <- cv_ind_brk[i + 1] - 1

    learn_indices[start_i:end_i]
  })

  names(fold_indices) <- paste0("k", seq_len(k))

  # Create a helper data object so that for a given select kX (e.g. k1, k2, etc)
  # for testing, provide the other k objects as training indices
  kfold_names <- paste0("k", seq_len(k))

  train_test_folds <- list()

  for (ak in kfold_names) {
    train_test_folds[[ak]] <- kfold_names[-which(ak == kfold_names)]

  }

  result <- list()
  result[['learn_indices']] <- learn_indices
  result[['valid_indices']] <- valid_indices
  result[['fold_indices']] <- fold_indices
  result[['traintest_folds']] <- train_test_folds
  result[['info']] <- list(learn_prop = learn_prop, valid_prop = valid_prop,
                           k = k, nobs = nobs, nlearn = length(learn_indices),
                           ntest = length(valid_indices),
                           seed = seed)

  return(result)
}
