# R Script used for generating box plots =====================================
# Christopher Lnag
# STA9690 - Advanced Data Mining, Fall 2017
# Final Project
library(tidyverse)
library(scales)

# Boxplots are developed and render with ggplot2 package, within tidyverse

# For Random Forest
result_rf <- read_csv('project/data/processed/rf_results.csv')

best_rf <- (
  result_rf %>%
    group_by(training_size, rf_mtry, rf_ntree, rf_replace) %>%
    summarize(accuracy = mean(accuracy, na.rm = T),
              misclassification = mean(misclassification, na.rm = T)) %>%
    group_by(training_size) %>%
    top_n(1, accuracy)
)

boxplot_data_rf <- list()
for (i in seq_len(nrow(best_rf))) {
  boxplot_data_rf[[i]] <- (
    filter(result_rf, (training_size == best_rf[i,][['training_size']] &
                         rf_mtry == best_rf[i,][['rf_mtry']] &
                         rf_ntree == best_rf[i,][['rf_ntree']] &
                         rf_replace == best_rf[i,][['rf_replace']]))
  )
}
boxplot_data_rf <- (
  bind_rows(boxplot_data_rf) %>%
    select(-trial_i, -total, -correct, -trial_run, -seed, -accuracy)
)

boxplot_params_rf <- (
  select(boxplot_data_rf, -misclassification) %>%
    distinct()
)

boxplot_data_rf <- (
  select(boxplot_data_rf, training_size, misclassification) %>%
    mutate(model_type = 'Random Forest')
)

rm(i, best_rf, result_rf)

# For Radial SVM
result_svm <- read_csv('project/data/processed/svm_results.csv')

boxplot_params_svm <- (
  result_svm %>%
    select(training_size, cost, gamma, param_type) %>%
    distinct()
)

boxplot_data_svm <- (
  result_svm %>%
    select(training_size, misclassification, param_type) %>%
    rename(model_type = param_type) %>%
    mutate(model_type = ifelse(model_type == 'minimum_error', '', '-1SE')) %>%
    mutate(model_type = paste0('SVM', model_type))
)

rm(result_svm)

# For Lasso and Ridge
result_reg <- read_csv('project/data/processed/reg_result.csv')

boxplot_params_reg <- (
  result_reg %>%
    select(training_size, model_type, lambda_type, lambda) %>%
    distinct()
)

boxplot_data_reg <- (
  result_reg %>%
    mutate(model_type = ifelse(model_type == 'lasso', 'LASSO', 'RIDGE')) %>%
    mutate(lambda_type = replace(lambda_type, lambda_type == 'lambda_min', '')) %>%
    mutate(lambda_type = replace(lambda_type, lambda_type == 'lambda_1se', '-1SE')) %>%
    mutate(lambda_type = replace(lambda_type, lambda_type == 'lambda_aicc', '-AIC')) %>%
    mutate(model_type = paste0(model_type, lambda_type)) %>%
    select(training_size, misclassification, model_type)
)

rm(result_reg)

# Combine all boxplot data
boxplot_data <- rbind(boxplot_data_rf, boxplot_data_reg, boxplot_data_svm)

model_order <- c('Random Forest', 'SVM', 'SVM-1SE', "LASSO", "LASSO-1SE",
                 "LASSO-AIC", "RIDGE", "RIDGE-1SE","RIDGE-AIC")

train_size_names <- c('Size 2p (p=54), nlearn=108, (p=54)',
                      'Size 10p , nlearn=540, (p=54)',
                      'Size n/2, nlearn=1347, (p=54)')

boxplot_data <- (
  boxplot_data %>%
    mutate(model_type = factor(model_type, model_order, ordered = T)) %>%
    mutate(tsize_type = factor(training_size, c(108, 540, 1347), c('2p', '10p', 'n/2'))) %>%
    mutate(tsize_size = factor(training_size, c(108, 540, 1347), c('nlearn=108',
                                                                   'nlearn=540',
                                                                   'nlearn=1347')))
)

# For each training size, render a one page box plot
sized_boxplots <- by(boxplot_data, boxplot_data$tsize_type, function(x) {
  x <- dplyr::filter(x, model_type != 'RIDGE-AIC')
  plot_title <- (
    paste0("Misclassification Error for Learning Size ", unique(x$tsize_type), ", ", unique(x$tsize_size))
  )
  plot_subtitle <- (
    paste0('Parameters tuned through a single 10-fold Cross Validation run')
  )
  ggplot(x) + aes(model_type, misclassification, group = model_type) +
    geom_boxplot(aes(fill = model_type)) +
    scale_y_continuous(labels = scales::percent) +
    labs(x = 'Models with Selection Method (no method means minimum error)',
         y = 'Misclassification Error (% incorrect)') +
    ggtitle(plot_title, plot_subtitle) +
    scale_fill_manual(
      values = c(
        "Random Forest" = "#759194",
        "SVM" = "#1F69D1",
        "SVM-1SE" = "#1F69D1",
        "LASSO" = "#a45664",
        "LASSO-1SE" = "#a45664",
        "LASSO-AIC" = "#a45664",
        "RIDGE-1SE" = "#e56a00",
        "RIDGE" = "#e56a00",
        "RIDGE-1SE" = "#e56a00",
        "RIDGE-AIC" = "#e56a00"
      )) +
    theme_bw() +
    theme(
      axis.title = element_text(face = 'bold'),
      axis.title.x = element_text(vjust = -2),
      axis.title.y = element_text(vjust = 2),
      legend.position = 'none',
      panel.grid = element_blank(),
      plot.title = element_text(size = 20, face = 'bold'),
      plot.margin = unit(c(0.5, 0.5, 0.25, 0.5), 'inches')
    )
})

ggsave('project/figures/boxplots_2p.png', sized_boxplots[[1]], width = 10,
       height = 7.5, units = 'in', antialias = 'subpixel')

ggsave('project/figures/boxplots_10p.png', sized_boxplots[[2]], width = 10,
       height = 7.5, units = 'in', antialias = 'subpixel')

ggsave('project/figures/boxplots_0.5n.png', sized_boxplots[[3]], width = 10,
       height = 7.5, units = 'in', antialias = 'subpixel')

# Similar to above, but now we facet the boxplots so that they appear on one
# page, with the exact same x-axis and y-axis ranges
plot_title <- (
  paste0("Misclassification Error for All Learning Data Sizes")
)
plot_subtitle <- (
  paste0('Parameters tuned through a single 10-fold Cross Validation run')
)

combined_boxplots <- (
  boxplot_data %>%
    filter(model_type != 'RIDGE-AIC') %>%
    mutate(facet_name = paste(tsize_type, tsize_size, sep = ", ")) %>%
    mutate(facet_name = factor(facet_name, c('2p, nlearn=108', '10p, nlearn=540',
                                             'n/2, nlearn=1347'),
                               c('2p, nlearn=108', '10p, nlearn=540',
                                 'n/2, nlearn=1347'), order = T)) %>%
    ggplot() + aes(model_type, misclassification, group = model_type) +
    geom_boxplot(aes(fill = model_type)) +
    facet_wrap(~ facet_name) +
    scale_y_continuous(labels = scales::percent) +
    labs(x = 'Models with Selection Method (no method means minimum error)',
         y = 'Misclassification Error (% incorrect)') +
    ggtitle(plot_title, plot_subtitle) +
    scale_fill_manual(
      values = c(
        "Random Forest" = "#759194",
        "SVM" = "#1F69D1",
        "SVM-1SE" = "#1F69D1",
        "LASSO" = "#a45664",
        "LASSO-1SE" = "#a45664",
        "LASSO-AIC" = "#a45664",
        "RIDGE-1SE" = "#e56a00",
        "RIDGE" = "#e56a00",
        "RIDGE-1SE" = "#e56a00",
        "RIDGE-AIC" = "#e56a00"
      )) +
    theme_bw() +
    theme(
      axis.title = element_text(face = 'bold'),
      axis.title.x = element_text(vjust = -2),
      axis.title.y = element_text(vjust = 2),
      legend.position = 'none',
      panel.grid = element_blank(),
      plot.title = element_text(size = 20, face = 'bold'),
      plot.margin = unit(c(0.5, 0.5, 0.25, 0.5), 'inches'),
      axis.text.x = element_text(angle = 45, hjust=0.95, vjust=0.9, size = 6)
    )
)

ggsave('project/figures/boxplots_all.png', combined_boxplots, width = 10,
       height = 7.5, units = 'in', antialias = 'subpixel')

rm(list=ls())
