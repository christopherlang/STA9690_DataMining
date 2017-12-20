# R Script used for generating variable importance heatmaps ===================
# Christopher Lnag
# STA9690 - Advanced Data Mining, Fall 2017
# Final Project
library(tidyverse)
library(scales)
library(classInt)

# Boxplots are developed and render with ggplot2 package, within tidyverse

reg_coef <- read_csv('project/data/processed/reg_coefficient_result.csv')

# Collapse lasso/ridge trial results
reg_coef <- (
  reg_coef %>%
    filter(variables != '(Intercept)') %>%
    filter(lambda_type =='lambda_min') %>%
    group_by(training_size, model_type, variables) %>%
    summarize(coef_mean = mean(coef, na.rm = T)) %>%
    ungroup() %>%
    arrange(training_size, model_type, desc(abs(coef_mean))) %>%
    rename(varweight = coef_mean) %>%
    mutate(model_type = replace(model_type, model_type == 'lasso', 'LASSO')) %>%
    mutate(model_type = replace(model_type, model_type == 'ridge', 'RIDGE'))
)

rf_imp <- read_csv('project/data/processed/rf_varimp_results.csv')
result_rf <- read_csv('project/data/processed/rf_results.csv')

best_rf <- (
  result_rf %>%
    group_by(training_size, rf_mtry, rf_ntree, rf_replace) %>%
    summarize(accuracy = mean(accuracy, na.rm = T),
              misclassification = mean(misclassification, na.rm = T)) %>%
    group_by(training_size) %>%
    top_n(1, accuracy) %>%
    select(-accuracy, -misclassification) %>%
    mutate(keep = TRUE)
)

rf_imp <- (
  left_join(rf_imp, best_rf) %>%
    filter(keep) %>%
    select(training_size, variables, MeanDecreaseGini) %>%
    group_by(training_size, variables) %>%
    summarize(MeanDecreaseGini = mean(MeanDecreaseGini, na.rm = T)) %>%
    mutate(model_type = 'Random Forest') %>%
    rename(varweight = MeanDecreaseGini) %>%
    select(training_size, model_type, variables, varweight)
)

varimp_data <- (
  bind_rows(reg_coef, rf_imp) %>%
    mutate(model_type = factor(model_type, c('RIDGE', 'LASSO', 'Random Forest'),
                               ordered = T)) %>%
    group_by(training_size, model_type) %>%
    mutate(varweight = abs(varweight)) %>%
    mutate(varweight = varweight / sum(varweight))
)

# Lets rename the variables so it is shorter in length
varimp_data <- (
  varimp_data %>%
    mutate(variables = gsub('^char_freq_', 'CF_', variables)) %>%
    mutate(variables = gsub('^word_freq_', 'WF_', variables)) %>%
    mutate(variables = gsub('^capital_run_length_', 'CPRL_', variables))
)

# We do column wise hierarchical cluster for visualization purposes
columnclust_reorder <- function(rawdata, seed = 12345) {
  set.seed(seed)

  variables_as_obs <- (
    rawdata %>%
      xtabs(varweight ~ model_type + variables, .) %>%
      t()
  )
  variable_order <- (
    variables_as_obs %>%
      dist(method = 'euclidean') %>%
      hclust(method = 'centroid') %>%
      .[['order']]
  )

  variable_order <- colnames(t(variables_as_obs))[variable_order]

  result <- (
    rawdata %>%
      mutate(variables = factor(variables, variable_order, ordered = T))
  )

  return(result)
}

# Rename training sizes
varimp_data <- (
  varimp_data %>%
    ungroup() %>%
    mutate(training_size = factor(training_size, c(108, 540, 1347),
                                  c('2p, n=108', '10p, n=540', 'n/2, n=1347')))
)

# Generate fill breaks and fill colors
fill_brks <- (
  varimp_data %>%
    .[['varweight']] %>%
    classIntervals(style = 'jenks') %>%
    .[['brks']]
)
fill_cols <- colorRampPalette(c('white', '#759194'))(length(fill_brks) - 1)
fill_brks <- scales::rescale(fill_brks)

testviz <- (
  varimp_data %>%
    columnclust_reorder() %>%
    ggplot(.) + aes(variables, model_type, fill = varweight) +
    geom_point() + geom_tile(colour = "lightgrey") +
    scale_fill_gradientn(colors = fill_cols, values = fill_brks,
                         label = scales::percent,
                         guide = guide_colourbar('Relative Importance')) +
    ggtitle("Variable Importance Agreement",
            "Averaged Coefficients and Mean Decrease Gini") +
    labs(x = 'Variable / Predictors') +
    facet_wrap(~ training_size, nrow = 3) +
    theme(
      panel.background = element_blank(),
      plot.background = element_blank(),
      legend.title = element_text(face = 'bold'),
      legend.position = 'bottom',
      axis.title = element_text(face = 'bold', size = 10),
      axis.title.y = element_blank(),
      axis.text.x = element_text(angle = 45, hjust=0.95, vjust=0.9, size = 6),
      axis.text.y = element_text(size = 10),
      plot.title = element_text(size = 20, face = 'bold'),
      plot.margin = unit(c(0.5, 0.5, 0.25, 0.5), 'inches')
    )
)

ggsave('project/figures/varimp.png', testviz, width = 10,
       height = 6.69, units = 'in', antialias = 'subpixel')

rm(list=ls())
