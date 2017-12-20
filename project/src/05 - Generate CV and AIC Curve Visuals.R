# R Script used for generating CV and AICC Curves =============================
# Christopher Lnag
# STA9690 - Advanced Data Mining, Fall 2017
# Final Project
library(tidyverse)
library(scales)

# Boxplots are developed and render with ggplot2 package, within tidyverse

# CV Curve for Lasso, Ridge
reg_curves <- read_csv('project/data/processed/lassoridge_cvcurve_result.csv')
params <- read_csv("project/data/processed/lassoridge_lamda_result.csv")

# Say we're only plotting the curve fo rn/2
reg_curves <- (
  reg_curves %>%
    select(-seeds, -cvsd)
)

reg_curves <- (
  reg_curves %>%
    mutate(training_size = factor(training_size, c(108, 540, 1347),
                                  c('2p, n=108', '10p, n=540', 'n/2, n=1347')))
)

# Create data for adding in degrees of freedom values in charts
# We only select a few, with regular intervals, to avoid overplotting DF #s
df_labels <- (
  reg_curves %>%
    select(training_size, model_type, df, lambda, cvm) %>%
    arrange(training_size, model_type, lambda) %>%
    group_by(training_size, model_type) %>%
    slice(seq(1, n(), by = 5))
)

# Create curve for Ridge regression CV
param_data <- gather(params, lambda_type, lambda, starts_with('lambda'))
param_data <- filter(param_data, lambda_type != 'lambda_aicc')
param_data <- (
  param_data %>%
    mutate(lambda_type = ifelse(lambda_type == 'lambda_min', 'Minimum error',
                                '1 Standard Error Rule')) %>%
    mutate(training_size = factor(training_size, c(108, 540, 1347),
                                  c('2p, n=108', '10p, n=540', 'n/2, n=1347')))
)

# For each curve, need to develop custom x-axis breaks, along with formatted
# actual lambda values to show (we're transforming x-axis onto log scale)
ridge_only_curve <- filter(reg_curves, model_type == 'ridge')
lambda_brks <- classInt::classIntervals(ridge_only_curve$lambda, n = 7,
                                        style = 'quantile')$brks

lambda_brks <- round(lambda_brks, 2)
lambda_brks[lambda_brks > 1] <- round(lambda_brks[lambda_brks > 1])
lambda_brks_labels <- as.character(lambda_brks)
lambda_brks_labels <- gsub('[.]00$', '', lambda_brks_labels)

# Extract degrees of freedom labels for specific model
df_labs <- filter(df_labels, model_type == 'ridge')

cv_curve <- (
  reg_curves %>%
    filter(model_type == 'ridge') %>%
    ggplot(.) + aes(lambda, cvm, group = model_type) +
    geom_errorbar(aes(ymin = cvlo, ymax = cvup), color = 'darkgrey') +
    geom_point(color = '#e23528', fill = '#e23528', shape = 21, size = 2) +
    facet_wrap(~ training_size, ncol = 1) +
    scale_x_log10(breaks = lambda_brks, labels = lambda_brks_labels) +
    scale_y_continuous(labels = scales::percent, limits = c(NA, 0.6)) +
    geom_vline(aes(xintercept = lambda, group = lambda_type, color = lambda_type),
               data = filter(param_data, model_type == 'ridge'), show.legend = T) +
    labs(x = 'Lambda') +
    geom_text(aes(label = df), y = 0.55, data = df_labs, size = 2.3) +
    theme_bw() +
    theme(
      axis.title = element_blank(),
      axis.text.x = element_text(size = 6),
      legend.position = 'none',
      panel.grid = element_blank(),
      plot.margin = unit(c(1.2, 0.05, 0.05, 0.05), 'inches')
    )
)

ggsave('project/figures/cv_ridge_curve.png', cv_curve, width = 10 / 3,
       height = 7.5, units = 'in', antialias = 'subpixel')

# Render LASSO CV curves
lasso_only_curve <- filter(reg_curves, model_type == 'lasso')
lambda_brks <- classInt::classIntervals(lasso_only_curve$lambda, n = 5,
                                        style = 'quantile')$brks

lambda_brks <- round(lambda_brks, 6)
lambda_brks_labels <- scales::scientific(lambda_brks)

df_labs <- filter(df_labels, model_type == 'lasso')

cv_curve <- (
  reg_curves %>%
    filter(model_type == 'lasso') %>%
    ggplot(.) + aes(lambda, cvm, group = model_type) +
    geom_errorbar(aes(ymin = cvlo, ymax = cvup), color = 'darkgrey') +
    geom_point(color = '#e23528', fill = '#e23528', shape = 21, size = 2) +
    # facet_grid(training_size ~ model_type, scales = 'free') +
    facet_wrap(~ training_size, ncol = 1) +
    scale_x_log10(breaks = lambda_brks, labels = lambda_brks_labels) +
    scale_y_continuous(labels = scales::percent, limits = c(NA, 0.6)) +
    geom_vline(aes(xintercept = lambda, group = lambda_type, color = lambda_type),
               data = filter(param_data, model_type == 'lasso'), show.legend = T) +
    labs(x = 'Lambda') +
    geom_text(aes(label = df), y = 0.55, data = df_labs, size = 2.3) +
    # ggtitle("Ridge Regression 10-fold Cross Validation Curve") +
    theme_bw() +
    theme(
      axis.title = element_blank(),
      axis.text.x = element_text(size = 6),
      legend.position = 'none',
      panel.grid = element_blank(),
      plot.margin = unit(c(1.2, 0.05, 0.05, 0.05), 'inches')
    )
)

ggsave('project/figures/cv_lasso_curve.png', cv_curve, width = 10 / 3,
       height = 7.5, units = 'in', antialias = 'subpixel')


# AIC curve for LASSO
# Load LASSO AIC Curve, combine with CV curves
lasso_aic <- read_csv("project/data/processed/lasso_aic_curve_result.csv")
lasso_aic <- (
  lasso_aic %>%
    mutate(lambda = round(lambda, 3)) %>%
    group_by(lambda, df, training_size) %>%
    summarize(mean_aicc = mean(aicc, na.rm = T)) %>%
    mutate(aicc_lo = mean_aicc,
           aicc_up = mean_aicc) %>%
    select(training_size, lambda, df, mean_aicc, aicc_lo, aicc_up) %>%
    mutate(training_size = factor(training_size, c(108, 540, 1347),
                                  c('2p, n=108', '10p, n=540', 'n/2, n=1347')))
)

lambda_brks <- classInt::classIntervals(lasso_aic$lambda, n = 4,
                                        style = 'quantile')$brks

lambda_brks <- round(lambda_brks, 2)
lambda_brks_labels <- scales::scientific(lambda_brks)

df_labs <- (
  lasso_aic %>%
    select(training_size, df, lambda, mean_aicc) %>%
    arrange(training_size, lambda) %>%
    group_by(training_size) %>%
    slice(seq(1, n(), length.out = 10))
)

lass_aic_curve <- (
  lasso_aic %>%
    ggplot(.) + aes(lambda, mean_aicc) +
    geom_errorbar(aes(ymin = aicc_lo, ymax = aicc_up), color = 'darkgrey') +
    geom_point(color = '#e23528', fill = '#e23528', shape = 21, size = 2) +
    scale_x_log10(breaks = lambda_brks, labels = lambda_brks_labels) +
    scale_y_continuous(labels = scales::comma, limits = c(NA, 300)) +
    facet_wrap(~ training_size, ncol = 1) +
    geom_text(aes(label = df), y = 0.9166667 * 300, data = df_labs, size = 2.3) +
    theme_bw() +
    theme(
      axis.title = element_blank(),
      axis.text.x = element_text(size = 6),
      legend.position = 'none',
      panel.grid = element_blank(),
      plot.margin = unit(c(1.2, 0.05, 0.05, 0.05), 'inches')
    )
)

ggsave('project/figures/lasso_aic_curve.png', lass_aic_curve, width = 10 / 3,
       height = 7.5, units = 'in', antialias = 'subpixel')
