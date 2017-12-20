# R Script used for generating SVM heatmaps ===================================
# Christopher Lnag
# STA9690 - Advanced Data Mining, Fall 2017
# Final Project
library(tidyverse)
library(scales)

# Boxplots are developed and render with ggplot2 package, within tidyverse

svm_fold_results <- read_csv("project/data/processed/svm_fold_results.csv")

data_for_heatmap <- (
  svm_fold_results %>%
    group_by(training_size, cost, gamma) %>%
    summarize(accuracy = mean(accuracy),
              misclassification = mean(misclassification)) %>%
    ungroup() %>%
    mutate_at(vars(-accuracy, -misclassification, -training_size),
              funs(factor(., ordered = T))) %>%
    mutate(training_size = factor(training_size, c(108, 540, 1347),
                                  c('(2p)  nlearn=108', '(10p)  nlearn=540',
                                    '(n/2)  nlearn=1347')))
)

kfold_heatmap <- (
  ggplot(data_for_heatmap) + aes(gamma, cost, fill = misclassification,
                                 group = training_size) +
    geom_tile(colour = "white") +
    scale_fill_gradient(low = "white",  high = "steelblue",
                        label = scales::percent) +
    facet_wrap(~ training_size, nrow = 3, ncol = 1) +
    ggplot2::ggtitle("Support Vector Machine Parameter Performance") +
    theme(
      panel.background = element_blank(),
      plot.background = element_blank(),
      legend.title = element_text(face = 'bold'),
      axis.title = element_text(face = 'bold'),
      axis.title.x = element_text(vjust = -2),
      axis.title.y = element_text(vjust = 2),
      axis.text = element_text(size = 6),
      plot.title = element_text(size = 20, face = 'bold'),
      plot.margin = unit(c(0.5, 0.5, 0.25, 0.5), 'inches')
    )
)

ggsave('project/figures/svm_heatmaps.png', kfold_heatmap, width = 10,
       height = 7.5, units = 'in', antialias = 'subpixel')
