library(tidyverse)
library(tidymodels)
library(vroom)
library(glmnet)
library(recipes)
library(embed)
library(lme4)
library(kknn)
library(themis)
library(ggmosaic)


test <- vroom("test.csv")
train <- vroom("train.csv")
missset <- vroom("trainWithMissingValues.csv")

recipe <- recipe(type ~ ., data = missset) %>% 
  step_impute_median(all_numeric_predictors())

baked_data <- bake(prep(recipe), new_data = missset)

rmse_vec(train[is.na(missset)],
         baked_data[is.na(missset)])
