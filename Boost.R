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
library(bonsai)
library(lightgbm)

test_data <- read_csv("test.csv")
train_data <- read_csv("train.csv")

ghostRecipe <- recipe(type ~ ., data = train_data) %>%
  step_mutate_at(c('color'), fn=factor) 

boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
set_engine("lightgbm") %>% 
  set_mode("classification")

boost_wf <- workflow() %>%
  add_recipe(ghostRecipe) %>%
  add_model(boost_model)

tuning_grid <- grid_regular(tree_depth(), trees(),learn_rate(),levels = 5)

folds <- vfold_cv(train_data, v = 5, repeats=1)

CV_results <- boost_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid, 
            metrics = metric_set(accuracy)) 

bestTune <- CV_results %>%
  select_best(metric="accuracy")

final_boost_wf <- boost_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)

boost_predictions <- final_boost_wf %>%
  predict(new_data = test_data, type = "class")

submission <- boost_predictions %>%
  bind_cols(., test_data) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

vroom_write(x=submission, file="./Boost.csv", delim=",")

bart_model <- bart(trees=tune()) %>% 
  set_engine("dbarts") %>% 
  set_mode("classification")

bart_wf <- workflow() %>%
  add_recipe(ghostRecipe) %>%
  add_model(boost_model)

tuning_grid <- grid_regular(tree_depth(), trees(),learn_rate(), levels = 5)

folds <- vfold_cv(train_data, v = 5, repeats=1)

CV_results <- bart_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid, 
            metrics = metric_set(accuracy)) 

bestTune <- CV_results %>%
  select_best(metric="accuracy")

final_bart_wf <- bart_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)

bart_predictions <- final_bart_wf %>%
  predict(new_data = test_data, type = "class")

submission <- bart_predictions %>%
  bind_cols(., test_data) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

vroom_write(x=submission, file="./Bart.csv", delim=",")
