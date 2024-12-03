library(tidyverse)
library(tidymodels)
library(vroom)
library(discrim)
library(dplyr)
library(recipes)
library(modeldata)
library(themis)

test_data <- read_csv("test.csv")
train_data <- read_csv("train.csv")

ghostRecipe <- recipe(type ~ ., data = train_data) %>%
  step_mutate_at(c('color'), fn=factor) %>%
  step_other(all_nominal_predictors(), threshold = .01) %>% 
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) %>%
  step_smote(all_outcomes(),neighbors = 20)
  


nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
            set_mode("classification") %>%
            set_engine("naivebayes") 

nb_wf <- workflow() %>%
        add_recipe(ghostRecipe) %>%
        add_model(nb_model)

tuning_grid <- grid_regular(Laplace(), smoothness(),levels = 5)

folds <- vfold_cv(train_data, v = 5, repeats=1)

CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid, 
            metrics = metric_set(accuracy)) 

bestTune <- CV_results %>%
  select_best(metric="accuracy")

final_nb_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)

nb_predictions <- final_nb_wf %>%
  predict(new_data = test_data, type = "class")

submission <- nb_predictions %>%
  bind_cols(., test_data) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

vroom_write(x=submission, file="./NaiveBayes.csv", delim=",")

