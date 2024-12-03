library(tidyverse)
library(tidymodels)
library(vroom)
library(kernlab)

test_data <- read_csv("test.csv")
train_data <- read_csv("train.csv")

ghostRecipe <- recipe(type ~ ., data = train_data) %>%
  step_mutate_at(c('color'), fn=factor)

svm_model <- svm_rbf(mode="classification", 
                     cost=tune(),
                     rbf_sigma=tune())

svm_workflow <- workflow() %>%
  add_model(svm_model) %>%
  add_recipe(ghostRecipe)

svm_grid <- grid_regular(
  cost(range = c(-5, 5)),   
  rbf_sigma(range = c(-5, 5)),
  levels = 5                      
)

folds <- vfold_cv(train_data, v = 5, repeats=1)

svm_results <- svm_workflow %>%
  tune_grid(
    resamples = folds,
    grid = svm_grid,
    metrics = metric_set(accuracy)
  )

bestTune <- svm_results %>%
  select_best(metric="accuracy")

final_svm_wf <- svm_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)

svm_predictions <- final_svm_wf %>%
  predict(new_data = test_data, type = "class")


submission <- svm_predictions %>%
  bind_cols(., test_data) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

vroom_write(x=submission, file="./SVM.csv", delim=",")
