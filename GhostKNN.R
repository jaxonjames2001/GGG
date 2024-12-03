library(tidyverse)
library(tidymodels)
library(vroom)

test_data <- read_csv("test.csv")
train_data <- read_csv("train.csv")

ghostRecipe <- recipe(type ~ ., data = train_data) %>%
  step_mutate_at(c('color'), fn=factor)

knn_model <- nearest_neighbor(neighbors=tune(), dist_power=tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn") 

knn_workflow <- workflow() %>%
  add_recipe(ghostRecipe) %>%
  add_model(knn_model)

knn_tuning_grid <- grid_regular(neighbors(),
                                dist_power(),
                                levels = 5)

folds <- vfold_cv(train_data, v = 5, repeats=1)

CV_results <- knn_workflow %>%
  tune_grid(resamples=folds,
            grid=knn_tuning_grid, 
            metrics = metric_set(accuracy)) 

bestTune <- CV_results %>%
  select_best(metric="accuracy")

final_knn_wf <- knn_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)

knn_predictions <- final_knn_wf %>%
  predict(new_data = test_data, type = "class")


submission <- knn_predictions %>%
  bind_cols(., test_data) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

vroom_write(x=submission, file="./KNN.csv", delim=",")

