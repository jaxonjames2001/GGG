library(tidyverse)
library(tidymodels)
library(vroom)

test_data <- read_csv("test.csv")
train_data <- read_csv("train.csv")

ghostRecipe <- recipe(type ~ ., data = train_data) %>%
  step_mutate_at(c('color'), fn=factor) %>%
  step_other(all_nominal_predictors(), threshold = .01) %>% 
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type))

ghost_rf <- rand_forest(mtry = tune(),
                        min_n=tune(),
                        trees=500) %>%
          set_engine("ranger") %>%
          set_mode("classification")

rf_workflow <- workflow() %>%
  add_recipe(ghostRecipe) %>%
  add_model(ghost_rf)

rf_tuning_grid <- grid_regular(mtry(range=c(1,10)),
                               min_n(),
                               levels = 5)

folds <- vfold_cv(train_data, v = 5, repeats=1)

CV_results <- rf_workflow %>%
  tune_grid(resamples=folds,
            grid=rf_tuning_grid, 
            metrics = metric_set(accuracy)) 

bestTune <- CV_results %>%
  select_best(metric="accuracy")

final_rf_wf <- rf_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)

rf_predictions <- final_rf_wf %>%
  predict(new_data = test_data, type = "class")

submission <- rf_predictions %>%
  bind_cols(., test_data) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

vroom_write(x=submission, file="./RandomForest.csv", delim=",")
