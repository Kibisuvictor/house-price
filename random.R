library(tidymodels)
library(recipes)
library(ranger)
library(magrittr)
library(tidyverse)
library(skimr)
library(rsample)
library(parsnip)
library(yardstick)
library(tune)
library(workflows)
library(dials)

#loading the dataset
#this is the training dataset from kaggle about predicting the house prices, its similar to the ames 
#housing data set
houses_tr <- read_csv("C:\\Users\\hope\\Desktop\\house-prices-advanced-regression-techniques\\train.csv")
houses_tr %<>% set_names(., tolower(names(.)))

#changing characters to factors
houses_tr <- houses_tr %>% mutate_if(is.character, as.factor)
houses_tr$mosold <- as.factor(houses_tr$mosold)
houses_tr %>% skim(mssubclass)
houses_tr$mssubclass <- as.factor(houses_tr$mssubclass)

#feature selection
houses_tr <- houses_tr %>% select(-id,-street,-utilities,-lotconfig,-landslope,-condition2,
                                  -roofmatl,-extercond,-bsmtcond,-bsmtfintype2, -bsmtfinsf2,
                                  -heating,- lowqualfinsf,-bsmthalfbath,-garagequal,-garagecond,
                                  -`3ssnporch`,-screenporch,-poolarea,-miscval,-mosold, -yrsold,
                                  -saletype)

#split the data
houses_split <- houses_tr %>% initial_split(prop = 0.9, strata = saleprice)
ames_train <- training(houses_split)

#validation
ames_cv <- vfold_cv(ames_train, v= 5, strata = saleprice)
ames_cv

#recipe
rec_ame <- recipe(saleprice ~., data = houses_tr) %>% 
  step_log(saleprice, base = 10) %>% 
  step_knnimpute(all_predictors(),-all_nominal()) %>%
  step_unknown(all_nominal(), new_level = "none") %>% 
  step_other(all_nominal(), other = "infrequent") %>% 
  step_corr(all_predictors(), -all_nominal()) %>% 
  step_dummy(all_nominal()) %>% 
  step_normalize(all_predictors())
rec_ame %>% check_missing(all_predictors())

library()
  

# specify model
rand_model <- rand_forest(mode = "regression",
                          mtry = tune(),
                          trees = tune(),
                          min_n = tune()
                          ) %>% 
  set_engine("ranger")

#grid
rand_grid <- grid_random(mtry() %>% range_set(c(2,20)),
                         trees() %>% range_set(c(500,1000)),
                         min_n() %>% range_set(c(2,10)),
                         size = 5)
rand_grid

#workflow
rand_wkflow <- workflow() %>% 
  add_model(rand_model) %>% 
  add_recipe(rec_ame)

#tune
library(tune)
doParallel::registerDoParallel()
rand_tune <- tune_grid(rand_wkflow,
                       resamples = ames_cv,
                       grid = rand_grid,
                       metrics = metric_set(rmse, rsq),
                       control = control_grid(save_pred = TRUE)
                       )

library(reprex)
rand_tune %>% pluck(".notes")
sessioninfo::package_info()
