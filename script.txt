library(tidyverse)
library(tidytext)
library(tm)
library(dtplyr)
library(textstem)

train_data <- read_csv(file = "data/data_complaints_train.csv")

train_data %>% 
  glimpse()

testing_data <- read_csv(file = "data/data_complaints_test.csv")

testing_data %>% glimpse()

train_data %>% 
  slice(1) %>% 
  pull(var = `Consumer complaint narrative`)


## adding IDs numbers ------------

train_data <- train_data %>% 
  mutate(
    id_number = row_number()
  )


data_token <- train_data %>% 
  select(Product, id_number, `Consumer complaint narrative`) %>% 
  unnest_tokens(output = word, input = `Consumer complaint narrative`) %>% 
  select(Product, id_number, word) %>%
  filter(
    !str_detect(word, "xx"),        # no successive x
    !str_detect(word, "[0-9]"),  # not just numbers
    nchar(word) <= 20, # max length 20
  ) %>% 
  mutate(
    word = lemmatize_words(word),
    is_english = hunspell_check(word, dict = hunspell::dictionary("en_US"))
  ) 

data_token %>% 
  filter(!is_english) %>% pull(word) %>% unique()


data_token <- data_token %>% 
  filter(is_english)
  



data_token %>% write_rds(file = "data/data_token.rds")

## visualization to see most frequent woords ------

data_token %>%
  count(word, sort = TRUE) %>%
  filter(n > 100000) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(n, word)) +
  geom_col() +
  labs(y = NULL)


## removing most stop words ---------

data("stop_words")
stop_words

data_token <- data_token %>% 
  anti_join(stop_words)

data_token %>%
  count(word, sort = TRUE) %>%
  filter(n > 50000) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(n, word)) +
  geom_col() +
  labs(y = NULL)


chosen_features <- data_token %>% 
  count(word, sort = TRUE) %>% pull(word) %>% .[1:20]


chosen_features

## percent of each word by complaint ------

word_by_compl <- data_token %>% 
  group_by(id_number, word) %>%
 summarise(
   w_in_compl = n()
 ) %>% 
  ungroup() %>% 
  left_join (
    
    data_token %>% 
      group_by(id_number) %>% 
      summarise(
        all_in_complaint = n()
      )
    
  )



#data_dtm <- word_by_compl %>% cast_dtm(document = id_number, term = word, value = w_in_compl)

tf_idf_data <- data_token %>% 
  count(word, id_number, sort=TRUE) %>% 
  bind_tf_idf(word, id_number, n) %>% 
  arrange(desc(tf_idf))


tf_idf_data %>% write_rds(file = "data/tf_idf_data.rds")
tf_idf_data <- read_rds(file = "data/tf_idf_data.rds")


## top words ----------


# top_words <- tf_idf_data %>% 
#   filter(str_length(word) <= 15) %>% 
#   pull(var = word) %>% unique() %>% 
#   str_subset("^[^\\d]" ) %>% 
#   .[1:100]
  

tf_idf_data_long <- tf_idf_data %>% 
  filter(word %in% chosen_features) %>% 
  select(word, id_number, tf_idf) %>%
  pivot_wider(id_cols = id_number, names_from = word, values_from = tf_idf, values_fill = 0.0) 

tf_idf_data_long %>% write_rds(file = "data/tf_idf_data_long.rds")



## adding products 
tf_idf_data_long <- read_rds(file = "data/tf_idf_data_long.rds")

#train_data$id_number <- 1:nrow(train_data)
final_train_data <- train_data %>% 
  select(id_number, Product) %>% 
  left_join(tf_idf_data_long) %>% 
  select(!id_number)


### Training machine learning data ----------

library(tidymodels)
library(ranger)

# 1. Split the data (80% train / 20% test)
set.seed(123)
#final_train_data_2 <- final_train_data %>% mutate(across(!Product, \(x) {x * 100}))
final_train_data <- final_train_data %>% mutate(across(!Product, ~replace_na(., 0.0)))
final_train_data %>% write_rds(file = "data/final_training_data.rds")

data_split <- initial_split(final_train_data, prop = 0.8, strata = Product)
training_data <- training(data_split)
testing_data  <- testing(data_split)

# 2. Recipe (prétraitement)
rf_recipe <- recipe(Product ~ ., data = training_data) %>% 
  step_impute_mean()

# 3. Spécification du modèle Random Forest
rf_spec <- rand_forest(
  mtry = 5,       # à ajuster selon le nombre de variables explicatives
  trees = 500,
  min_n = 5
) %>%
  set_engine("ranger", importance = "impurity") %>%  # importance des variables
  set_mode("classification")

# 4. Workflow
rf_workflow <- workflow() %>%
  add_recipe(rf_recipe) %>%
  add_model(rf_spec)

# 5. Entraînement du modèle
rf_fit <- rf_workflow %>%
  fit(data = training_data)

write_rds(x = rf_fit, file = "data/random_forest_fited.rds")

# 6. Prédictions sur les données test
rf_preds <- predict(rf_fit, testing_data) %>%
  bind_cols(testing_data)


# 7. Évaluation
metrics <- rf_preds %>%
  mutate(
    Product = factor(Product),
    .pred_class = factor(.pred_class)
  ) %>% 
  metrics(truth = Product, estimate = .pred_class)

conf_mat <- rf_preds %>%
  mutate(
    Product = factor(Product),
    .pred_class = factor(.pred_class)
  ) %>% 
  conf_mat(truth = Product, estimate = .pred_class)

print(metrics)
print(conf_mat)


