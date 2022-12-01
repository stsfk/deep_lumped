if (!require("pacman")) {
  install.packages("pacman")
}

pacman::p_load(
  tidyverse,
  lubridate,
  zeallot
)


# Read time series data ---------------------------------------------------

collection_names <- dir("./data/Caravan/timeseries/csv/")
ts_filename <- lapply(collection_names, function(x) paste0("./data/Caravan/timeseries/csv/", x)) %>%
  lapply(dir)

n_catchment <- unlist(ts_filename) %>% length()

data_ts <- tibble(
  file_name = unlist(ts_filename),
  collection_name = str_extract(file_name, ".*(?=[_])"),
  catchment_name = str_extract(file_name, "(?<=[_]).*(?=[\\.])"),
  data = vector("list", 1)
)

for (i in 1:nrow(data_ts)){
  data_ts$data[[i]] <-
    read_csv(
      paste0(
        "./data/Caravan/timeseries/csv/",
        data_ts$collection_name[[i]],
        '/',
        data_ts$file_name[[i]]
      ),
      show_col_types = FALSE
    )
}

data_ts <- data_ts %>%
  select(-file_name)

save(data_ts, file = "./data/Caravan/all_ts.Rda")


# Filtering meteorological forcing  ---------------------------------------

load("./data/Caravan/all_ts.Rda")

data_raw <- data_ts %>%
  mutate(data = purrr::map(data, function(x)
    x %>% dplyr::select(
      date,
      P = total_precipitation_sum,
      T = temperature_2m_mean,
      PET = potential_evaporation_sum,
      Q = streamflow
    )))

data_process <- data_raw  %>%
  filter(collection_name != "camels") %>%
  mutate(
    catchment_id = paste(collection_name, catchment_name, sep = "-")
  ) %>%
  select(catchment_id, data)


# many missing data, some catchment only have 396 records
n_complete_record <- data_process %>%
  mutate(
    missingness = purrr::map_dbl(
      data, function(x)
        x %>% complete.cases() %>% sum()
    )
  ) %>%
  pull(missingness)

# all the catchment's records are of the same length
data_process %>%
  mutate(
    record_length = purrr::map_dbl(
      data, function(x)
        nrow(x)
    )
  ) %>%
  pull(record_length) %>%
  unique()

# the start date is different, some from 1981-01-01, some from 1981-01-02
data_process %>%
  mutate(
    start_date = purrr::map(
      data, function(x)
        x$date[[1]]
    )
  ) %>%
  unnest(start_date) %>%
  pull(start_date) %>%
  table()

# the end date is different, some 2020-12-30, some from 2020-12-31
data_process %>%
  mutate(
    start_date = purrr::map(
      data, function(x)
        last(x$date)
    )
  ) %>%
  unnest(start_date) %>%
  pull(start_date) %>%
  table()


# Splitting data ----------------------------------------------------------

rm(data_ts)
gc()

data_process <- data_process %>%
  unnest(data)

# use record from 1981-01-02 to 2020-12-30, as the data has different starting and ending dates
data_process <- data_process %>%
  filter(date > ymd("1981-01-01"),
         date < ymd("2020-12-31"))

# training and validation from 1981-01-01 to 2010-12-31, where data until 2000-12-31 is for training
# testing from 2011-01-01 to 2020-12-31

# all the forcing data is available, some of the flow data is missing
# catchments with missing Q records is stored in `incomplete_catchments`

incomplete_catchment_train <- data_process %>%
  filter(date < ymd("2001-01-01")) %>%
  group_by(catchment_id) %>%
  summarise(data = list(tibble(Q))) %>%
  mutate(
    n_complete_record = map_dbl(
      data, function(x) complete.cases(x) %>% sum()
    )
  ) %>%
  filter(n_complete_record < 365) %>%
  pull(catchment_id)

incomplete_catchment_val <- data_process %>%
  filter(date > ymd("2000-12-31"),
         date < ymd("2011-01-01")) %>%
  group_by(catchment_id) %>%
  summarise(data = list(tibble(Q))) %>%
  mutate(
    n_complete_record = map_dbl(
      data, function(x) complete.cases(x) %>% sum()
    )
  ) %>%
  filter(n_complete_record < 365) %>%
  pull(catchment_id)

incomplete_catchment_test <- data_process %>%
  filter(date > ymd("2010-12-31")) %>%
  group_by(catchment_id) %>%
  summarise(data = list(tibble(Q))) %>%
  mutate(
    n_complete_record = map_dbl(
      data, function(x) complete.cases(x) %>% sum()
    )
  ) %>%
  filter(n_complete_record < 365) %>%
  pull(catchment_id)

incomplete_catchments <-
  c(incomplete_catchment_train,
    incomplete_catchment_test,
    incomplete_catchment_val) %>%
  unique()

data_process <- data_process %>%
  filter(!(catchment_id %in% incomplete_catchments))

# 1758 catchments left

# Split the data ----------------------------------------------------------

# Q until 2011-01-01 is used for training and validation
data_train_val <- data_process %>% 
  filter(date < ymd("2011-01-01"))

data_train_val %>% count(catchment_id) %>% pull(n) %>% unique() # length = 10956

# Q until 2001-01-01 is used for training
data_train <- data_process %>% 
  filter(date < ymd("2001-01-01"))

data_train %>% count(catchment_id) %>% pull(n) %>% unique() # length = 7304

# Q from 2001-01-01 to 2010-12-31 is used for validation, forcing from 2000-01-02 is used
data_val <- data_process %>% 
  filter(date > ymd("2000-01-01"), date < ymd("2011-01-01"))

data_val %>% count(catchment_id) %>% pull(n) %>% unique() # length = 4017

# Q from 2011-01-01 is used for testing, forcing from 2010-01-01 is used
data_test <- data_process %>% 
  filter(date > ymd("2009-12-31"))

data_test %>% count(catchment_id) %>% pull(n) %>% unique() # length = 4017

# All the data
data_process %>% count(catchment_id) # length = 14608

# data range

data_train_val$date %>% range() # from "1981-01-02" to "2010-12-31", with the first year for warm-up only
data_train$date %>% range() # from "1981-01-02" to "2000-12-31", with the first year for warm-up only
data_val$date %>% range() # from "2000-01-02" to "2010-12-31", with the first year for warm-up only
data_test$date %>% range() # from "2010-01-01" to "2020-12-30", with the first year for warm-up only

# save data ---------------------------------------------------------------

data_train_val <- data_train_val %>%
  group_by(catchment_id) %>%
  arrange(date) %>% 
  ungroup() %>%
  select(P:Q)

write_csv(data_train_val, file = "./data/data_train_val_w_missing.csv")

data_train <- data_train %>%
  group_by(catchment_id) %>%
  arrange(date) %>% 
  ungroup() %>%
  select(P:Q)

write_csv(data_train, file = "./data/data_train_w_missing.csv")

data_val <- data_val %>%
  group_by(catchment_id) %>%
  arrange(date) %>% 
  ungroup() %>%
  select(P:Q)

write_csv(data_val, file = "./data/data_val_w_missing.csv")


data_test <- data_test %>%
  group_by(catchment_id) %>%
  arrange(date) %>% 
  ungroup() %>%
  select(P:Q)

write_csv(data_test, file = "./data/data_test_w_missing.csv")

data_all <- data_process %>%
  group_by(catchment_id) %>%
  arrange(date) %>% 
  ungroup() %>%
  select(P:Q)

write_csv(data_all, file = "./data/data_all_w_missing.csv")
