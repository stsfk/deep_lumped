if (!require("pacman")) {
  install.packages("pacman")
}

pacman::p_load(
  tidyverse,
  lubridate,
  zeallot,
  sf
)


# read data ---------------------------------------------------------------

# read catchment names
file_folder <- "/Users/yang/Documents/projects/data/CAMELS_CH/Caravan_extension_CH/timeseries/csv/camelsch"
all_catchments <- dir(file_folder) %>%
  str_sub(1, -5)

data_ts <- tibble(
  catchment_id = all_catchments,
  file_name = paste0(all_catchments, ".csv"),
  # read only catchment not in the us
  path = paste0(file_folder,
                '/',
                file_name),
  data = vector("list", 1)
) 

data_ts <- data_ts %>%
  mutate(data = map(path, read_csv, show_col_types = FALSE))

data_ts <- data_ts %>%
  mutate(data = purrr::map(data, function(x)
    x %>% dplyr::select(
      date,
      P = total_precipitation_sum,
      T = temperature_2m_mean,
      PET = potential_evaporation_sum,
      Q = streamflow
    )))

data_ts <- data_ts %>%
  select(catchment_id, data)

save(data_ts, file = "/Users/yang/Documents/projects/data/CAMELS_CH/Caravan_extension_CH/forcing_runoff.Rda")


# process data ------------------------------------------------------------
load("/Users/yang/Documents/projects/data/CAMELS_CH/Caravan_extension_CH/forcing_runoff.Rda")

# process
data_process <- data_ts

# many missing data, some catchment have 0 records, the forcing data is not missing
n_complete_record <- data_process %>%
  mutate(
    missingness = purrr::map_dbl(
      data, function(x)
        x %>% complete.cases() %>% sum()
    )
  ) %>%
  pull(missingness)

# all the catchment's records are of the same length = 14610
data_process %>%
  mutate(
    record_length = purrr::map_dbl(
      data, function(x)
        nrow(x)
    )
  ) %>%
  pull(record_length) %>%
  unique()

# the start date is 1981-01-01
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

# the end date is 2020-12-31
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


# Subsetting --------------------------------------------------------------

rm(data_ts)
gc()

# catchments with no intersections with CARAVAN
selected_catchments <-
  read_csv(
    "/Users/yang/Documents/projects/indexing_catchment_model/data/selected_catchments_camelsch.csv"
  ) %>%
  select(catchment_id = gauge_id)


data_process <- data_process %>%
  filter(catchment_id %in% selected_catchments$catchment_id) %>%
  unnest(data)

# training and validation from 1981-01-02 to 2010-12-31, where data until 2000-12-31 is for training
# testing from 2011-01-01 to 2020-12-31

# all the forcing data is available, some of the flow data is missing
# catchments with missing Q records is stored in `incomplete_catchments`

minimal_required_Q_length = 365*2

incomplete_catchment_train <- data_process %>%
  filter(date < ymd("2001-01-01")) %>%
  group_by(catchment_id) %>%
  summarise(data = list(tibble(Q))) %>%
  mutate(
    n_complete_record = map_dbl(
      data, function(x) complete.cases(x) %>% sum()
    )
  ) %>%
  filter(n_complete_record < minimal_required_Q_length) %>%
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
  filter(n_complete_record < minimal_required_Q_length) %>%
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
  filter(n_complete_record < minimal_required_Q_length) %>%
  pull(catchment_id)

incomplete_catchment_train_val <- data_process %>%
  filter(date < ymd("2011-01-01")) %>%
  group_by(catchment_id) %>%
  summarise(data = list(tibble(Q))) %>%
  mutate(
    n_complete_record = map_dbl(
      data, function(x) complete.cases(x) %>% sum()
    )
  ) %>%
  filter(n_complete_record < minimal_required_Q_length) %>%
  pull(catchment_id)

incomplete_catchments <-
  c(incomplete_catchment_train_val,
    incomplete_catchment_test) %>%
  unique()

# 157 catchments left
data_process %>%
  filter(!(catchment_id %in% incomplete_catchments)) %>% pull(catchment_id) %>% unique() %>% length()

data_process <- data_process %>%
  filter(!(catchment_id %in% incomplete_catchments))

# Split the data ----------------------------------------------------------

# data until 2011-01-01 is used for training and validation
data_train_val <- data_process %>% 
  filter(date < ymd("2011-01-01"))

data_train_val %>% count(catchment_id) %>% pull(n) %>% unique() # length = 10957

# Q until 2001-01-01 is used for training
data_train <- data_process %>% 
  filter(date < ymd("2001-01-01"))

data_train %>% count(catchment_id) %>% pull(n) %>% unique() # length = 7305

# Q from 2001-01-01 to 2010-12-31 is used for validation, forcing from 2000-01-02 is used
data_val <- data_process %>% 
  filter(date > ymd("2000-01-01"), date < ymd("2011-01-01"))

data_val %>% count(catchment_id) %>% pull(n) %>% unique() # length = 4017

# Q from 2011-01-01 is used for testing, forcing from 2010-01-01 is used
data_test <- data_process %>% 
  filter(date > ymd("2009-12-31"))

data_test %>% count(catchment_id) %>% pull(n) %>% unique() # length = 4018

# All the data
data_process %>% count(catchment_id) %>% pull(n) %>% unique() # length = 14610

# data range

data_train_val$date %>% range() # from "1981-01-01" to "2010-12-31", with the first year for warm-up only
data_train$date %>% range() # from "1981-01-01" to "2000-12-31", with the first year for warm-up only
data_val$date %>% range() # from "2000-01-02" to "2010-12-31", with the first year for warm-up only
data_test$date %>% range() # from "2010-01-01" to "2020-12-31", with the first year for warm-up only

# save data ---------------------------------------------------------------

data_train_val %>%
  arrange(catchment_id, date) %>%
  select(P:Q) %>%
  write_csv(file = "./data/data_train_val_CARAVAN_CH.csv")

data_train %>%
  arrange(catchment_id, date) %>%
  select(P:Q) %>%
  write_csv(file = "./data/data_train_CARAVAN_CH.csv")

data_val %>%
  arrange(catchment_id, date) %>%
  select(P:Q) %>%
  write_csv(file = "./data/data_val_CARAVAN_CH.csv")

data_test %>%
  arrange(catchment_id, date) %>%
  select(P:Q) %>% 
  write_csv(file = "./data/data_test_CARAVAN_CH.csv")

data_process %>%
  arrange(catchment_id, date) %>%
  select(P:Q) %>% 
  write_csv(file = "./data/data_all_CARAVAN_CH.csv")






