if (!require("pacman")) {
  install.packages("pacman")
}

pacman::p_load(
  tidyverse,
  lubridate,
  zeallot
)


# Load data ---------------------------------------------------------------
load("/Users/yang/Documents/projects/data/Caravan/Caravan1.4/non_US_catchment.Rda")

# QC ----------------------------------------------------------------------

# all the forcing data is complete
n_complete_record <- data_process %>%
  mutate(
    missingness = purrr::map_dbl(
      data, function(x)
        x %>% select(-Q) %>% complete.cases() %>% sum()
    )
  ) %>%
  pull(missingness)


# many missing Q data, some catchment only have 365 records
n_complete_record <- data_process %>%
  mutate(
    missingness = purrr::map_dbl(
      data, function(x)
        x %>% complete.cases() %>% sum()
    )
  ) %>%
  pull(missingness)

n_complete_record %>% range()

# all the catchment's records are of the same length = 26662
data_process %>%
  mutate(
    record_length = purrr::map_dbl(
      data, function(x)
        nrow(x)
    )
  ) %>%
  pull(record_length) %>%
  unique()

# the start date is different, some from 1951-01-01, some from 1951-01-02
data_process %>%
  mutate(
    start_date = purrr::map(
      data, function(x)
        x$Date[[1]]
    )
  ) %>%
  unnest(start_date) %>%
  pull(start_date) %>%
  table()

# the end date is different, some 2023-12-30, some from 2023-12-31
data_process %>%
  mutate(
    start_date = purrr::map(
      data, function(x)
        last(x$Date)
    )
  ) %>%
  unnest(start_date) %>%
  pull(start_date) %>%
  table()


# check the available data for each day
# many missing data before 1980 and after around 2015
data_process %>%
  unnest(data) %>%
  group_by(Date)%>%
  summarise(Q_availablity=sum(!is.na(Q))/n()) %>%
  ggplot(aes(Date, Q_availablity))+
  geom_line()



# Subsetting --------------------------------------------------------------

gc()

data_process <- data_process %>%
  unnest(data)

# use record from 1981-01-01 to 2020-12-31 for the modeling study
# the data from 1980-01-02 is for warm-up
data_process <- data_process %>%
  filter(Date >= ymd("1980-01-02"),
         Date <= ymd("2020-12-31"))

# training and validation are from 1981-01-01 to 2010-12-31, where data until 2000-12-31 are for training
# testing from 2011-01-01 to 2020-12-31

# all the forcing data is available, some of the Q data is missing
# catchments with missing Q records is stored in `incomplete_catchments`

minimal_required_Q_length = 365*2 # at least 2 years of data should be avaiable in each period

incomplete_catchment_train <- data_process %>%
  filter(Date <= ymd("2000-12-31"),
         Date >= ymd("1981-01-01")) %>%
  group_by(catchment_name) %>%
  summarise(data = list(tibble(Q))) %>%
  mutate(
    n_complete_record = map_dbl(
      data, function(x) complete.cases(x) %>% sum()
    )
  ) %>%
  filter(n_complete_record < minimal_required_Q_length) %>%
  pull(catchment_name)

incomplete_catchment_val <- data_process %>%
  filter(Date >= ymd("2001-01-01"),
         Date <= ymd("2010-12-31")) %>%
  group_by(catchment_name) %>%
  summarise(data = list(tibble(Q))) %>%
  mutate(
    n_complete_record = map_dbl(
      data, function(x) complete.cases(x) %>% sum()
    )
  ) %>%
  filter(n_complete_record < minimal_required_Q_length) %>%
  pull(catchment_name)

incomplete_catchment_test <- data_process %>%
  filter(Date >= ymd("2011-01-01"),
         Date <= ymd("2020-12-31")) %>%
  group_by(catchment_name) %>%
  summarise(data = list(tibble(Q))) %>%
  mutate(
    n_complete_record = map_dbl(
      data, function(x) complete.cases(x) %>% sum()
    )
  ) %>%
  filter(n_complete_record < minimal_required_Q_length) %>%
  pull(catchment_name)

incomplete_catchments <-
  c(incomplete_catchment_train,
    incomplete_catchment_test,
    incomplete_catchment_val) %>%
  unique()

# 3681 catchments left
data_process %>%
  filter(!(catchment_name %in% incomplete_catchments)) %>% pull(catchment_name) %>% unique() %>% length()

# tibble(catchment_id = data_process %>% filter(!(catchment_id %in% incomplete_catchments)) %>% pull(catchment_id) %>% unique()) %>%
#   write_csv(file = "data/catchment_ids_Caravan.csv")

data_process <- data_process %>%
  filter(!(catchment_name %in% incomplete_catchments))

# Split the data ----------------------------------------------------------

# training and validation are from 1981-01-01 to 2010-12-31
# forcing from 1980-01-02 is used for warm-up
data_train_val <- data_process %>% 
  filter(Date <= ymd("2010-12-31"),
         Date >= ymd("1980-01-02"))

data_train_val %>% count(catchment_name) %>% pull(n) %>% unique() # length = 11322

# Q until 2000-12-31 is used for training
data_train <- data_process %>% 
  filter(Date <= ymd("2000-12-31"),
         Date >= ymd("1980-01-02"))

data_train %>% count(catchment_name) %>% pull(n) %>% unique() # length = 7670

# Q from 2001-01-01 to 2010-12-31 is used for validation, forcing from 2000-01-02 is used for warm-up
data_val <- data_process %>% 
  filter(Date >= ymd("2000-01-02"), 
         Date <= ymd("2010-12-31"))

data_val %>% count(catchment_name) %>% pull(n) %>% unique() # length = 4017

# Q from 2011-01-01 is used for testing, forcing from 2010-01-01 is used for warm-up
data_test <- data_process %>% 
  filter(Date >= ymd("2010-01-01"),
         Date <= ymd("2020-12-31"))

data_test %>% count(catchment_name) %>% pull(n) %>% unique() # length = 4018

# All the data used in modeling
data_all <- data_process %>% 
  filter(Date >= ymd("1980-01-02"),
         Date <= ymd("2020-12-31"))

data_all %>% count(catchment_name) %>% pull(n) %>% unique() # length = 14975

# date range
data_train_val$Date %>% range() # from "1980-01-02" to "2010-12-31", with the first year for warm-up only
data_train$Date %>% range() # from "1980-01-02" to "2000-12-31", with the first year for warm-up only
data_val$Date %>% range() # from "2000-01-02" to "2010-12-31", with the first year for warm-up only
data_test$Date %>% range() # from "2010-01-01" to "2020-12-31", with the first year for warm-up only
data_all$Date %>% range() # from "1980-01-02" to "2020-12-31", with the first year for warm-up only


# save data ---------------------------------------------------------------
data_train_val %>%
  arrange(catchment_name, Date) %>%
  select(P:Q) %>%
  write_csv(file = "./Caravan/Caravan1.4/data_train_val_CARAVAN.csv")

data_train %>%
  arrange(catchment_name, Date) %>%
  select(P:Q) %>%
  write_csv(file = "./Caravan/Caravan1.4/data_train_CARAVAN.csv")

data_val %>%
  arrange(catchment_name, Date) %>%
  select(P:Q) %>%
  write_csv(file = "./Caravan/Caravan1.4/data_val_CARAVAN.csv")

data_test %>%
  arrange(catchment_name, Date) %>%
  select(P:Q) %>% 
  write_csv(file = "./Caravan/Caravan1.4/data_test_CARAVAN.csv")

data_all %>%
  arrange(catchment_name, Date) %>%
  select(P:Q) %>% 
  write_csv(file = "./Caravan/Caravan1.4/data_all_CARAVAN.csv")

