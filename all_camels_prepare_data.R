# This script is to process the data used in Knoben 2020, https://doi.org/10.1029/2019WR025975Received

if (!require("pacman")) {
  install.packages("pacman")
}

pacman::p_load(
  tidyverse,
  lubridate,
  zeallot
)

# data --------------------------------------------------------------------

filenames <- dir("./data/CAMELS_Knoben/processed/")

date <- read_csv("./data/CAMELS_Knoben/date.csv", col_names = "date") # the raw data only have a datenum column

# function ----------------------------------------------------------------

data_raw <- vector("list", length(filenames))

for (i in 1:length(filenames)){
  data_raw[[i]] <- read_csv(
    paste0("./data/CAMELS_Knoben/processed/", filenames[[i]]),
    col_names = c("date", "P", "T", "PET", "Q")
  ) %>%
    select(-date) %>%
    bind_cols(date)  %>%
    mutate(catchment_id = str_extract(filenames[[i]], "[0-9]+"))
}

data_raw <- data_raw %>% 
  bind_rows() %>%
  select(catchment_id, date, everything())

# process data ------------------------------------------------------------

# change missing Q (marked by negative values) to NA
data_process <- data_raw %>% 
  mutate(Q = replace(Q, Q<0, NA_real_))

# split and write csv files, note there is a one-year warm-up period for each subset, 
# record length = 3652 for each catchment, 9 years data excluding warm up
data_process %>%
  filter(date <= ymd("1998-12-31")) %>%
  select(-catchment_id, -date) %>%
  write_csv(file = "./data/all_camels_train_val.csv")

# record length = 2922 for each catchment, 7 years data excluding warm up
data_process %>%
  filter(date <= ymd("1996-12-31")) %>%
  select(-catchment_id, -date) %>%
  write_csv(file = "./data/all_camels_train.csv")

# record length = 1095 for each catchment, 2 years data excluding warm up
data_process %>%
  filter(date <= ymd("1998-12-31"), date >= ymd("1996-01-02")) %>%
  select(-catchment_id, -date) %>%
  write_csv(file = "./data/all_camels_val.csv")

# record length = 4383 for each catchment, 11 years data excluding warm up
data_process %>%
  filter(date >= ymd("1998-01-01")) %>%
  select(-catchment_id, -date) %>%
  write_csv(file = "./data/all_camels_test.csv")
