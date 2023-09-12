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

# 559 catchments used in the calibration study 
selected_catchment_id <- read_csv("./data/CAMELS_Knoben/knoben_selected_catchment.csv") %>% pull(catchment_id) %>% unique()

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

# select catchments used in the KNOBEN ET AL. 2020 study
data_process <- data_raw %>%
  filter(catchment_id %in% selected_catchment_id)

# change missing Q (marked by negative values) to NA
data_process <- data_process %>% 
  mutate(Q = replace(Q, Q<0, NA_real_))

# write data for all catchments, record length = 7670 for each catchment
data_process %>%
  select(-catchment_id, -date) %>%
  write_csv(file = "./data/camels_all.csv")

# split and write csv files, note there is a one-year warm-up period for each subset, 
# record length = 3652 for each catchment
data_process %>%
  filter(date <= ymd("1998-12-31")) %>%
  select(-catchment_id, -date) %>%
  write_csv(file = "./data/camels_train_val.csv")

# record length = 2922 for each catchment
data_process %>%
  filter(date <= ymd("1996-12-31")) %>%
  select(-catchment_id, -date) %>%
  write_csv(file = "./data/camels_train.csv")

# record length = 1095 for each catchment
data_process %>%
  filter(date <= ymd("1998-12-31"), date >= ymd("1996-01-02")) %>%
  select(-catchment_id, -date) %>%
  write_csv(file = "./data/camels_val.csv")

# record length = 4383 for each catchment
data_process %>%
  filter(date >= ymd("1998-01-01")) %>%
  select(-catchment_id, -date) %>%
  write_csv(file = "./data/camels_test.csv")
