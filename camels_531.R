# This code uses the method presented in Gnann, S. (2022, April). Sebastiangnann/camels_matlab: First release. Zenodo. 
# Rerieved from https://doi.org/10.5281/zenodo.6462821 doi: 10.5281/ zenodo.6462821

if (!require("pacman")) {
  install.packages("pacman")
}

pacman::p_load(
  tidyverse,
  lubridate,
  zeallot
)


# data --------------------------------------------------------------------

load("./data/camels_all_data_raw.Rda")

catchment_id_531 <- read_csv("./data/531_basin_list.txt", col_names = "catchment_id")


# Processing --------------------------------------------------------------

data_process <- all_data

# change missing Q (marked by negative values) to NA
data_process <- data_process %>% 
  mutate(Q = replace(Q, Q<0, NA_real_))

# Record length of each catchment is different,
# create a catchment-date template and left_join data

catchment_ids <- data_process$catchment_id %>% unique()

data_process <- tibble(
  catchment_id = rep(catchment_ids, each=as.numeric(ymd("2010-09-30") - ymd("1980-10-01"))+1),
  date = rep(seq(from=ymd("1980-10-01"), to = ymd("2010-09-30"), by="1 day"), length(catchment_ids))
) %>%
  left_join(data_process, by = c("catchment_id", "date"))

# filter catchments
data_process <- data_process %>%
  filter(catchment_id %in% catchment_id_531$catchment_id)


# length of record each catchment = 3653, "1998-10-01" to "2008-09-30" for warm-up; 9 year data excluding warm-up
data_process %>%
  filter(date >= ymd("1998-10-01"), date <= ymd("2008-09-30")) %>%
  select(-catchment_id, -date) %>%
  write_csv(file = "./data/531_camels_train_val.csv")


# record length = 2557 for each catchment, "1980-10-01" to "1981-09-30" for warm-up; 6 year data excluding warm-up
data_process %>%
  filter(date >= ymd("1998-10-01"), date <= ymd("2005-09-30")) %>%
  select(-catchment_id, -date) %>%
  write_csv(file = "./data/531_camels_train.csv")


# record length = 1461 for each catchment, "2004-10-01" to "2005-09-30" for warm-up; 3 year data excluding warm-up
data_process %>%
  filter(date >= ymd("2004-10-01"), date <= ymd("2008-09-30")) %>%
  select(-catchment_id, -date) %>%
  write_csv(file = "./data/531_camels_val.csv")


# record length = 4017 for each catchment, "1988-10-01" to "1999-09-30" for warm-up; 10 year data excluding warm-up
data_process %>%
  filter(date >= ymd("1988-10-01"), date <= ymd("1999-09-30")) %>%
  select(-catchment_id, -date) %>%
  write_csv(file = "./data/531_camels_test.csv")


# Quality check -----------------------------------------------------------

# check regular interval
data_process$date %>%
  diff() %>%
  table()

# check missing data
data_process %>%
  sapply(function(x) sum(is.na(x))/length(x))

# check negative
data_process %>%
  select(P, PET) %>%
  lapply(function(x) sum(x < 0)/length(x))

