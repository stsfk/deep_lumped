if (!require("pacman")) {
  install.packages("pacman")
}

pacman::p_load(
  tidyverse,
  lubridate,
  zeallot,
  fs
)




# Read data ---------------------------------------------------------------

timeseries_files <- dir_ls("D:/data/CAMELS DE/Version 0.1.0/timeseries/")

data <- vector("list", length(timeseries_files))

for (i in seq_along(timeseries_files)){
  timeseries_file <- timeseries_files[[i]]
  catchment_name <- basename(timeseries_file) %>%
    str_split("_", simplify = T) %>%
    .[length(.)] %>%
    str_sub(end =-5)
  
  timeseries <- read_csv(timeseries_file, show_col_types = FALSE) %>%
    select(date, Q = discharge_spec, P = precipitation_mean, T = temperature_mean)
  
  timeseries_simulated_file <- paste0("D:/data/CAMELS DE/Version 0.1.0/timeseries_simulated/CAMELS_DE_discharge_sim_", catchment_name, ".csv")
  timeseries_simulated <- read_csv(timeseries_simulated_file, show_col_types = FALSE) %>%
    select(date, PET = pet_hargreaves)
  
  data[[i]] <- tibble(
    catchment_name = catchment_name,
    data = list(timeseries %>% left_join(timeseries_simulated, by = join_by(date)) %>%
      select(date, P, T, PET, Q))
  )
}

data <- data %>% bind_rows()

save(data, file = "./data/camels_de.Rda")


# Split data --------------------------------------------------------------

gc()

data_process <- data %>%
  unnest(data) %>%
  rename(Date = date)

# use record from 1989-01-01 to 2009-12-31 for the modeling study
# the data from 1988-01-02 is for warm-up
data_process <- data_process %>%
  filter(Date >= ymd("1988-01-02"),
         Date <= ymd("2009-12-31"))

# calibration are from 1989-01-01 to 1998-12-31
# testing from 1999-01-01 to 2009-12-31

# all the forcing data is available, some of the Q data is missing
# catchments with missing Q records is stored in `incomplete_catchments`

minimal_required_Q_length = 365*2 # at least 2 years of data should be available in each period

incomplete_catchment_calibration <- data_process %>%
  filter(Date <= ymd("1998-12-31"),
         Date >= ymd("1989-01-01")) %>%
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
  filter(Date >= ymd("1999-01-01")) %>%
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
  c(incomplete_catchment_calibration,
    incomplete_catchment_test) %>%
  unique()

# 1347 catchments left
data_process %>%
  filter(!(catchment_name %in% incomplete_catchments)) %>% pull(catchment_name) %>% unique() %>% length()


# keep data of complete catchments
data_process <- data_process %>%
  filter(!(catchment_name %in% incomplete_catchments))

# Split the data ----------------------------------------------------------

# calibration are from 1989-01-01 to 1998-12-31
# forcing from 1998-01-02 is used for warm-up
data_calibration <- data_process %>%
  filter(Date >= ymd("1988-01-02"), 
         Date <= ymd("1998-12-31"))

data_calibration %>% count(catchment_name) %>% pull(n) %>% unique() # length = 4017


# Q from 1999-01-01 is used for testing, forcing from 1998-01-01 is used for warm-up
data_test <- data_process %>% 
  filter(Date >= ymd("1998-01-01"))

data_test %>% count(catchment_name) %>% pull(n) %>% unique() # length = 4383



# date range
data_calibration$Date %>% range() # from "1988-01-02" to "1998-12-31", with the first year for warm-up only
data_test$Date %>% range() # from "1998-01-01" to "2009-12-31", with the first year for warm-up only

# save data ---------------------------------------------------------------
data_calibration %>%
  arrange(catchment_name, Date) %>%
  select(P:Q) %>%
  write_csv(file = "./data/data_calibration_camels_de.csv")

data_test %>%
  arrange(catchment_name, Date) %>%
  select(P:Q) %>% 
  write_csv(file = "./data/data_test_camels_de.csv")


