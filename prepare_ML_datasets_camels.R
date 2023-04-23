if (!require("pacman")) {
  install.packages("pacman")
}

pacman::p_load(
  tidyverse,
  lubridate,
  zeallot
)

# Processing basin attributes ---------------------------------------------

load("./data/CAMELS_US/camels_original_forcing.Rda")

# Processing --------------------------------------------------------------

collection_names <- all_data %>% names()

for (i in 1:length(collection_names)){
  
  collection_name <- collection_names[[i]]
  
  data_process <- all_data[[collection_name]]
  
  # change missing Q (marked by negative values) to NA
  data_process <- data_process %>% 
    mutate(Q = replace(Q, Q<0, NA_real_))
  
  # Record length of each catchment can be different,
  # create a catchment-date template and left_join data
  catchment_ids <- data_process$catchment_id %>% unique()
  
  data_process <- tibble(
    catchment_id = rep(catchment_ids, each=as.numeric(ymd("2010-09-30") - ymd("1980-10-01"))+1),
    date = rep(seq(from=ymd("1980-10-01"), to = ymd("2010-09-30"), by="1 day"), length(catchment_ids))
  ) %>%
    left_join(data_process, by = c("catchment_id", "date"))
  
  # simple scaling to make forcing variables between -1 and 1
  # data_process$P <- data_process$P/max(abs(data_process$P), na.rm = T)
  # data_process$Tmax <- data_process$Tmax/max(abs(data_process$Tmax), na.rm = T)
  # data_process$Tmin <- data_process$Tmin/max(abs(data_process$Tmin), na.rm = T)
  # data_process$Srad <- data_process$Srad/max(abs(data_process$Srad), na.rm = T)
  # data_process$Vp <- data_process$Vp/max(abs(data_process$Vp), na.rm = T)
  
  # length of record each catchment = 5478, "1980-10-01" to "1981-09-30" for warm-up; 14 year data excluding warm-up
  data_process %>%
    filter(date >= ymd("1980-10-01"), date <= ymd("1995-09-30")) %>%
    select(-catchment_id, -date) %>%
    write_csv(file = paste0("./data/671_", collection_name, "_original_camels_train_val.csv"))
  
  # length of record each catchment = 4017, "1980-10-01" to "1981-09-30" for warm-up; 10 year data excluding warm-up
  data_process %>%
    filter(date >= ymd("1980-10-01"), date <= ymd("1991-09-30")) %>%
    select(-catchment_id, -date) %>%
    write_csv(file = paste0("./data/671_", collection_name, "_original_camels_train.csv"))
  
  # record length = 1826 for each catchment, "1990-10-01" to "1991-09-30" for warm-up; 4 year data excluding warm-up
  data_process %>%
    filter(date >= ymd("1990-10-01"), date <= ymd("1995-09-30")) %>%
    select(-catchment_id, -date) %>%
    write_csv(file = paste0("./data/671_", collection_name, "_original_camels_val.csv"))
  
  
  # record length = 5844 for each catchment, "1994-10-01" to "1995-09-30" for warm-up; 15 year data excluding warm-up
  data_process %>%
    filter(date >= ymd("1994-10-01"), date <= ymd("2010-09-30")) %>%
    select(-catchment_id, -date) %>%
    write_csv(file = paste0("./data/671_", collection_name, "_original_camels_test.csv"))
}



# Different splits --------------------------------------------------------


# filter catchments
catchment_id_531 <- read_csv("./data/531_basin_list.txt", col_names = "catchment_id")

data_process <- data_process %>%
  filter(catchment_id %in% catchment_id_531$catchment_id)

# record length = 2557 for each catchment, "1980-10-01" to "1981-09-30" for warm-up; 6 year data excluding warm-up
data_process %>%
  filter(date >= ymd("1998-10-01"), date <= ymd("2005-09-30")) %>%
  select(-catchment_id, -date) %>%
  write_csv(file = paste0("./data/531_", collection_name, "_original_camels_train.csv"))


# record length = 1461 for each catchment, "2004-10-01" to "2005-09-30" for warm-up; 3 year data excluding warm-up
data_process %>%
  filter(date >= ymd("2004-10-01"), date <= ymd("2008-09-30")) %>%
  select(-catchment_id, -date) %>%
  write_csv(file = paste0("./data/531_", collection_name, "_original_camels_val.csv"))


# record length = 4017 for each catchment, "1988-10-01" to "1999-09-30" for warm-up; 10 year data excluding warm-up
data_process %>%
  filter(date >= ymd("1988-10-01"), date <= ymd("1999-09-30")) %>%
  select(-catchment_id, -date) %>%
  write_csv(file = paste0("./data/531_", collection_name, "_original_camels_test.csv"))
