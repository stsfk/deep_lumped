require(ncdf4)
require(tidyverse)
require(ncdump)
require(lubridate)
library(qgraph)
library(tripack)
library(hydroGOF)

# Data --------------------------------------------------------------------

nc_file_names <- dir("./data/conventional_lump/")

# Process -----------------------------------------------------------------

val_metrics <- vector("list", length(nc_file_names))

for (i in 1:length(nc_file_names)){
  nc_file_name <- paste0("./data/conventional_lump/",nc_file_names[[i]])
  
  metadata <- NetCDF(nc_file_name)
  data_raw <- nc_open(nc_file_name)
  
  catchment_id <- tibble(catchment_id = ncvar_get(data_raw, "Gauge_ID")) %>%
    transmute(catchment_id = str_c("000", catchment_id)) %>%
    mutate(catchment_id = str_sub(catchment_id, start = -8, end = -1))
  
  val_metric <- ncvar_get(data_raw, "Objective_function_eval") %>%
    as_tibble(.name_repair = "minimal") %>%
    setNames(c("KGE", "IKGE", "mean_KGE")) %>%
    select(KGE)
  
  val_metrics[[i]] <- catchment_id %>%
    bind_cols(val_metric) %>%
    mutate(model_name = str_sub(nc_file_name, 26, -17 ))
}

val_metrics <- val_metrics %>%
  bind_rows()

write_csv(val_metrics, file = "./data/lumped_KGEs.csv")
