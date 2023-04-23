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

# Processing basin attributes ---------------------------------------------

catchment_id <- read_delim(
  "./data/CAMELS_US/camels_attributes_v2.0/camels_attributes_v2.0/camels_clim.txt",
  delim = ";"
) %>%
  pull(gauge_id)

camels_topo <-
  read_delim(
    "./data/CAMELS_US/camels_attributes_v2.0/camels_attributes_v2.0/camels_topo.txt",
    delim = ";"
  ) 


# Read Q ------------------------------------------------------------------

read_streamflow <- function(fpath) {
  read_table(
    fpath,
    col_names = c(
      "catchment_id",
      "year",
      "month",
      "day",
      "discharge",
      "quality_flag"
    ),
    col_types = cols(
      catchment_id = col_character(),
      year = col_character(),
      month = col_character(),
      day = col_character(),
      discharge = col_double(),
      quality_flag = col_character()
    )
  ) %>%
    mutate(date = ymd(paste(year, month, day))) %>%
    dplyr::select(catchment_id, date, discharge, quality_flag) %>%
    arrange(date)
}

# read catchment discharge, there are 674 catchments with Q series, however only 671 included in camels 
catchment_discharge <-
  tibble(
    subfolder = dir(
      './data/CAMELS_US/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/usgs_streamflow/'
    ),
    mainfoler = "./data/CAMELS_US/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/usgs_streamflow/"
  ) %>%
  transmute(folder = paste0(mainfoler, subfolder),
            fname = map(folder, dir)) %>%
  unnest(fname) %>%
  transmute(
    catchment_id = map_chr(fname, str_extract, "[0-9]+"),
    fpath = paste0(folder, "/", fname),
    discharge = map(fpath, read_streamflow)
  ) %>%
  dplyr::select(discharge) %>%
  unnest(discharge)

# normalize by catchment area
Q <- catchment_discharge %>%
  left_join(camels_topo %>% select(catchment_id = gauge_id, area_gages2), by = c("catchment_id")) %>%
  mutate(discharge = discharge * 24 * 3600 * 0.0283168466/(area_gages2 * 1000000) * 1000) %>%# CFS to to ft3 to m3 divided by m2 to mm
  select(-area_gages2, Q = discharge)


# Forcing -----------------------------------------------------------------


read_forcing <- function(fpath) {
  read_table(
    fpath,
    skip = 4,
    col_names = c(
      "Year",
      "Mnth",
      "Day",
      "Hr",
      "Dayl(s)",
      "PRCP(mm/day)",
      "SRAD(W/m2)",
      "SWE(mm)",
      "Tmax(C)",
      "Tmin(C)",
      "Vp(Pa)"
    ),
    col_types = cols(
      Year = col_character(),
      Mnth = col_character(),
      Day = col_character(),
      Hr = col_character(),
      `Dayl(s)` = col_double(),
      `PRCP(mm/day)` = col_double(),
      `SRAD(W/m2)` = col_double(),
      `SWE(mm)` = col_double(),
      `Tmax(C)` = col_double(),
      `Tmin(C)` = col_double(),
      `Vp(Pa)` = col_double()
    )
  ) %>%
    transmute(
      date = ymd(paste(Year, Mnth, Day)),
      P = `PRCP(mm/day)`,
      Tmax = `Tmax(C)`,
      Tmin = `Tmin(C)`,
      Srad = `SRAD(W/m2)`,
      Vp = `Vp(Pa)`
    )
}

read_forcing_collection <-
  function(collection_path = './data/CAMELS_US/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/daymet/') {
    tibble(subfolder = dir(collection_path),
           mainfoler = collection_path) %>%
      transmute(
        folder = paste0(mainfoler, subfolder),
        fname = map(folder, dir, pattern = "_forcing_leap.txt")
      ) %>%
      unnest(fname) %>%
      transmute(
        catchment_id = map_chr(fname, str_extract, "[0-9]+"),
        fpath = paste0(folder, "/", fname)
      ) %>%
      dplyr::filter(catchment_id %in% camels_topo$gauge_id) %>%
      mutate(forcing = map(fpath, read_forcing)) %>%
      select(-fpath) %>%
      unnest(forcing)
  }

daymet <- read_forcing_collection('./data/CAMELS_US/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/daymet/')
maurer <- read_forcing_collection('./data/CAMELS_US/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/maurer/')
nldas <- read_forcing_collection('./data/CAMELS_US/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/nldas/')

daymet <- daymet %>%
  left_join(Q %>% select(-quality_flag), by = c("catchment_id", "date"))

maurer <- maurer %>%
  left_join(Q %>% select(-quality_flag), by = c("catchment_id", "date"))

nldas <- nldas %>%
  left_join(Q %>% select(-quality_flag), by = c("catchment_id", "date"))

all_data <-
  list(
    daymet = daymet,
    maurer = maurer,
    nldas = nldas
  )

# Save data ---------------------------------------------------------------
save(all_data, file = "./data/CAMELS_US/camels_original_forcing.Rda")

# Quality check -----------------------------------------------------------

# check regular interval
daymet$date %>%
  diff() %>%
  table()

maurer$date %>%
  diff() %>%
  table()

nldas$date %>%
  diff() %>%
  table()

# check missing data
daymet %>%
  sapply(function(x) sum(is.na(x))/length(x))

maurer %>%
  sapply(function(x) sum(is.na(x))/length(x))

nldas %>%
  sapply(function(x) sum(is.na(x))/length(x))

# check negative
daymet %>%
  select(P) %>%
  lapply(function(x) sum(x < 0)/length(x))

maurer %>%
  select(P) %>%
  lapply(function(x) sum(x < 0)/length(x))

nldas %>%
  select(P) %>%
  lapply(function(x) sum(x < 0)/length(x))
