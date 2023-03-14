if (!require("pacman")) {
  install.packages("pacman")
}

pacman::p_load(
  tidyverse,
  lubridate,
  zeallot,
  sf,
  maps,
  ggthemes
)


# data --------------------------------------------------------------------
selected_catchment <- read_csv("./data/selected_catchments.csv") # selected catchments used in modeling, 536 catchment
lumped <- read_csv("./data/results/lumped_KGEs.csv") %>% 
  filter(catchment_id %in% selected_catchment$catchment_id) # 559 catchments

ga_method <- read_csv("./data/ga_KGEs.csv", col_names = "KGE") %>% bind_cols(selected_catchment) %>% mutate(model_name = "ga")
finetune_method <- read_csv("./data/ft_KGEs.csv", col_names = "KGE") %>% bind_cols(selected_catchment) %>% mutate(model_name = "finetune")

catchment_points <- st_read("./data/CAMELS_US/maps/physio_shp/physio.shp")

camels_topo <-
  read_delim(
    "./data/camels_topo.txt",
    delim = ";"
  ) %>%
  select(gauge_id, gauge_lat, gauge_lon)

usa <- st_as_sf(maps::map("state", fill=TRUE, plot =FALSE))

# Process -----------------------------------------------------------------

data_process <- lumped %>%
  bind_rows(ga_method) %>%
  bind_rows(finetune_method)

# which model perform the best at each location
data_plot <- data_process %>%
  group_by(catchment_id) %>%
  mutate(rank = rank(-KGE, na.last = T)) %>%
  ungroup() %>%
  filter(rank == 1) %>%
  mutate(model_name = fct_lump(model_name, n = 5)) %>%
  left_join(camels_topo, by = c("catchment_id" = "gauge_id")) %>%
  st_as_sf(coords = c("gauge_lon","gauge_lat"), remove = T)

st_crs(data_plot) <-  st_crs(st_crs(usa))

ggplot(usa) +
  geom_sf(color = "#2b2b2b", fill = "white", size=0.125) +
  geom_sf(data = data_plot, aes(color = model_name, shape=model_name), size  = 0.8) +
  coord_sf(crs = st_crs("+proj=laea +lat_0=45 +lon_0=-100 +x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs"), datum = NA) +
  labs(color="Model class", shape="Model class",title = "Which model class is the best?", subtitle = "Created using data from Knoben et al. (2020)")+
  scale_color_manual(values = c("#66c2a5","#fc8d62","#8da0cb","#e78ac3","#a6d854","#ffd92f"))+
  ggthemes::theme_map(base_size = 10)+
  theme(legend.position = "bottom")
  
ggsave(filename = "data/fig_best_model_of_catchment.pdf", height = 12, width = 16, units = "cm")


# Rank of the GA model ----------------------------------------------------
data_plot <- data_process %>%
  group_by(catchment_id) %>%
  mutate(rank = rank(-KGE, na.last = T)) %>%
  ungroup() %>%
  filter(model_name == 'ga')

ggplot(data_plot, aes(rank))+
  geom_histogram()

# Rank of the finetune model ------------------------------------------------
data_plot <- data_process %>%
  group_by(catchment_id) %>%
  mutate(rank = rank(-KGE, na.last = T)) %>%
  ungroup() %>%
  filter(model_name == 'finetune')

ggplot(data_plot, aes(rank))+
  geom_histogram()


# Map of finetune ranking -------------------------------------------------

data_process <- lumped %>%
  bind_rows(ga_method) %>%
  bind_rows(finetune_method)

data_plot <- data_process %>%
  group_by(catchment_id) %>%
  mutate(rank = rank(-KGE, na.last = T)) %>%
  ungroup() %>%
  filter(model_name == 'finetune') %>%
  left_join(camels_topo, by = c("catchment_id" = "gauge_id")) %>%
  st_as_sf(coords = c("gauge_lon","gauge_lat"), remove = T)

st_crs(data_plot) <-  st_crs(st_crs(usa))

ggplot(usa) +
  geom_sf(color = "#2b2b2b", fill = "white", size=0.125) +
  geom_sf(data = data_plot, aes(color = rank)) +
  scale_colour_gradientn(colours = terrain.colors(10))+
  coord_sf(crs = st_crs("+proj=laea +lat_0=45 +lon_0=-100 +x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs"), datum = NA) +
  labs(color="Rank of fine tune models")+
  ggthemes::theme_map()


# Map of ga ranking -------------------------------------------------

data_process <- lumped %>%
  bind_rows(ga_method) %>%
  bind_rows(finetune_method)

data_plot <- data_process %>%
  group_by(catchment_id) %>%
  mutate(rank = rank(-KGE, na.last = T)) %>%
  ungroup() %>%
  filter(model_name == 'ga') %>%
  left_join(camels_topo, by = c("catchment_id" = "gauge_id")) %>%
  st_as_sf(coords = c("gauge_lon","gauge_lat"), remove = T)

st_crs(data_plot) <-  st_crs(st_crs(usa))

ggplot(usa) +
  geom_sf(color = "#2b2b2b", fill = "white", size=0.125) +
  geom_sf(data = data_plot, aes(color = rank)) +
  scale_colour_gradientn(colours = terrain.colors(10))+
  coord_sf(crs = st_crs("+proj=laea +lat_0=45 +lon_0=-100 +x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs"), datum = NA) +
  labs(color="Rank of GA models")+
  ggthemes::theme_map()

# Map of finetune KGE -------------------------------------------------

data_process <- lumped %>%
  bind_rows(ga_method) %>%
  bind_rows(finetune_method)

data_plot <- data_process %>%
  group_by(catchment_id) %>%
  mutate(rank = rank(-KGE, na.last = T)) %>%
  ungroup() %>%
  filter(model_name == 'finetune') %>%
  left_join(camels_topo, by = c("catchment_id" = "gauge_id")) %>%
  st_as_sf(coords = c("gauge_lon","gauge_lat"), remove = T)

st_crs(data_plot) <-  st_crs(st_crs(usa))

ggplot(usa) +
  geom_sf(color = "#2b2b2b", fill = "white", size=0.125) +
  geom_sf(data = data_plot, aes(color = KGE)) +
  scale_colour_gradientn(colours = terrain.colors(10))+
  coord_sf(crs = st_crs("+proj=laea +lat_0=45 +lon_0=-100 +x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs"), datum = NA) +
  labs(color="KGE of fine tune models")+
  ggthemes::theme_map()


# Map of ga KGE -------------------------------------------------

data_process <- lumped %>%
  bind_rows(ga_method) %>%
  bind_rows(finetune_method)

data_plot <- data_process %>%
  group_by(catchment_id) %>%
  mutate(rank = rank(-KGE, na.last = T)) %>%
  ungroup() %>%
  filter(model_name == 'ga') %>%
  left_join(camels_topo, by = c("catchment_id" = "gauge_id")) %>%
  st_as_sf(coords = c("gauge_lon","gauge_lat"), remove = T)

st_crs(data_plot) <-  st_crs(st_crs(usa))

ggplot(usa) +
  geom_sf(color = "#2b2b2b", fill = "white", size=0.125) +
  geom_sf(data = data_plot, aes(color = KGE)) +
  scale_colour_gradientn(colours = terrain.colors(10))+
  coord_sf(crs = st_crs("+proj=laea +lat_0=45 +lon_0=-100 +x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs"), datum = NA) +
  labs(color="KGE of fine tune models")+
  ggthemes::theme_map()




# recycle -----------------------------------------------------------------





data_process <- lumped %>%
  bind_rows(ga_method) %>%
  bind_rows(finetune_method)

data_deep <- data_process %>%
  group_by(catchment_id) %>%
  mutate(rank = rank(-KGE)) %>%
  filter(model_name == "deep") 

data_deep %>%
  ggplot(aes(rank))+
  geom_histogram(bins = 37)


# Plot --------------------------------------------------------------------

usa <- st_as_sf(maps::map("state", fill=TRUE, plot =FALSE))

data_plot <- deep_lumped %>%
  left_join(camels_topo, by = c("catchment_id" = "gauge_id")) %>%
  st_as_sf(coords = c("gauge_lon","gauge_lat"), remove = T)

st_crs(data_plot) <-  st_crs(st_crs(usa))

ggplot(usa) +
  geom_sf(color = "#2b2b2b", fill = "white", size=0.125) +
  geom_sf(data = data_plot %>% filter(KGE > 0), aes(color = KGE)) +
  scale_colour_gradientn(colours = terrain.colors(10))+
  coord_sf(crs = st_crs("+proj=laea +lat_0=45 +lon_0=-100 +x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs"), datum = NA) +
  ggthemes::theme_map()



