if (!require("pacman")) {
  install.packages("pacman")
}

pacman::p_load(
  tidyverse,
  lubridate,
  zeallot,
  sf,
  maps,
  ggthemes,
  cowplot,
  grDevices,
  viridis
)


# data --------------------------------------------------------------------
selected_catchment <- read_csv("./data/CAMELS_Knoben/selected_catchments.txt") %>%
  transmute(catchment_id = str_sub(paste0("00", ID), -8, -1))

lumped <- read_csv("./data/CAMELS_Knoben/lumped_KGEs.csv") # 559 catchments

deep_lumped <- read_csv("./data/out_of_sample_results/CAMELS_10/camels_KGEs.csv", col_names = "KGE") %>% bind_cols(selected_catchment) %>% mutate(model_name = "deep")

catchment_points <- st_read("./data/maps/physio_shp/physio.shp")

camels_topo <-
  read_delim(
    "./data/maps/camels_topo.txt",
    delim = ";"
  ) %>%
  select(gauge_id, gauge_lat, gauge_lon)

usa <- st_as_sf(maps::map("state", fill=TRUE, plot =FALSE))


# Read CAMELS-DE data -----------------------------------------------------

data_process2 <- read_csv("data/out_of_sample_results/CAMELS_DE_10/camels_de_KGEs.csv",
                          col_names = "KGE")


# Plots -------------------------------------------------------------------

deep_lumped$KGE %>% mean() # 0.6336853
deep_lumped$KGE %>% median() # 0.7222111

data_process2$KGE %>% median() # 0.7518076

# CDF
data_process <- lumped %>%
  mutate(type = "Conventional lumped models") %>%
  bind_rows(deep_lumped %>% mutate(type = "Generative model")) %>%
  filter(!is.na(KGE))
#  mutate(type = factor(type, levels = c("Generative", "Conventional"))) %>%

A <- ggplot()+
  stat_ecdf(data = data_process, aes(KGE, group = model_name, color = type),linewidth=0.3, alpha = 0.8) +
  scale_color_manual(values = c("grey60", "red")) +
  coord_cartesian(xlim = c(-1,1), expand = 0) +
  annotate("text", x = -0.5, y = 0.5, label = paste0("Test median = 0.722"), size = 3.5)+
  labs(color = "Model type", y = "Non-exceedance probability", title = "(a) CAMLES")+
  theme_bw(base_size = 10)+
  theme(
    legend.position = c(0.05, 0.98),
    legend.justification = c(0, 1),
    plot.margin = margin(0.35, 0.5, 0.35, 0.35, "cm"),
    legend.background = element_rect(fill = NA)
  )


B <- ggplot()+
  stat_ecdf(data = data_process2, aes(KGE), color = "red",linewidth=0.3, alpha = 0.8) +
  coord_cartesian(xlim = c(-1,1), expand = 0) +
  annotate("text", x = -0.5, y = 0.5, label = paste0("Test median = 0.752"), size = 3.5)+
  labs(y = "Non-exceedance probability", title = "(b) CAMLES-DE")+
  theme_bw(base_size = 10)+
  theme(plot.margin = margin(0.35, 0.5, 0.35, 0.35, "cm"))

C <- cowplot::plot_grid(A, B, rel_widths = c(1,1), nrow = 1)

grDevices::cairo_pdf("./data/figures/fig_generative_vs_lumped.pdf", width = 8, height = 4)
C
dev.off()


# KGEs of deep learning models vs. best lumped model
best_lumped <- lumped %>%
  group_by(catchment_id)%>%
  summarise(KGE = max(KGE, na.rm = T))

deep_lumped %>%
  left_join(best_lumped, by = "catchment_id") %>%
  ggplot(aes(KGE.y, KGE.x))+
  geom_point(
    fill = "steelblue",
    shape = 21,
    alpha = 0.5,
    size = 1.25,
    stroke = 0.5
  ) +
  geom_abline(
    aes(
      slope = 1,
      intercept = 0,
      linetype = "1:1 reference line"
    ),
    colour = 'coral2',
    linewidth = 0.6
  ) +
  scale_linetype_manual(values = c(2))+
  coord_cartesian(xlim = c(-1,1), expand = 0, ylim = c(-1,1)) +
  labs(x = "KGE of best conventional lumped model",
       y = "KGE of generative model",
       linetype="")+
  theme_bw(base_size = 10)+
  theme(legend.position = "top", legend.key.height = unit(0.25, "cm"),
        axis.text = element_text(size = 6.5),
        plot.margin = margin(0.35,0.5,0.35,0.35, "cm"))

ggsave(filename = "./data/figures/fig_best_lump_vs_generative.pdf", width = 5, height = 4, units = "in")



# Rank of the models
data_process <- lumped %>%
  bind_rows(deep_lumped)

data_plot <- data_process %>%
  group_by(catchment_id) %>%
  mutate(rank = rank(-KGE, na.last = T)) %>%
  ungroup() %>%
  filter(model_name == 'deep') %>%
  left_join(camels_topo, by = c("catchment_id" = "gauge_id")) %>%
  st_as_sf(coords = c("gauge_lon","gauge_lat"), remove = T)

st_crs(data_plot) <-  st_crs(st_crs(usa))

ggplot(data_plot)+
  geom_histogram(aes(rank), bins = 37, color = "white")+
  labs(x = "Rank of generative model among 37 types of models")+
  theme_bw(base_size = 10) # histogram

ggsave(filename = "./data/figures/fig_rank_histogram.pdf", width = 5, height = 3, units = "in")



# maps of KGE
data_process <- lumped %>%
  bind_rows(deep_lumped)

data_plot <- data_process %>%
  group_by(catchment_id) %>%
  mutate(rank = rank(-KGE, na.last = T)) %>%
  ungroup() %>%
  filter(model_name == 'deep') %>%
  left_join(camels_topo, by = c("catchment_id" = "gauge_id")) %>%
  st_as_sf(coords = c("gauge_lon","gauge_lat"), remove = T)

st_crs(data_plot) <-  st_crs(st_crs(usa))

data_plot2 <- data_plot %>%
  mutate(KGE_group = cut(KGE, breaks = c(
    -Inf, -0.414, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1
  )))

best_lumped <- lumped %>%
  group_by(catchment_id)%>%
  summarise(KGE = max(KGE, na.rm = T))

data_plot_diff <- deep_lumped %>%
  left_join(best_lumped, by = "catchment_id") %>%
  mutate(diff = KGE.x - KGE.y) %>%
  mutate(diff_group = cut(diff, breaks = c(
    -Inf, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1
  ))) %>%
  left_join(camels_topo, by = c("catchment_id" = "gauge_id")) %>%
  st_as_sf(coords = c("gauge_lon","gauge_lat"), remove = T)

st_crs(data_plot_diff) <-  st_crs(st_crs(usa))


A <- ggplot(usa) +
  geom_sf(color = "#2b2b2b", fill = "white", size=0.125) +
  geom_sf(data = data_plot2, aes(color = KGE_group), size = 0.8) +
  scale_color_viridis(discrete = TRUE, option="D")+
  coord_sf(crs = st_crs("+proj=laea +lat_0=45 +lon_0=-100 +x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs"), datum = NA) +
  labs(color="KGE group") + #  title = "(a) KGE group of generative model")+
  ggthemes::theme_map(base_size = 10)+
  theme(legend.position = "right",
        legend.key.height = unit(0.5, "cm"))

B <- ggplot(usa) +
  geom_sf(color = "#2b2b2b", fill = "white", size=0.125) +
  geom_sf(data = data_plot_diff, aes(color = diff_group), size = 0.8) +
  scale_color_manual(values = c('#a50026','#d73027','#f46d43','#fdae61','#fee08b','#d9ef8b','#a6d96a','#66bd63','#1a9850','#006837'), drop=F)+#scale_color_brewer(palette="RdYlGn", drop=FALSE)+
  coord_sf(crs = st_crs("+proj=laea +lat_0=45 +lon_0=-100 +x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs"), datum = NA) +
  labs(color="Difference in KGE\ncompard to best\nlumped model")+
  ggthemes::theme_map(base_size = 10)+
  theme(legend.position = "right",
        legend.key.height = unit(0.5, "cm"))

D <- ggplot(usa) +
  geom_sf(color = "#2b2b2b", fill = "white", size=0.125) +
  geom_sf(data = data_plot, aes(color = rank), size = 0.8) +
  scale_colour_gradientn(colours = terrain.colors(10))+
  coord_sf(crs = st_crs("+proj=laea +lat_0=45 +lon_0=-100 +x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs"), datum = NA) +
  labs(color="Rank") + #title = "(b) Rank of generative model")+
  ggthemes::theme_map(base_size = 10)+
  theme(legend.position = "right")

C <- cowplot::plot_grid(A, B, rel_widths = c(1,1), align="v", nrow = 2, labels = c("(a)", "(b)"), label_fontface = "plain", label_size = 12)

grDevices::cairo_pdf("./data/figures/fig_KGE_map.pdf", width = 6, height = 6)
C
dev.off()


st_write(obj = data_plot2, dsn = "./data/figures/camels_kge.shp",delete_dsn = T)
st_write(obj = data_plot_diff, dsn = "./data/figures/camels_kge_diff.shp",delete_dsn = T)
