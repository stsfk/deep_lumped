if (!require("pacman")) {
  install.packages("pacman")
}

pacman::p_load(
  tidyverse,
  lubridate,
  zeallot
)

# 01013500;01;Fish River near Fort Kent, Maine

# data --------------------------------------------------------------------

original <- read_csv("./data/hydrograph/original.csv", col_names = "Q") %>%
  mutate(time = 1:n())

date <- read_csv("./data/CAMELS_Knoben/date.csv", col_names = "date") # the raw data only have a datenum column
selected_catchment_id <- read_csv("./data/CAMELS_Knoben/knoben_selected_catchment.csv") %>% pull(catchment_id) %>% unique()
 

variations <- expand_grid(dim = c(0:7),
            var = c(-0.5, -0.25, -0.1, 0.1, 0.25, 0.5)) %>%
  mutate(
    file_name = paste0("./data/hydrograph/dim_", dim, "_var_", var, ".csv"),
    data = vector("list", 1),
    data = map(file_name, function(x)
      read_csv(x, col_names = "Q_var") %>% mutate(time = 1:n()))
  )

data_plot <- variations %>%
  unnest(data)

data_plot_original <- variations %>%
  unnest(data) %>%
  left_join(original, by = "time") %>%
  select(-Q_var) %>%
  rename(Q_var = Q) %>%
  filter(var == -0.5) %>%
  mutate(var = 0)

data_plot <- data_plot %>%
  bind_rows(data_plot_original)%>%
  mutate(dim = dim + 1,
         dim = factor(dim, labels = paste0("dim=", 1:8))) %>%
  filter(time %in% c(6650:6850), # which max = 6696
         var %in%  c(-0.5, -0.1, 0, 0.1, 0.5)) %>%
  mutate(var = factor(var, levels = c(-0.5, -0.1, 0, 0.1, 0.5), labels = c("-50%", "-10%", "no change", "+10%", "50%")),
         time = ymd("1989-12-31") + time)

ggplot(data_plot) +
  geom_line(aes(time, Q_var, color = var), alpha = 0.9, linewidth = 0.4) +
  scale_color_manual(values = c("steelblue","palegreen3", "black","yellow3", "#FB9A99"))+
  facet_wrap( ~ dim, nrow = 4, strip.position = "right")+
  scale_x_date(date_labels = "%d-%b-%Y")+
  labs(x="Time", y="Q [mm/day]", color = "Change in value")+
  theme_bw(base_size = 10)+
  theme(legend.position = "top",
        axis.title.x = element_blank())

ggsave(filename = "./data/results/fig_hydrographs.pdf", width = 7, height = 4, units = "in")
 
  
