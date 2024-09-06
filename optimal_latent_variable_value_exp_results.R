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
  cowplot
)



# Plot parameter value distribution ---------------------------------------


data_plot <- read_csv("./data/optimal_latent_variable_exp_results10/KGEs.csv", col_names = "KGE") %>%
  mutate(trial = 1:10) %>%
  cbind(read_csv("./data/optimal_latent_variable_exp_results10/normalized_embeddings.csv", col_names = paste0("Latent value ", 1:8)))

data_plot %>%
  gather(item, value, `Latent value 1`: `Latent value 8`) %>%
  ggplot(aes(item, value))+
  geom_line(aes(group = trial, color = KGE))+
  geom_point(aes(color = KGE), size = 2)+
  scale_color_continuous(type = "viridis")+
  labs(y = "Normalized value") +
  theme_bw(base_size = 10)+
  theme(axis.title.x =element_blank(),
        axis.text.x = element_text(angle = 30,hjust = 1),
        legend.position = "right",
        legend.key.height=unit(0.5,"cm"))

ggsave(filename = "data/figures/fig_latent_variable_distribution.pdf", width = 6, height = 3.5, units = "in")



# Plot predicted discharge ------------------------------------------------


ob <- read_csv("./data/optimal_latent_variable_exp_results10/ob.csv", col_names = c("ob"))

preds <- read_csv("./data/optimal_latent_variable_exp_results10/preds.csv",col_names = F)
preds <- as_tibble(t(preds))

data_plot <- cbind(ob, preds) %>%
  tibble() %>%
  mutate(Date = seq(from = ymd("1999-01-01"), by = "1 day", length.out = nrow(preds))) %>%
  gather(version, value, ob:V10) %>%
  mutate(type = replace(version, version != "ob", "pred"),
         type = factor(type, levels = c("ob", "pred"), labels = c("observation", "Prediction")))


data_plot %>%
  filter(Date > '2008-01-01', Date < '2009-01-01') %>%
  ggplot() +
  geom_line(aes(Date, value, group = version, color = type),  alpha = 0.5)+
  scale_color_manual(values = c("red3", "grey60")) +
  labs(y = "Discharge [mm/day]",
       color = "") +
  theme_bw(base_size = 10)+
  theme(legend.position = "top")

ggsave(filename = "data/figures/fig_predicted_hydrograph.pdf", width = 8, height = 4, units = "in")

