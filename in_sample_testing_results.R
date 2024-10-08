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


# data --------------------------------------------------------------------

catchment_id <- read_csv("data/data/Caravan/non_US_catchments.csv") %>%
  rename(catchment_id = catchment_name)

train_scores <- read_csv("data/in_sample_results/train_scores_KGEs_test.csv", col_names = "Train") %>%
  mutate(catchment_id = catchment_id$catchment_id)
test_scores <- read_csv("data/in_sample_results/test_scores_KGEs_test.csv", col_names = "Test") %>%
  mutate(catchment_id = catchment_id$catchment_id)

data_process <- train_scores %>%
  left_join(test_scores, by = "catchment_id")


# Plot --------------------------------------------------------------------

# fig_ecdf
data_plot <- data_process %>%
  gather(item, value, Train, Test) %>%
  mutate(item = factor(item, levels = c("Train", "Test"), labels = c("Training", "Test")))

ggplot()+
  stat_ecdf(data = data_plot, aes(value, color = item)) +
  scale_y_continuous(breaks = c(0, 0.25, 0.5, 0.75, 1), labels = c(0, 0.25, 0.5, 0.75, 1))+
  coord_cartesian(xlim = c(-1,1), expand = 0) +
  annotate("text", x = -0.65, y = 0.28, label = paste0("Test median = ", sprintf("%.3f", round(median(data_process$Test, na.rm = T),3))), size = 3.8)+
  annotate("text", x = -0.6, y = 0.36, label = paste0("Training median = ", sprintf("%.3f", round(median(data_process$Train, na.rm = T),3))), size = 3.8)+
  labs(x = "KGE", y = "Non-exceedance probability", color = "")+
  theme_bw(base_size = 10)+
  theme(legend.position = "top")

ggsave(filename = "data/figures/fig_ecdf.pdf", width = 5, height = 4, units = "in")


# fig_caravan_train_vs_test
cor(data_process$Train[!is.na(data_process$Test)], data_process$Test[!is.na(data_process$Test)]) # 0.2707795

data_plot <- data_process %>%
  mutate(
    collection = sub("\\_.*", "", catchment_id),
    collection = factor(
      collection,
      levels = c(
        "camelsaus",
        "camelsbr",
        "camelscl",
        "camelsgb",
        "hysets",
        "lamah"
      ),
      labels = c(
        "CAMELS-AUS\n(Australia)",
        "CAMELS-BR\n(Brazil)",
        "CAMELS-CL\n(Chile)",
        "CAMELS-GB\n(Great Britain)",
        "HYSETS\n(North America)",
        "LamaH-CE\n(Central Europe)"
      )
    )
  )

ggplot(data_plot, aes(Train, Test)) +
  geom_point(
    fill = "steelblue",
    shape = 21,
    alpha = 0.5,
    size = 1,
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
  labs(x = "Training KGE", y = "Test KGE", linetype = "") +
  facet_wrap(~ collection, scales = "free") +
  theme_bw(10)+
  theme(legend.position = "top", legend.key.height = unit(0.25, "cm"),
        axis.text = element_text(size = 6.5),
        plot.margin = margin(0.35,0.5,0.35,0.35, "cm"))

ggsave(filename = "./data/figures/fig_caravan_train_vs_test.pdf", width = 7, height = 4, units = "in")
