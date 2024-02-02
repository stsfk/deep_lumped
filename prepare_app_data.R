library(tidyverse)

data_train_val <- read_csv("./data/camels_train_val.csv")
data_test <- read_csv("./data/camels_test.csv")


app_train <- data_train_val[1:3652,]
app_test <- data_test[1:4383,]

write_csv(app_train, col_names = F, file = "./data/app_train.csv")
write_csv(app_test, col_names = F, file = "./data/app_test.csv")
