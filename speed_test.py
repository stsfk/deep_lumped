# %%
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset

import numpy as np

import math

import time

import dataloader
import models
import training_fun

import optuna

import joblib

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LENGTH = 365 * 2
TARGET_SEQ_LENGTH = 365
BASE_LENGTH = SEQ_LENGTH - TARGET_SEQ_LENGTH

FORCING_DIM = 3

N_CATCHMENTS = 2346

# training hyperparameters
TRAIN_YEAR = 19

use_amp = True
compile_model = True

if compile_model:
    torch.set_float32_matmul_precision("high")

memory_saving = True
if memory_saving:
    storge_device = "cpu"
    computing_device = DEVICE
    VAL_STEPS = 500
else:
    storge_device = DEVICE
    computing_device = DEVICE

# %%
dtrain = dataloader.Forcing_Data(
    "data/data_train_w_missing.csv",
    record_length=7304,
    storge_device=storge_device,
    seq_length=SEQ_LENGTH,
    target_seq_length=TARGET_SEQ_LENGTH,
    base_length=BASE_LENGTH,
)

dval = dataloader.Forcing_Data(
    "data/data_val_w_missing.csv",
    record_length=4017,
    storge_device=storge_device,
    seq_length=SEQ_LENGTH,
    target_seq_length=TARGET_SEQ_LENGTH,
    base_length=BASE_LENGTH,
)

# %%
class TCN_Model:
    def __init__(
        self, hidden_channel_dim=128, kernel_size=3, p=0.5, feature_dim=3, latent_dim=4
    ):
        # N_CATCHMENT is from global

        # num_channels
        # ref: https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/
        base = 2  # dilation factor
        n_levels = math.log(
            (BASE_LENGTH - 1) * (base - 1) / (kernel_size - 1) / 2 + 1
        ) / math.log(2)
        n_levels = math.ceil(n_levels)

        num_channels = []
        for i in range(n_levels - 1):
            num_channels.append(hidden_channel_dim)

        num_channels.append(1)  # output dim = 1

        self.decoder = models.TCN_decoder(
            latent_dim=latent_dim,
            feature_dim=feature_dim,
            num_channels=num_channels,
            kernel_size=kernel_size,
            p=p,
        )

        self.embedding = nn.Embedding(N_CATCHMENTS, latent_dim)


class LSTM_model:
    def __init__(
        self,
        latent_dim,
        lstm_hidden_dim,
        n_lstm_layers,
        fc_hidden_dims,
        p,
        feature_dim=3,
        output_dim=1,
    ):
        # N_CATCHMENT is from global
        self.decoder = models.LSTM_decoder(
            latent_dim=latent_dim,
            feature_dim=feature_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            fc_hidden_dims=fc_hidden_dims,
            num_lstm_layers=n_lstm_layers,
            output_dim=output_dim,
            p=p,
        )

        self.embedding = nn.Embedding(N_CATCHMENTS, latent_dim)


def speed_test(model, epochs=2, batch_size=64, lr_embedding=0.001, lr_decoder=0.001):

    # prepare early stopper
    early_stopper = training_fun.EarlyStopper(patience=1000, min_delta=0)

    # define model
    embedding, decoder = model.embedding.to(computing_device), model.decoder.to(
        computing_device
    )

    if compile_model:
        # pytorch2.0 new feature, complile model for fast training
        embedding, decoder = torch.compile(embedding), torch.compile(decoder)

    # define optimizers
    embedding_optimizer = optim.Adam(embedding.parameters(), lr=lr_embedding)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr_decoder)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # train model
    for epoch in range(epochs):

        # for each epoch get_random_batch method generates a batch that contains one year data for each catchment
        # repeat TRAIN_YEAR times to finish an epoch
        decoder.train()
        embedding.train()

        for year in range(TRAIN_YEAR):

            x_batch, y_batch = dtrain.get_random_batch()

            if memory_saving:
                x_batch, y_batch = x_batch.to(computing_device), y_batch.to(
                    computing_device
                )

            catchment_index = torch.randperm(
                N_CATCHMENTS, device=computing_device
            )  # add randomness

            # interate over catchments
            for i in range(int(N_CATCHMENTS / batch_size)):

                # prepare data
                ind_s = i * batch_size
                ind_e = (i + 1) * batch_size

                selected_catchments = catchment_index[ind_s:ind_e]

                x_sub, y_sub = x_batch[ind_s:ind_e, :, :], y_batch[ind_s:ind_e, :]

                # prepare training, put the models into training mode
                decoder_optimizer.zero_grad()
                embedding_optimizer.zero_grad()

                # forward pass
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=use_amp
                ):
                    code = embedding(selected_catchments)
                    out = decoder.decode(code, x_sub)

                    # backprop
                    loss = training_fun.mse_loss_with_nans(out, y_sub)

                scaler.scale(loss).backward()
                scaler.step(embedding_optimizer)
                scaler.step(decoder_optimizer)
                scaler.update()

        # validate model after each epochs
        decoder.eval()
        embedding.eval()

        if memory_saving:
            val_loss = training_fun.val_model_mem_saving(
                embedding=embedding,
                decoder=decoder,
                dataset=dval,
                storge_device=storge_device,
                computing_device=computing_device,
                use_amp=use_amp,
                val_metric=training_fun.mse_loss_with_nans,
                return_summary=True,
                val_steps=VAL_STEPS,
            )
        else:
            val_loss = (
                training_fun.val_model(
                    embedding=embedding,
                    decoder=decoder,
                    dataset=dval,
                    storge_device=storge_device,
                    computing_device=computing_device,
                    use_amp=use_amp,
                    val_metric=training_fun.mse_loss_with_nans,
                    return_summary=True,
                )
                .detach()
                .cpu()
                .numpy()
            )

        # Early stop using early_stopper, break for loop
        if early_stopper.early_stop(val_loss):
            break

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return early_stopper.min_validation_loss


# lstm_model = LSTM_model(
#     latent_dim=4,
#     lstm_hidden_dim=128,
#     n_lstm_layers=2,
#     fc_hidden_dims=[16, 8, 4],
#     p=0.5,
#     feature_dim=3,
#     output_dim=1,
# )

# starting_time = time.time()
# print("Process started...")

# fit = speed_test(lstm_model, epochs=10)

# print("Process ended...")
# ending_time = time.time()
# print(ending_time - starting_time)

# print(f"fit={fit}")


tcn_model = TCN_Model(hidden_channel_dim=128, kernel_size=3, p=0.5)

starting_time = time.time()
print("Process started...")

fit = speed_test(tcn_model, epochs=10)

print("Process ended...")
ending_time = time.time()
print(ending_time - starting_time)

print(f"fit={fit}")
