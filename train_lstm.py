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

N_CATCHMENTS = 3007

# training hyperparameters
EPOCHS = 200
TRAIN_YEAR = 19
PATIENCE = 10

use_amp = True
compile_model = False

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
    "./data/data_train_CARAVAN.csv",
    record_length=7304,
    storge_device=storge_device,
    seq_length=SEQ_LENGTH,
    target_seq_length=TARGET_SEQ_LENGTH,
    base_length=BASE_LENGTH,
)

dval = dataloader.Forcing_Data(
    "data/data_val_CARAVAN.csv",
    record_length=4017,
    storge_device=storge_device,
    seq_length=SEQ_LENGTH,
    target_seq_length=TARGET_SEQ_LENGTH,
    base_length=BASE_LENGTH,
)


# %%
class Objective:
    def __init__(self, model_builder):
        self.model_builder = model_builder

    def objective(self, trial):
        # prepare early stopper
        early_stopper = training_fun.EarlyStopper(patience=PATIENCE, min_delta=0)

        # define model
        embedding, decoder = self.model_builder.define_model(trial)
        embedding, decoder = embedding.to(computing_device), decoder.to(
            computing_device
        )

        if compile_model:
            # pytorch2.0 new feature, complile model for fast training
            embedding, decoder = torch.compile(embedding), torch.compile(decoder)

        # define optimizers
        lr_embedding = trial.suggest_float("lr_embedding", 5e-5, 1e-2, log=True)
        embedding_optimizer = optim.Adam(embedding.parameters(), lr=lr_embedding)

        lr_decoder = trial.suggest_float("lr_decoder", 5e-5, 1e-2, log=True)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr_decoder)

        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # define batch size
        batch_size_power = trial.suggest_int("batch_size_power", 4, 8)
        batch_size = 2**batch_size_power

        # steps per epoch
        steps = round(N_CATCHMENTS * TRAIN_YEAR / batch_size)

        # train model
        for epoch in range(EPOCHS):
            # for each epoch get_random_batch method generates a batch that contains one year data for each catchment
            # repeat TRAIN_YEAR times to finish an epoch
            decoder.train()
            embedding.train()

            for step in range(steps):
                decoder_optimizer.zero_grad()
                embedding_optimizer.zero_grad()

                # put the models into training mode
                decoder.train()
                embedding.train()

                # get training batch and pass to device
                (x_batch, y_batch, selected_catchments) = dtrain.get_random_batch(
                    batch_size
                )

                x_batch, y_batch, selected_catchments = (
                    x_batch.to(computing_device),
                    y_batch.to(computing_device),
                    selected_catchments.to(computing_device),
                )

                # slice batch for training
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=use_amp
                ):
                    code = embedding(selected_catchments)

                    # pass through decoder
                    out = decoder.decode(code, x_batch)

                    # compute loss
                    loss = training_fun.mse_loss_with_nans(out, y_batch)

                scaler.scale(loss).backward()
                scaler.step(embedding_optimizer)
                scaler.step(decoder_optimizer)
                scaler.update()

            # validate model after each epochs
            decoder.eval()
            embedding.eval()

            # Handle pruning based on the intermediate value
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

            trial.report(val_loss, epoch)

            if trial.should_prune():
                torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()

            # Early stop using early_stopper, break for loop
            if early_stopper.early_stop(val_loss):
                break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return early_stopper.min_validation_loss


# %%
LSTM_model_builder = training_fun.LSTM_model_builder(
    n_catchments=N_CATCHMENTS, base_length=BASE_LENGTH, forcing_dim=FORCING_DIM
)

LSTM_objective = Objective(LSTM_model_builder).objective

# %%
study = optuna.create_study(
    study_name="base_model", direction="minimize", pruner=optuna.pruners.NopPruner()
)
study.optimize(LSTM_objective, n_trials=200)

joblib.dump(study, "data/base_lstm_study2.pkl")
