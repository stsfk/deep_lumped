import numpy as np

import torch
import torch.nn as nn

import math

import models


class EarlyStopper:
    # Reference: https://stackoverflow.com/a/73704579/3361298

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def mse_loss_with_nans(input, target):
    # Adapted from https://stackoverflow.com/a/59851632/3361298

    # Missing data are nans
    mask = torch.isnan(target)

    out = (input[~mask] - target[~mask]) ** 2
    loss = out.mean()

    return loss


def val_model(
    embedding,
    decoder,
    dataset,
    storge_device,
    computing_device,
    use_amp,
    val_metric,
    return_summary=True,
):
    """Validate embedding and decoder using the validation batch from dataset and val_metric.

    Args:
        embedding (Embedding): model that map catchment_id (Tensor.int) to latent code [tensor].
        decoder (Decoder): decorder model.
        dataset (Forcing_Data): dataset to be used in validation.
        val_metric (function, optional): compute gof metric. Defaults to mse_loss_with_nans.
        return_summary (bool, optional): whether the gof metric or the raw prediciton should be returned. Defaults to True.
        val_steps(int, optional): Number of catchments evaluated at each steps. Defaults to 500.

    Returns:
        tensor: gof metric or raw prediction.
    """
    x, y = dataset.get_val_batch()

    embedding.eval()
    decoder.eval()

    preds = torch.ones(size=y.shape, device=storge_device)

    n_catchments = y.shape[1]
    selected_catchments = torch.arange(n_catchments, device=computing_device)

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        with torch.no_grad():
            code = embedding(selected_catchments)
            for i in range(x.shape[0]):
                x_sub = x[i, :, :, :]
                preds[i, :, :] = decoder.decode(code, x_sub)

    if return_summary:
        out = val_metric(preds, y)
    else:
        out = preds

    return out


def val_model_mem_saving(
    embedding,
    decoder,
    dataset,
    storge_device,
    computing_device,
    use_amp,
    val_metric,
    return_summary,
    val_steps,
):
    """Validate embedding and decoder using the validation batch from dataset and val_metric.

    Args:
        embedding (Embedding): model that map catchment_id (Tensor.int) to latent code [tensor].
        decoder (Decoder): decorder model.
        dataset (Forcing_Data): dataset to be used in validation.
        val_metric (function, optional): compute gof metric. Defaults to mse_loss_with_nans.
        return_summary (bool, optional): whether the gof metric or the raw prediciton should be returned. Defaults to True.
        val_steps(int, optional): Number of catchments evaluated at each steps. Defaults to 500.

    Returns:
        tensor: gof metric or raw prediction.
    """
    x, y = dataset.get_val_batch()

    embedding.eval()
    decoder.eval()

    preds = torch.ones(size=y.shape, device=storge_device)
    n_catchments = y.shape[1]

    # iterate over years
    for i in range(x.shape[0]):
        # iterate over catchments
        for j in range(math.ceil(n_catchments / val_steps)):
            start_catchment_ind = j * val_steps
            end_catchment_ind = min((j + 1) * val_steps, n_catchments)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                with torch.no_grad():
                    code = embedding(
                        torch.arange(
                            start=start_catchment_ind,
                            end=end_catchment_ind,
                            device=computing_device,
                        )
                    )
                    x_sub = x[i, start_catchment_ind:end_catchment_ind, :, :].to(
                        computing_device
                    )
                    preds[i, start_catchment_ind:end_catchment_ind, :] = decoder.decode(
                        code, x_sub
                    ).to(storge_device)

    if return_summary:
        out = val_metric(preds, y)
    else:
        out = preds

    return out


class TCN_model_builder:
    def __init__(self, n_catchments, base_length=365, forcing_dim=3):
        self.n_catchments = n_catchments
        self.base_length = base_length
        self.forcing_dim = forcing_dim

    def define_model(self, trial):

        # FORCING_DIM is from global

        # latent dim
        latent_dim_power = trial.suggest_int("latent_dim_power", 1, 2)
        latent_dim = 2**latent_dim_power

        # kernel_size
        kernel_size = trial.suggest_int("kernel_size", 2, 6)

        # num_channels
        # ref: https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/
        hidden_channel_dim = trial.suggest_int("hidden_channel_dim", 1, 256)
        base = 2  # dilation factor
        n_levels = math.log(
            (self.base_length - 1) * (base - 1) / (kernel_size - 1) / 2 + 1
        ) / math.log(2)
        n_levels = math.ceil(n_levels)

        num_channels = []
        for i in range(n_levels - 1):
            num_channels.append(hidden_channel_dim)

        num_channels.append(1)  # output dim = 1

        # p
        drop_out_flag = trial.suggest_categorical("drop_out_flag", [True, False])

        if drop_out_flag:
            p = trial.suggest_float("dropout_rate", 0.1, 0.5)
        else:
            p = 0

        # define model
        decoder = models.TCN_decoder(
            latent_dim=latent_dim,
            feature_dim=self.forcing_dim,
            num_channels=num_channels,
            kernel_size=kernel_size,
            p=p,
        )

        embedding = nn.Embedding(self.n_catchments, latent_dim)

        return embedding, decoder


class LSTM_model_builder:
    def __init__(self, n_catchments, base_length=365, forcing_dim=3):
        self.n_catchments = n_catchments
        self.base_length = base_length
        self.forcing_dim = forcing_dim

    def define_model(self, trial):
        lstm_hidden_dim = trial.suggest_int("lstm_hidden_dim", 4, 256)
        n_lstm_layers = trial.suggest_int("n_lstm_layers", 1, 2)
        n_fc_layers = trial.suggest_int("n_fc_layers", 1, 3)
        latent_dim_power = trial.suggest_int("LATENT_DIM_power", 1, 4)
        latent_dim = 2**latent_dim_power

        drop_out_flag = trial.suggest_categorical("drop_out_flag", [True, False])

        if drop_out_flag:
            p = trial.suggest_float("dropout_rate", 0.2, 0.5)
        else:
            p = 0

        fc_hidden_dims = []
        for i in range(n_fc_layers):
            fc_dim = trial.suggest_int(f"fc_dim{i}", 2, 32)
            fc_hidden_dims.append(fc_dim)

        decoder = models.LSTM_decoder(
            latent_dim=latent_dim,
            feature_dim=self.forcing_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            fc_hidden_dims=fc_hidden_dims,
            num_lstm_layers=n_lstm_layers,
            output_dim=1,
            p=p,
        )

        embedding = nn.Embedding(self.n_catchments, latent_dim)

        return embedding, decoder
