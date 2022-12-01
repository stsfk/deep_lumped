# %%
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset

import numpy as np

from collections import defaultdict

import math

import pandas as pd

import optuna

import joblib

import tcn

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LENGTH = 365 * 2
TARGET_SEQ_LENGTH = 365
BASE_LENGTH = SEQ_LENGTH - TARGET_SEQ_LENGTH

FORCING_DIM = 3

N_CATCHMENT = 1758

EPOCHS = 500

TRAIN_YEAR = 19

PATIENCE = 10

dtypes = defaultdict(lambda: float)
dtypes["catchment_id"] = str

use_amp = True

# %%
class Forcing_Data(Dataset):
    def __init__(
        self,
        fpath="data/Caravan/data_train_w_missing.csv",
        record_length=7304,
        n_feature=3,
    ):
        data_raw = pd.read_csv(fpath, dtype=dtypes)

        # normalization and then reshape to catchment*record*feature
        x = data_raw.loc[:, "P":"PET"]

        x = torch.tensor(x.values, dtype=torch.float32)
        x = x.view(-1, record_length, n_feature)
        self.x = x.to(DEVICE)

        # normalization and then reshape to catchment*record
        y = data_raw["Q"]

        y = torch.tensor(y.values, dtype=torch.float32)
        y = y.view(-1, record_length)
        self.y = y.to(DEVICE)

        self.record_length = self.x.shape[1]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]

    def get_random_batch(self):
        # This fuction return a input and output pair for each catchment
        # SEQ_LENGTH, BASE_LENGTH, and DEVICE is from global
        # reference: https://medium.com/@mbednarski/understanding-indexing-with-pytorch-gather-33717a84ebc4
        # https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms

        # randomly selects a starting time step for each catchment
        index = torch.randint(
            low=0,
            high=self.record_length - SEQ_LENGTH + 1,
            size=(N_CATCHMENT,),
            device=DEVICE,
        )

        # expand the index to have the length of SEQ_LENGTH, adding 0 to SEQ_LENGTH to get correct index
        index_y = index.unsqueeze(-1).repeat(1, SEQ_LENGTH) + torch.arange(
            SEQ_LENGTH, device=DEVICE
        )
        index_x = index_y.unsqueeze(-1).repeat(1, 1, FORCING_DIM)

        # use gather function to output values
        x_batch, y_batch = self.x.gather(dim=1, index=index_x), self.y.gather(
            dim=1, index=index_y
        )

        return x_batch, y_batch[:, BASE_LENGTH:]

    def get_val_batch(self, device=DEVICE):
        n_years = math.ceil((self.record_length - BASE_LENGTH) / TARGET_SEQ_LENGTH)

        out_x = (
            torch.ones([n_years, N_CATCHMENT, SEQ_LENGTH, FORCING_DIM], device=device)
            * torch.nan
        )
        out_y = (
            torch.ones([n_years, N_CATCHMENT, SEQ_LENGTH], device=device) * torch.nan
        )

        for i in range(n_years):
            start_record_ind = BASE_LENGTH * i

            if i == n_years - 1:
                end_record_ind = self.record_length

                out_x[i, :, 0 : (end_record_ind - start_record_ind), :] = self.x[
                    :, start_record_ind:end_record_ind, :
                ]
                out_y[i, :, 0 : (end_record_ind - start_record_ind)] = self.y[
                    :, start_record_ind:end_record_ind
                ]

            else:
                end_record_ind = start_record_ind + SEQ_LENGTH

                out_x[i, :, :, :] = self.x[:, start_record_ind:end_record_ind, :]
                out_y[i, :, :] = self.y[:, start_record_ind:end_record_ind]

        return out_x, out_y[:, :, BASE_LENGTH:]


# %%
dtrain = Forcing_Data("data_train_w_missing.csv", record_length=7304)
dval = Forcing_Data("data_val_w_missing.csv", record_length=4017)

# %%
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


# %%
class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        feature_dim,
        num_channels,
        kernel_size,
        p=0.2,
    ):
        super(Decoder, self).__init__()

        self.num_inputs = latent_dim + feature_dim
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.p = p

        self.tcn = tcn.TemporalConvNet(
            num_inputs=self.num_inputs,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.p,
        )

    def forward(self, inputs):
        """Forward pass

        Args:
            inputs (tensor): shape = [batch_size, input channels, seq_len]

        Returns:
            tensor: shape = [batch_size, output channels, seq_len]
        """
        return self.tcn(inputs)

    def decode(self, code, x, base_length=BASE_LENGTH):
        """Predict hydrographs under forcing x for catchments charaterized by `code`

        Args:
            code (tensor): shape = [batch_size, latent dim]
            x (tensor): shape = [batch_size, seq_len, latent dim]
            base_length (int): predictions for the first 'base_length' time step is ignored. Defaults to BASE_LENGTH.

        Returns:
            tensor: shape = [base_size, seq_len - base_length]
        """

        code = code.expand(x.shape[1], -1, -1).transpose(
            0, 1
        )  # new shape [batch_size, seq_len, latent dim

        x = torch.cat((code, x), 2)  # concatenate code and x in dim 2

        x = x.transpose(
            1, 2
        ).contiguous()  # new shape [batch_size, input channel, seq_len]

        out = (
            self.tcn(x).transpose(1, 2)[:, base_length:, :].squeeze()
        )  # new shape [batch_size, Target length]

        return out


# %%
def mse_loss_with_nans(input, target):
    # Adapted from https://stackoverflow.com/a/59851632/3361298

    # Missing data are nans
    mask = torch.isnan(target)

    out = (input[~mask] - target[~mask]) ** 2
    loss = out.mean()

    return loss


# %%
def val_model(
    embedding, decoder, dataset, val_metric=mse_loss_with_nans, return_summary=True
):
    """Validate embedding and decoder using the validation batch from dataset and val_metric.

    Args:
        embedding (Embedding): model that map catchment_id (Tensor.int) to latent code [tensor].
        decoder (Decoder): decorder model.
        dataset (Forcing_Data): dataset to be used in validation.
        val_metric (function, optional): compute gof metric. Defaults to mse_loss_with_nans.
        return_summary (bool, optional): whether the gof metric or the raw prediciton should be returned. Defaults to True.

    Returns:
        tensor: gof metric or raw prediction.
    """
    x, y = dataset.get_val_batch()

    embedding.eval()
    decoder.eval()

    preds = torch.ones(size=y.shape).to(DEVICE)

    selected_catchments = torch.arange(N_CATCHMENT).to(DEVICE)

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        with torch.no_grad():
            code = embedding(selected_catchments)

            for i in range(x.shape[0]):
                x_sub = x[i, :, :, :]
                preds[i, :, :] = decoder.decode(code, x_sub)

    if return_summary:
        out = val_metric(preds, y)
    else:
        out = (preds, y)

    return out


# %%
def define_model(trial):

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
        (BASE_LENGTH - 1) * (base - 1) / (kernel_size - 1) / 2 + 1
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
    decoder = Decoder(
        latent_dim=latent_dim,
        feature_dim=FORCING_DIM,
        num_channels=num_channels,
        kernel_size=kernel_size,
        p=p,
    )

    embedding = nn.Embedding(N_CATCHMENT, latent_dim)

    return embedding, decoder


# %%
def objective(trial):

    val_losses = []

    # prepare early stopper
    early_stopper = EarlyStopper(patience=PATIENCE, min_delta=0)

    # define model
    embedding, decoder = define_model(trial)
    embedding, decoder = embedding.to(DEVICE), decoder.to(DEVICE)

    # define optimizers
    lr_embedding = trial.suggest_float("lr_embedding", 5e-5, 1e-2, log=True)
    embedding_optimizer = optim.Adam(embedding.parameters(), lr=lr_embedding)

    lr_decoder = trial.suggest_float("lr_decoder", 5e-5, 1e-2, log=True)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr_decoder)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # define batch size
    batch_size_power = trial.suggest_int("batch_size_power", 4, 7)
    batch_size = 2**batch_size_power

    # train model
    for epoch in range(EPOCHS):

        # for each epoch get_random_batch method generates a batch that contains one year data for each catchment
        # repeat TRAIN_YEAR times to finish an epoch
        decoder.train()
        embedding.train()

        for year in range(TRAIN_YEAR):

            x_batch, y_batch = dtrain.get_random_batch()
            catchment_index = torch.randperm(N_CATCHMENT).to(DEVICE)  # add randomness

            # interate over catchments
            for i in range(int(N_CATCHMENT / batch_size)):

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
                    loss = mse_loss_with_nans(out, y_sub)

                scaler.scale(loss).backward()
                scaler.step(embedding_optimizer)
                scaler.step(decoder_optimizer)
                scaler.update()

        # validate model after each epochs
        decoder.eval()
        embedding.eval()

        val_loss = val_model(embedding, decoder, dval).detach().cpu().numpy()

        trial.report(val_loss, epoch)

        # Handle pruning based on the intermediate value
        val_losses.append(val_loss)

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
study = optuna.create_study(
    study_name="base_model", direction="minimize", pruner=optuna.pruners.NopPruner()
)
study.optimize(objective, n_trials=10)

# %%
joblib.dump(study, "base_TCN_study.pkl")
