import torch

from torch.utils.data import Dataset

import numpy as np

import math

import random


class Forcing_Data(Dataset):
    def __init__(
        self,
        fpath="data/data_train_w_missing.csv",
        record_length=7304,
        n_feature=3,
        storge_device="cpu",
        seq_length=730,
        target_seq_length=365,
        base_length=365,
    ):
        data_raw = np.genfromtxt(fpath, delimiter=",", skip_header=1)

        # normalization and then reshape to catchment*record*feature
        x = torch.from_numpy(data_raw[:, 0:n_feature]).to(dtype=torch.float32)
        x = x.view(-1, record_length, n_feature).contiguous()
        self.x = x.to(storge_device)

        # normalization and then reshape to catchment*record
        y = torch.from_numpy(data_raw[:, n_feature]).to(dtype=torch.float32)
        y = y.view(-1, record_length).contiguous()
        self.y = y.to(storge_device)

        self.n_catchment = y.shape[0]

        self.n_feature = n_feature

        self.record_length = self.x.shape[1]
        self.seq_length = seq_length
        self.target_seq_length = target_seq_length
        self.base_length = base_length

        self.storge_device = storge_device

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]

    def get_random_batch(self, batch_size=64, remove_nan=True):
        # This fuction return a input and output pair for each catchment
        # reference: https://medium.com/@mbednarski/understanding-indexing-with-pytorch-gather-33717a84ebc4
        # https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
        indices = random.sample(range(self.n_catchment), batch_size)
        selected_catchment_index = torch.tensor(indices, device=self.storge_device)
        
        # selected_catchment_index = torch.randint(
        #     low=0, high=self.n_catchment, size=(batch_size,), device=self.storge_device
        # )

        x_sub = torch.index_select(self.x, dim=0, index=selected_catchment_index)
        y_sub = torch.index_select(self.y, dim=0, index=selected_catchment_index)

        # randomly selects a starting time step for each catchment
        index = torch.randint(
            low=0,
            high=self.record_length - self.seq_length + 1,
            size=(batch_size,),
            device=self.storge_device,
        )

        # expand the index to have the length of seq_length, adding 0 to seq_length to get correct index
        index_y = index.unsqueeze(-1).repeat(1, self.seq_length) + torch.arange(
            self.seq_length, device=self.storge_device
        )
        index_x = index_y.unsqueeze(-1).repeat(1, 1, self.n_feature)

        # use gather function to output values
        x_batch, y_batch = x_sub.gather(dim=1, index=index_x), y_sub.gather(
            dim=1, index=index_y
        )

        if remove_nan:
            valid_sample_id = ~torch.any(
                x_batch.isnan().view(x_batch.shape[0], -1), dim=1
            )
            x_batch = x_batch[valid_sample_id]
            y_batch = y_batch[valid_sample_id]
            selected_catchment_index = selected_catchment_index[valid_sample_id]

        return (
            x_batch,
            y_batch[:, self.base_length :],
            selected_catchment_index,
        )

    def get_val_batch(self):
        n_years = math.ceil(
            (self.record_length - self.base_length) / self.target_seq_length
        )

        out_x = (
            torch.ones(
                [n_years, self.n_catchment, self.seq_length, self.n_feature],
                device=self.storge_device,
            )
            * torch.nan
        )
        out_y = (
            torch.ones(
                [n_years, self.n_catchment, self.seq_length], device=self.storge_device
            )
            * torch.nan
        )

        for i in range(n_years):
            start_record_ind = self.base_length * i

            if i == n_years - 1:
                end_record_ind = self.record_length

                out_x[i, :, 0 : (end_record_ind - start_record_ind), :] = self.x[
                    :, start_record_ind:end_record_ind, :
                ]
                out_y[i, :, 0 : (end_record_ind - start_record_ind)] = self.y[
                    :, start_record_ind:end_record_ind
                ]

            else:
                end_record_ind = start_record_ind + self.seq_length

                out_x[i, :, :, :] = self.x[:, start_record_ind:end_record_ind, :]
                out_y[i, :, :] = self.y[:, start_record_ind:end_record_ind]

        return out_x, out_y[:, :, self.base_length :]

    def get_catchment_random_batch(self, selected_catchment, batch_size=64):

        # This fuction return a input and output pair for each catchment
        # reference: https://medium.com/@mbednarski/understanding-indexing-with-pytorch-gather-33717a84ebc4
        # https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms

        selected_catchment_index = (
            torch.ones(size=(batch_size,), dtype=torch.int64, device=self.storge_device)
            * selected_catchment
        )

        x_sub = torch.index_select(self.x, dim=0, index=selected_catchment_index)
        y_sub = torch.index_select(self.y, dim=0, index=selected_catchment_index)

        # randomly selects a starting time step for each catchment
        index = torch.randint(
            low=0,
            high=self.record_length - self.seq_length + 1,
            size=(batch_size,),
            device=self.storge_device,
        )

        # expand the index to have the length of seq_length, adding 0 to seq_length to get correct index
        index_y = index.unsqueeze(-1).repeat(1, self.seq_length) + torch.arange(
            self.seq_length, device=self.storge_device
        )
        index_x = index_y.unsqueeze(-1).repeat(1, 1, self.n_feature)

        # use gather function to output values
        x_batch, y_batch = x_sub.gather(dim=1, index=index_x), y_sub.gather(
            dim=1, index=index_y
        )

        return (
            x_batch,
            y_batch[:, self.base_length :],
            selected_catchment_index,
        )

    def get_catchment_val_batch(self, selected_catchment):
        n_years = math.ceil(
            (self.record_length - self.base_length) / self.target_seq_length
        )

        out_x = (
            torch.ones(
                [n_years, self.n_catchment, self.seq_length, self.n_feature],
                device=self.storge_device,
            )
            * torch.nan
        )
        out_y = (
            torch.ones(
                [n_years, self.n_catchment, self.seq_length], device=self.storge_device
            )
            * torch.nan
        )

        for i in range(n_years):
            start_record_ind = self.base_length * i

            if i == n_years - 1:
                end_record_ind = self.record_length

                out_x[i, :, 0 : (end_record_ind - start_record_ind), :] = self.x[
                    :, start_record_ind:end_record_ind, :
                ]
                out_y[i, :, 0 : (end_record_ind - start_record_ind)] = self.y[
                    :, start_record_ind:end_record_ind
                ]

            else:
                end_record_ind = start_record_ind + self.seq_length

                out_x[i, :, :, :] = self.x[:, start_record_ind:end_record_ind, :]
                out_y[i, :, :] = self.y[:, start_record_ind:end_record_ind]

        return (
            out_x[:, selected_catchment, :, :],
            out_y[:, selected_catchment, self.base_length :],
        )
