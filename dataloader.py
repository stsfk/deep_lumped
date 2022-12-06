import torch

from torch.utils.data import Dataset

import numpy as np

import math


class Forcing_Data(Dataset):
    def __init__(
        self,
        fpath="data/data_train_w_missing.csv",
        record_length=7304,
        n_feature=3,
    ):
        data_raw = np.genfromtxt(fpath, delimiter=",", skip_header=1)

        # normalization and then reshape to catchment*record*feature
        x = torch.from_numpy(data_raw[:, 0:3]).to(dtype=torch.float32)
        x = x.view(-1, record_length, n_feature).contiguous()
        self.x = x.to(storge_device)

        # normalization and then reshape to catchment*record
        y = torch.from_numpy(data_raw[:, 3]).to(dtype=torch.float32)
        y = y.view(-1, record_length).contiguous()
        self.y = y.to(storge_device)

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
            device=storge_device,
        )

        # expand the index to have the length of SEQ_LENGTH, adding 0 to SEQ_LENGTH to get correct index
        index_y = index.unsqueeze(-1).repeat(1, SEQ_LENGTH) + torch.arange(
            SEQ_LENGTH, device=storge_device
        )
        index_x = index_y.unsqueeze(-1).repeat(1, 1, FORCING_DIM)

        # use gather function to output values
        x_batch, y_batch = self.x.gather(dim=1, index=index_x), self.y.gather(
            dim=1, index=index_y
        )

        return x_batch, y_batch[:, BASE_LENGTH:]

    def get_val_batch(self):
        n_years = math.ceil((self.record_length - BASE_LENGTH) / TARGET_SEQ_LENGTH)

        out_x = (
            torch.ones(
                [n_years, N_CATCHMENT, SEQ_LENGTH, FORCING_DIM], device=storge_device
            )
            * torch.nan
        )
        out_y = (
            torch.ones([n_years, N_CATCHMENT, SEQ_LENGTH], device=storge_device)
            * torch.nan
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
