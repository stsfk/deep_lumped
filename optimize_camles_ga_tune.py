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

import pygad

import HydroErr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LENGTH = 365 * 2
TARGET_SEQ_LENGTH = 365
BASE_LENGTH = SEQ_LENGTH - TARGET_SEQ_LENGTH

FORCING_DIM = 3
N_CATCHMENTS = 559


# training hyperparameters
use_amp = True
compile_model = False

if compile_model:
    torch.set_float32_matmul_precision("high")

memory_saving = False
if memory_saving:
    storge_device = "cpu"
    computing_device = DEVICE
    VAL_STEPS = 500
else:
    storge_device = DEVICE
    computing_device = DEVICE

dtrain_val = dataloader.Forcing_Data(
    "data/camels_train_val.csv",
    record_length=3652,
    storge_device=storge_device,
    seq_length=SEQ_LENGTH,
    target_seq_length=TARGET_SEQ_LENGTH,
    base_length=BASE_LENGTH,
)

dtrain = dataloader.Forcing_Data(
    "data/camels_train.csv",
    record_length=2922,
    storge_device=storge_device,
    seq_length=SEQ_LENGTH,
    target_seq_length=TARGET_SEQ_LENGTH,
    base_length=BASE_LENGTH,
)

dval = dataloader.Forcing_Data(
    "data/camels_val.csv",
    record_length=1095,
    storge_device=storge_device,
    seq_length=SEQ_LENGTH,
    target_seq_length=TARGET_SEQ_LENGTH,
    base_length=BASE_LENGTH,
)

dtest = dataloader.Forcing_Data(
    "data/camels_test.csv",
    record_length=4383,
    storge_device=storge_device,
    seq_length=SEQ_LENGTH,
    target_seq_length=TARGET_SEQ_LENGTH,
    base_length=BASE_LENGTH,
)

class Objective_builder:
    def __init__(self, x, y, eval_fun):
        self.eval_fun = eval_fun
        self.x = x.contiguous()
        self.y = y.contiguous()

    def eval(self, ga_instance, solution, solution_idx):

        # numpy to torch tensor
        solution = (
            torch.from_numpy(solution)
            .unsqueeze(0)
            .to(dtype=torch.float32)
            .to(computing_device)
        )
        solution = solution.expand(self.x.shape[0], -1)

        # BASE_LENGTH is from global
        pred = decoder.decode(solution, self.x).view(-1).detach().cpu().numpy()

        ob = self.y.view(-1).detach().cpu().numpy()

        gof = self.eval_fun(simulated_array=pred, observed_array=ob)

        return gof

x_batch_train, y_batch_train = dtrain.get_val_batch()
x_batch_val, y_batch_val = dval.get_val_batch()
x_batch_train_val, y_batch_train_val = dtrain_val.get_val_batch()
x_batch_test, y_batch_test = dtest.get_val_batch()


# optimization
for i in range(10):
    print(f"Trial {i} starts")

    embedding = torch.load(
        f"data/final_lstm_embedding{i}.pt", map_location=torch.device("cpu")
    ).to(computing_device)
    decoder = torch.load(
        f"data/final_lstm_decoder{i}.pt", map_location=torch.device("cpu")
    ).to(computing_device)

    embedding.eval()
    decoder.eval()

    # dimension of embedding
    catchment_embeddings = [x.data for x in embedding.parameters()][0]
    LATENT_dim = catchment_embeddings.shape[1]

    # Hyperparameters of GA
    num_generations = 500
    num_parents_mating = 20

    sol_per_pop = 200
    num_genes = LATENT_dim

    # Calculate the minimal and maximal values for each column
    min_vals, _ = catchment_embeddings.min(dim=0)
    max_vals, _ = catchment_embeddings.max(dim=0)

    # Scale the values by 20%, considering the sign
    min_scaled_values = [
        (min_val * 1.2 if min_val < 0 else min_val * 0.8) for min_val in min_vals
    ]
    max_scaled_values = [
        (max_val * 0.8 if max_val < 0 else max_val * 1.2) for max_val in max_vals
    ]

    # Convert the results to lists
    init_range_low = [val.item() for val in min_scaled_values]
    init_range_high = [val.item() for val in max_scaled_values]

    # Print the results
    parent_selection_type = "sss"

    crossover_type = "single_point"

    mutation_type = "random"
    mutation_probability = 0.25

    def fitting_wrapper(selected_catchment):

        # Subsetting training, validation, and test data of selected catchments
        x_train = x_batch_train[:, selected_catchment, :, :].to(computing_device)
        y_train = y_batch_train[:, selected_catchment, :].to(computing_device)

        x_val = x_batch_val[:, selected_catchment, :, :].to(computing_device)
        y_val = y_batch_val[:, selected_catchment, :].to(computing_device)

        x_train_val = x_batch_train_val[:, selected_catchment, :, :].to(
            computing_device
        )
        y_train_val = y_batch_train_val[:, selected_catchment, :].to(computing_device)

        x_test = x_batch_test[:, selected_catchment, :, :].to(computing_device)
        y_test = y_batch_test[:, selected_catchment, :].to(computing_device)

        # Creating evaluation functions
        fn_train = Objective_builder(x_train, y_train, HydroErr.kge_2009)
        fn_val = Objective_builder(x_val, y_val, HydroErr.kge_2009)
        fn_train_val = Objective_builder(x_train_val, y_train_val, HydroErr.kge_2009)
        fn_test = Objective_builder(x_test, y_test, HydroErr.kge_2009)

        # Setting up callback functions for early stop
        early_stopper = training_fun.EarlyStopper(patience=20)
        val_losses = []

        def on_generation(instance):

            solution, solution_fitness, solution_idx = instance.best_solution()
            val_loss = fn_val.eval(instance, solution, solution_idx)

            val_losses.append(val_loss)

            if early_stopper.early_stop(val_loss):
                return "stop"
            else:
                return val_loss

        # Identifying optimal number of generations
        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            fitness_func=fn_train_val.eval,
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            init_range_low=init_range_low,
            init_range_high=init_range_high,
            parent_selection_type=parent_selection_type,
            crossover_type=crossover_type,
            mutation_type=mutation_type,
            mutation_probability=mutation_probability,
            stop_criteria="saturate_5",
            on_generation = on_generation
        )

        ga_instance.run()

        # Evaluating best solution
        # solution = ga_instance.best_solutions[np.argmax(val_losses),:]
        solution, solution_fitness, solution_idx = ga_instance.best_solution()

        return fn_test.eval(ga_instance, solution, 1), solution

    calibrated_KGES = np.ones(N_CATCHMENTS)
    camels_embeddings = np.ones([N_CATCHMENTS, LATENT_dim])

    for j in range(N_CATCHMENTS):
        print(f"Catchment i={j} starts")
        calibrated_KGES[j], camels_embeddings[j, :] = fitting_wrapper(j)
        print(f"fit={calibrated_KGES[j]}")

    np.savetxt(f"data/ga_KGEs_test{i}.csv", calibrated_KGES, delimiter=",")
    np.savetxt(
        f"data/ga_camels_embeddings_test{i}.csv", camels_embeddings, delimiter=","
    )

    print(f"meadian = {np.median(calibrated_KGES)}")
    print(f"mean = {np.mean(calibrated_KGES)}")
