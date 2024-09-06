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

import pygad

import HydroErr

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LENGTH = 365 * 2
TARGET_SEQ_LENGTH = 365
BASE_LENGTH = SEQ_LENGTH - TARGET_SEQ_LENGTH

FORCING_DIM = 3

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

# %%
embedding = torch.load("data/final_lstm_embedding0.pt", map_location=torch.device('cpu')).to(computing_device)
decoder = torch.load("data/final_lstm_decoder0.pt", map_location=torch.device('cpu')).to(computing_device)

embedding.eval()
decoder.eval()

# dimension of embedding
catchment_embeddings=[x.data for x in embedding.parameters()][0]
LATENT_dim = catchment_embeddings.shape[1]

# %%
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

# %%
class Objective_builder_batch:
    def __init__(self, x, y, eval_fun):
        self.eval_fun = eval_fun
        
        self.x = x.contiguous() 
        self.year = x.shape[0] # the long time series is split into x.shape[0] years
        
        self.y = y
    
    def eval(self, ga_instance, solutions, solution_idx):
        
        if len(solutions.shape)==1:
          solutions = np.expand_dims(solutions, axis=0) 
        
        batch_size = solutions.shape[0]
        
        x = self.x.repeat(batch_size, 1, 1).contiguous() # repeat batch_size times
        y = self.y.reshape(-1).contiguous() # combine to a multiple year time series
        
        # numpy to torch tensor
        solutions = torch.from_numpy(solutions).to(dtype=torch.float32).to(computing_device)
        # repeat to match the size of x, which split a long time series into multiple years
        solutions = solutions.repeat_interleave(self.year, dim = 0)
        
        pred = decoder.decode(solutions, x).reshape(batch_size, -1).detach().cpu().numpy()
        ob = y.detach().cpu().numpy()

        gofs = np.ones([batch_size])
        for i in range(batch_size):
          gofs[i] = self.eval_fun(simulated_array=pred[i,:], observed_array=ob)    
        
        return gofs.tolist()

# %%
x_batch_train, y_batch_train = dtrain.get_val_batch()
x_batch_val, y_batch_val = dval.get_val_batch()
x_batch_train_val, y_batch_train_val = dtrain_val.get_val_batch()
x_batch_test, y_batch_test = dtest.get_val_batch()

# %%
# Hyperparameters of GA
num_generations = 500
num_parents_mating = 10

sol_per_pop = 200
num_genes = LATENT_dim

# Calculate the minimal and maximal values for each column
min_vals, _ = catchment_embeddings.min(dim=0)
max_vals, _ = catchment_embeddings.max(dim=0)

# Scale the values by 20%, considering the sign
min_scaled_values = [(min_val * 1.2 if min_val < 0 else min_val * 0.8) for min_val in min_vals]
max_scaled_values = [(max_val * 0.8 if max_val < 0 else max_val * 1.2) for max_val in max_vals]

# Convert the results to lists
init_range_low = [val.item() for val in min_scaled_values]
init_range_high = [val.item() for val in max_scaled_values]

# Print the results
parent_selection_type = "sss"

crossover_type = "single_point"

mutation_type = "random"
mutation_probability = 0.25

# %%
def fitting_wrapper(selected_catchment, batch_size=50):

    # Subsetting training, validation, and test data of selected catchments
    x_train = x_batch_train[:,selected_catchment,:,:].to(computing_device)
    y_train = y_batch_train[:,selected_catchment,:].to(computing_device)

    x_val = x_batch_val[:,selected_catchment,:,:].to(computing_device)
    y_val = y_batch_val[:,selected_catchment,:].to(computing_device)

    x_train_val = x_batch_train_val[:,selected_catchment,:,:].to(computing_device)
    y_train_val = y_batch_train_val[:,selected_catchment,:].to(computing_device)

    x_test = x_batch_test[:,selected_catchment,:,:].to(computing_device)
    y_test = y_batch_test[:,selected_catchment,:].to(computing_device)

    # Creating evaluation functions
    fn_train = Objective_builder_batch(x_train,y_train,HydroErr.kge_2009)
    fn_val = Objective_builder_batch(x_val,y_val,HydroErr.kge_2009)
    fn_train_val = Objective_builder_batch(x_train_val,y_train_val,HydroErr.kge_2009)
    fn_test = Objective_builder_batch(x_test,y_test,HydroErr.kge_2009)

    # Setting up callback functions for early stop
    early_stopper = training_fun.EarlyStopper(patience=20)
    val_losses = []

    def on_generation(instance):
        
        solution, solution_fitness, solution_idx = instance.best_solution()
        val_loss = fn_val.eval(instance, solution, solution_idx)
        val_loss = -np.array(val_loss)[0]
        val_losses.append(val_loss)
        
        if early_stopper.early_stop(val_loss):
            return "stop"
        else:
            return val_loss

    # Identifying optimal number of generations
    ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        fitness_func=fn_train_val.eval,
                        sol_per_pop=sol_per_pop,
                        num_genes=num_genes,
                        init_range_low=init_range_low,
                        init_range_high=init_range_high,
                        parent_selection_type=parent_selection_type,
                        fitness_batch_size = batch_size,
                        crossover_type=crossover_type,
                        mutation_type=mutation_type,
                        mutation_probability = mutation_probability,
                        stop_criteria="saturate_10")

    ga_instance.run()

    # Evaluating best solution
    #solution = ga_instance.best_solutions[np.argmax(val_losses),:]
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    
    torch.cuda.empty_cache()

    return fn_test.eval(ga_instance, solution, 1), solution

# %%
N_CATCHMENTS = 559
calibrated_KGES = np.ones(N_CATCHMENTS)
camels_embeddings = np.ones([N_CATCHMENTS, LATENT_dim])

for i in range(N_CATCHMENTS):
    print(f'i={i} starts')
    calibrated_KGE, camels_embedding = fitting_wrapper(i)
    calibrated_KGES[i], camels_embeddings[i,:]  = np.array(calibrated_KGE)[0], camels_embedding
    print(f'fit={calibrated_KGES[i]}')

# %%
np.savetxt("data/out_of_sample_results/camels_KGEs.csv", calibrated_KGES, delimiter=",")
np.savetxt("data/out_of_sample_results/camels_embeddings.csv", camels_embeddings, delimiter=",")