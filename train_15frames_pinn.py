"""
    Model training
    Using the dataset droplet_sim_2prjs_1exp_15frames_stride5.h5
    With/without PINN (random dice mode)
    Angle: 0, 23.8
"""

#%%
import numpy as np
np.random.seed(27)
import os
import time
import torch
import torch.nn as nn
from models.options import ParamOptions
from models.trainer import TrainModel
from models.DatasetTest import TestDataset

#%%
opt = ParamOptions().parse()
#####################
# for debug only
opt.in_channel = 2
opt.d_out = 6  # at, phase, vx, vy, vz, p
opt.num_coarse = 128  # default 256
opt.use_time = True
opt.n_views = 1   # default 4
#opt.num_val_views = 2   # default 2
opt.radiance_field_noise_std = 1e-5  # default 1e-9, previous 0
opt.noise_decay_steps = 100   # default 100
opt.noise_decay_factor = 0.9  # default 0.001
opt.activation = nn.SiLU()
opt.input_view_idx = "random"  # default: None
opt.gpu_id = 0    # default 0
opt.use_camera_space_pnts = False   # default True, for the version with encoder, it has to be true
opt.enable_encoding_fn_time = True # default True
opt.enable_encoding_fn_xyz = True  # default True
opt.num_encoding_fn_xyz = 10        # default 10
opt.num_encoding_fn_time = 6       # default 6

opt.load_path = "../droplet_reproducible_sim/droplet_sim_2prjs_1exp_15frames_stride5.h5"
opt.batch_size = 1    # default 2
opt.use_gradient_sampling = True   # if False, get_ray_sample_new is implemented
opt.num_random_rays = 256  # default 1024，previous 256
opt.chunksize = 512   # default 4096, previous 1024
opt.no_shuffle_input = True  # default False, should set as true for new batch sampler

opt.lr = 2e-4         # 8e-4
opt.lr_d = 2e-4       # lr for discriminator
opt.lr_decay_steps = 800    # 200
opt.lr_decay_factor = 0.9  # default 0.001

opt.print_loss_freq_iter = 10   # default 10
opt.save_model_freq_epoch = 800 # default 10
opt.save_plot_freq_epoch = 800  # default 5, no longer relevant
opt.num_epochs = 9600          # default 1000

opt.start_time = 1.493138*0/199           # default 0
opt.end_time = 1.493138*70/199    # default 1
opt.pde_start_time = 1.493138*0/199    # default 0
opt.pde_end_time = 1.493138*75/199    #*95/199        # default 1

opt.allow_parallel = True # default True, for the version with encoder or n_view>1, it has to be false
opt.use_encoder = False    # default True
opt.use_camera_pixel = False # default False

opt.enable_v_p = False    # default False
opt.lambda_u = 0.0
opt.lambda_p = 0.0
opt.lambda_psi = 0.0      # newly added

opt.epoch_vp_sampling = 1  # default 5
opt.epoch_pde_sampling = 1 # default 5

opt.enable_pinn = True  # default True
#opt.lambda_mse = 1e-4
#opt.lambda_pde = 1e-2   # default 0.0
opt.lambda_generator = 1.0
opt.pinn_sampling_mode = 'dynamic'   # default none

opt.random_proj = False   # default False
opt.num_pts_ratio = 6      # default 1

opt.random_time_stamp = False # default False

#####################
model = TrainModel(opt)
model.init_model()
destination = f"{opt.run_path}/{opt.run_name}"
print(destination)
if opt.load_pretrain:
    model.load_trained_models(opt.model_path, opt.load_epoch)  # Pretrain
model.save_parameters()
num_input = 2
if opt.use_time:
    num_input += 1
for epoch in range(opt.num_epochs):
    now = time.time()
    model.epoch = epoch
    if epoch % 1 == 0:
        dice = np.array([0,1])
        flag = np.random.choice(dice,size=(1),p=[0.5,0.5])  
    if (epoch >= 800 and flag == 1):
        print('   Random_proj.')
        model.random_proj = True
        model.lambda_mse = 1e-4
        model.lambda_pde = 1       #1e-1      #1e-2   
        model.opt.random_time_stamp = True
        model.modify_time_stamp()
    else:
        print('   Fixed_proj.')
        model.random_proj = False
        model.lambda_mse = 1e4
        model.lambda_pde = 0  #1e-1 #1.0
        model.opt.random_time_stamp = False
        model.modify_time_stamp()        
        
    model.update_parameters(epoch)
    for i, train_data in enumerate(model.train_loader):
        model.set_input(train_data)
        model.optimization()   
        if i % opt.print_loss_freq_iter == opt.print_loss_freq_iter - 1:
            losses = model.get_current_losses()
            model.print_current_losses(epoch=epoch, iters=i, losses=losses)
            
    if (
        epoch == 0
        or epoch % opt.save_model_freq_epoch == opt.save_model_freq_epoch - 1
        ):
        model.save_models(epoch)
    now = time.time() - now
    print(
        f"training time for {opt.run_name} epoch {epoch+1}: {now//60} min {(now - now//60*60):.2f} s"
        )
print(destination)
