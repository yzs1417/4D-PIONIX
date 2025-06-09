#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Generate 3D models fooor a sequence of objects (npy)

"""

#%%
import torch
import torch.nn as nn
from itertools import combinations
import os
import time
from models.eval_options import ParamOptions
from models.trainer import TrainModel

# for validation
#import tomopy
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage
#from pyvtk import *

# import dxchange
import h5py as h5
from skimage.transform import rotate, resize
from skimage.metrics import structural_similarity
from scipy.signal import medfilt2d


class MultiEvalOptions(ParamOptions):
    def __init__(self):
        super(ParamOptions, self).__init__()
        self.initialized = False

    def initialize(self, parser):
        parser = ParamOptions.initialize(self, parser)
        parser.add_argument(
            "--write_path", type=str, default='./', help="Directory to save the output."
        )
        parser.add_argument(
            "--num_frame", type=int, default=1, help="Number of frames to be evaluated."
        )
        self.initialized = True
        return parser

#%%
if __name__ == "__main__":
    opt = MultiEvalOptions().parse()
    #####################
    # for debug only
    opt.in_channel = 2
    opt.d_out = 6
    opt.load_path = "../droplet_reproducible_sim/droplet_sim_2prjs_1exp_15to75frames.h5"   # modify filename and path accordingly
    #opt.load_path = "../droplet_reproducible_sim/droplet_sim_2prjs_1exp_75frames.h5"      # modify filename and path accordingly
    opt.model_path = './results/Apr11_11_05'        # modify filename and path accordingly
    opt.write_path = './results/Apr11_11_05/eval_15to75frames_9600epoch_val_0_0'  # modify filename and path accordingly   #t_r_check_
    opt.load_epoch = 9600
    
    opt.chunksize = 4096
    opt.num_coarse = 128  # default 256
    
    opt.activation = nn.SiLU()
    opt.use_time = True
    opt.n_views = 1   # default 4
    opt.val_view_idx = [0]   # previous [0,1] default '1,3,5,7'
    opt.val_idx = 0   # default 9
    opt.num_frame = 75 # default 1
    opt.start_time = 0            # default 0
    opt.end_time = 1.493138*74/199   # default 1
    opt.gpu_id = 0    # default 0
    
    opt.use_camera_space_pnts = False   # default True
    opt.enable_pinn = False  # for eval, always set false
    opt.use_camera_pixel = False
    opt.allow_parallel = True # default True
    opt.use_encoder = False  # default True
    
    opt.enable_encoding_fn_time = True # default True
    opt.enable_encoding_fn_xyz = True  # default True
    opt.num_encoding_fn_xyz = 10
    opt.num_encoding_fn_time = 6       # default 6
    
    opt.enable_v_p = False    # default False
    opt.random_proj = False    # default False
    #####################
    num_frame = opt.num_frame
    write_path = opt.write_path
    npy_path = f"{write_path}/npy"
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)
    num_input = 3 if opt.use_time else 2

    for obj_idx in range(0,num_frame):
        print(f"\n{'*'*15} {obj_idx} {'*'*15}\n")
        now = time.time()
        opt.test_obj_idx = obj_idx
        model = TrainModel(opt)
        model.is_train = False
        model.generate_3D = True
        model.init_model()
        model.load_trained_models(opt.model_path, opt.load_epoch)
        attenuation_path = f"{npy_path}/attenuation_{obj_idx}"
        phase_path = f"{npy_path}/phase_{obj_idx}"
        model.attenuation_path = attenuation_path
        model.phase_path = phase_path
        
        with torch.no_grad():
            model.reshaped_3d_att = []
            model.reshaped_3d_phase = []
            for k, test_data in enumerate(model.test_loader):
                if k == 0:
                    model.set_input(test_data[:num_input])
                    model.validation()
                    model.visual_iter(0, k, test_data[num_input].item())
            now = time.time() - now
            print(f"validation time: {now//60} min {(now - now//60*60):.3f} s")
            to_save_att = np.squeeze(np.array(model.reshaped_3d_att))
            np.save(model.attenuation_path,to_save_att)
# %%
