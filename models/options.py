"""
    Training options 
"""

import os
from datetime import datetime
import argparse
import torch.nn as nn

class ParamOptions:
    """This class defines options used during training time."""

    def __init__(self):
        self.initialized = False
        self.time = datetime.now()
        self.cwd = os.getcwd()

    def initialize(self, parser):
        parser.add_argument(
            "--in_channel",
            type=int,
            default=2,
            help="Number of input channels. Both phase and attenuation channels are used by default.",
        )  # For the current implementation, d_out=in_channel. Remember to change d_out when changing in_channel.
        parser.add_argument(
            "--load_path",
            "-p",
            type=str,
            default="data.npy",
            help="Path to the training file, the customized dataset use npy files",
        )
        parser.add_argument(
            "--run_path",
            type=str,
            default="results",
            help="path to save results",
        )
        parser.add_argument(
            "--gpu_id",
            type=int,
            default=0,
            help="GPU ID. Currently only one gpu is supported",
        )
        parser.add_argument(
            "--test_obj_idx",
            type=int,
            default=0,   #22
            help="Index of test object",
        )
        # parser.add_argument('--run_path', type=str, default = F'{self.cwd}/results/fig', help='path to save results')
        parser.add_argument(
            "--run_name",
            "-e",
            type=str,
            default=self.time.strftime("%b%d_%H_%M"),
            help="Run name",
        )
        parser.add_argument(
            "--n_views",
            "-n",
            type=int,
            default=4,
            help="Number of views used in each training iteration",
        )
        parser.add_argument(
            "--batch_size",
            "-b",
            type=int,
            default=2,
            help="Number of objects trained in each epoch",
        )
        parser.add_argument(
            "--lr", type=float, default=0.005, help="Initial learning rate"
        )
        parser.add_argument(
            "--lr_d",
            type=float,
            default=0.001,
            help="Initial learning rate for the discriminator",
        )
        parser.add_argument(
            "--lr_decay_steps",
            type=int,
            default=500,
            help="Decay the learning rate after every certain epochs",
        )
        parser.add_argument(
            "--lr_decay_factor",
            type=float,
            default=0.001,
            help="Decay factor of the learning rate. No lr decay if set to 1.",
        )
        parser.add_argument(
            "--beta1", type=float, default=0.5, help="Momentum term of Adam optimizer"
        )
        parser.add_argument(
            "--num_epochs", type=int, default=1000, help="Total number of epochs"
        )
        parser.add_argument(
            "--num_encoder_epochs", type=int, default=10000, help="Total number of epochs for encoder, after that, the encoder is fixed."
        )        
        parser.add_argument(
            "--print_loss_freq_iter", type=int, default=10, help="Print loss frequency"
        )
        parser.add_argument(
            "--save_model_start_epoch",
            type=int,
            default=30,
            help="Do not save model for the beginning epochs",
        )
        parser.add_argument(
            "--save_model_freq_epoch", type=int, default=10, help="Save model frequency"
        )
        parser.add_argument(
            "--num_val_views",
            "-v",
            type=int,
            default=2,
            help="Number of validation views. If you want to use all views for the training, leave it 0, and the validation will be generated from a random view",
        )  # This is the views that are avaiable but not used for the training, not the number of constraints.
        parser.add_argument(
            "--use_gradient_sampling",
            action="store_true",
            default=True,
            help="Whether or not to use gradient sampling to sample the rays. Set False to use random sampling",
        )
        parser.add_argument(
            "--gradient_sampling_channel",
            type=int,
            default=0,
            help="Specify which channel is used for calculating gradients: 0/1. 0 for attenuation channel, 1 for phase channel (depending on the dataset). Only needed when in_channel is not 1",
        )
        parser.add_argument(
            "--num_random_rays", type=int, default=2048, help="Number of rays"
        )
        parser.add_argument(
            "--enable_encoding_fn_xyz",
            type=bool,
            default=True,
            help="Whether or not to implement positional encoding for xyz.",
        )        
        parser.add_argument(
            "--num_encoding_fn_xyz",
            type=int,
            default=10,
            help="Degree of positional encoding for xyz.",
        )
        parser.add_argument(
            "--include_input_xyz",
            action="store_true",
            default=True,
            help="Include xyz in the mlp input",
        )
        parser.add_argument(
            "--log_sampling_xyz",
            action="store_true",
            default=True,
            help="Use log sampling or linear sampling. By default log sampling is used.",
        )
        parser.add_argument(
            "--num_coarse",
            type=int,
            default=256,
            help="Number of depth samples per ray for the coarse network",
        )
        parser.add_argument(
            "--lindisp",
            action="store_true",
            help="Sample linearly in disparity space, as opposed to in depth space.",
        )
        parser.add_argument("--z_near", type=float, default=0.5, help="Near bound")
        parser.add_argument("--z_far", type=float, default=1.5, help="Far bound")
        parser.add_argument(
            "--clip_max",
            type=float,
            default=1.0,
            help="maximum value for the gradient clipping, set to 0 if do not want to use gradient clipping.",
        )
        parser.add_argument(
            "--lambda_mse", type=float, default=10000.0, help="Weight for the MSE loss"
        )
        parser.add_argument(
            "--chunksize",
            type=int,
            default=4096,
            help="Used for get_minibatch. For the current implementation this need to be divisible with no remainder by H*W",
        )  # For multi view only works when bigger than (n_views x num_random_rays)
        parser.add_argument(
            "--perturb",
            action="store_true",
            default=True,
            help="Whether or not to perturb the sampled depth values.",
        )
        parser.add_argument(
            "--radiance_field_noise_std",
            type=float,
            default=1e-9,
            help="Noise level for the radiance field",
        )
        parser.add_argument(
            "--noise_decay_steps",
            type=int,
            default=100,
            help="Decay the noise after every certain epochs",
        )
        parser.add_argument(
            "--noise_decay_factor",
            type=float,
            default=0.001,
            help="Decay factor of the noise level. No noise decay if set to 1",
        )
        parser.add_argument(
            "--num_fine",
            type=int,
            default=0,
            help="Number of depth samples per ray for the fine network. We have not tested the fine network, but feel that it would be good to keep this option here.",
        )
        parser.add_argument(
            "--use_camera_pixel",
            action="store_true",
            default=False,
            help="Whether or not to use the camera pixels if not using encoder.",
        )        
        parser.add_argument(
            "--use_encoder",
            action="store_true",
            default=True,
            help="Whether or not to use the encoder. We have not carefully tested the no encoder option.",
        )
        parser.add_argument(
            "--encoder_pretrain",
            action="store_true",
            default=True,
            help="Pretrain the encoder (suggested).",
        )
        parser.add_argument(
            "--encoder_num_layers",
            type=int,
            default=3,
            help="Number of layers in the encoder.",
        )
        parser.add_argument(
            "--use_first_pool",
            action="store_true",
            default=True,
            help="Use the first pooling layer.",
        )
        parser.add_argument(
            "--allow_parallel",
            default=True,
            help="Whether or not to use parallel (multi-gpu).",
        )
        parser.add_argument(
            "--index_interp", type=str, default="bilinear", help="Type of interpolation"
        )
        parser.add_argument(
            "--index_padding", type=str, default="border", help="Type of padding"
        )
        parser.add_argument(
            "--use_camera_space_pnts",
            action="store_true",
            default=True,
            help="Use camera space points.",
        )
        parser.add_argument(
            "--backbone", type=str, default="resnet34", help="Feature extractor"
        )
        parser.add_argument(
            "--n_blocks", type=int, default=5, help="Number of MLP blocks."
        )
        parser.add_argument(
            "--d_hidden", type=int, default=128, help="Number of hidden layers in MLP."
        )
        parser.add_argument(
            "--d_out", type=int, default=2, help="Number of MLP output layers."
        )
        parser.add_argument(
            "--activation",
            default=nn.ReLU(),
            help="Activation function of Resnetfc",
        )       # New argument
        parser.add_argument(
            "--combine_layer",
            type=int,
            default=3,
            help="Combine after the third layer by average.",
        )
        parser.add_argument(
            "--combine_type",
            type=str,
            default="average",
            help="Feature combination type: average or max",
        )
        parser.add_argument(
            "--save_plot_freq_epoch",
            type=int,
            default=5,
            help="Save plots after every several epochs",
        )
        parser.add_argument(
            "--load_pretrain", action="store_true", help="Load pretrained model."
        )
        parser.add_argument(
            "--model_path",
            type=str,
            default="pretrain",
            help="Path to the pretrain run directory",
        )
        parser.add_argument(
            "--load_epoch",
            type=int,
            default=300,
            help="Load epochs if load_pretrain is used",
        )
        parser.add_argument(
            "--use_time",
            action="store_true",
            default=False,
            help="Whether or not to include time",
        )
        parser.add_argument(
            "--start_time",
            default=0,
            help="Time stamp of the first frame",
        )
        parser.add_argument(
            "--end_time",
            default=1,
            help="Time stamp of the last frame",
        )
        parser.add_argument(
            "--pde_start_time",
            default=0,
            help="Time stamp when pde starts",
        )        
        parser.add_argument(
            "--pde_end_time",
            default=1,
            help="Time stamp when pde ends",
        )        
        parser.add_argument(
            "--eval_H",
            type=int,
            default=None,
            help="If specified, use this as the height for the reconstruction",
        )
        parser.add_argument(
            "--eval_W",
            type=int,
            default=None,
            help="If specified, use this as the width for the reconstruction",
        )
        parser.add_argument(
            "--enable_encoding_fn_time",
            type=bool,
            default=True,
            help="Whether or not to implement positional encoding for time.",
        )
        parser.add_argument(
            "--num_encoding_fn_time",
            type=int,
            default=6,
            help="Degree of positional encoding for time",
        )
        parser.add_argument(
            "--include_input_time",
            action="store_true",
            default=True,
            help="Include time in the mlp input",
        )
        parser.add_argument(
            "--log_sampling_time",
            action="store_true",
            default=True,
            help="Use log sampling or linear sampling. By default log sampling is used.",
        )
        # Frame Variation options
        parser.add_argument(
            "--no_shuffle_input",
            action="store_true",
            default=False,
            help="Whether to shuffle the training dataset.",
        )
        parser.add_argument(
            "--lambda_fv",
            type=float,
            default=0.0,
            help="Weight for the frame variation loss",
        )
        parser.add_argument(
            "--enable_pinn",
            default=True,
            help="Whether or not to include PINNs.",
        )
        parser.add_argument(
            "--enable_v_p",
            default=False,
            help="Whether or not to include supervised term for v and p fields.",
        )

        parser.add_argument(
            "--enable_psi",
            default=False,
            help="Whether or not to include supervised term (initial condition) for psi fields.",
        )        
                
        
        parser.add_argument(
            "--pinn_sampling_mode",
            default="none",
            help="Method for sampling pinn points (Valid modes: dynamic). Input none if you want to use the same points as ONIX",
        )
        parser.add_argument(
            "--epoch_pde_sampling",
            type=int,
            default=5,
            help="How often do we resample pde points",
        )   
        parser.add_argument(
            "--epoch_vp_sampling",
            type=int,
            default=5,
            help="How often do we resample v,p field points",
        )         
        
        parser.add_argument(
            "--num_pts_ratio",
            default=1,
            help="ratio between number of points for ONIX and that for PDE loss",
        )        
        parser.add_argument(
            "--lambda_pde",
            type=float,
            default=0.0,
            help="Weight for the pde loss",
        )
        
        parser.add_argument(
            "--lambda_u",
            type=float,
            default=0.0,
            help="Weight for the loss of velocity field",
        )

        parser.add_argument(
            "--lambda_p",
            type=float,
            default=0.0,
            help="Weight for the loss of pressure field",
        )        

        parser.add_argument(
            "--lambda_psi",
            type=float,
            default=0.0,
            help="Weight for the loss of psi field (supervised term)",
        )         
        
        parser.add_argument(
            "--share_gradient",
            action="store_true",
            default=False,
            help="Share gradient over adjecant frames (used for fv)",
        )
        
        parser.add_argument(
            "--input_view_idx",
            type=list,
            default=None,
            help="Input view index for the constraints. Need to match number of views.",
        )
        
        parser.add_argument(
            "--val_view_idx",
            type=str,
            nargs="+",
            default="1, 3, 5, 7",
            help="View index for the constraints. Need to match number of views.",
        )
        parser.add_argument(
            "--val_idx",
            type=int,
            default=9,
            help="Index of view angle for the validation. Set to -1 if you want to validate using a random view.",  # todo: Only works for specific senerios. Implement for convenience.
        )
        
        parser.add_argument(
            "--random_proj",
            action="store_true",
            default=False,
            help="Whether or not to use random projection angles in the training. This is mainly used when there are few views available, and the network is supervised by the GAN loss.",
        )        
        parser.add_argument(
            "--random_time_stamp",
            default=False,
            help="Whether or not to use random t during training.",
        )         
        
        parser.add_argument(
            "-h",
            "--help",
            action="help",
            default=argparse.SUPPRESS,
            help="Invoke help functions",
        )
        parser.add_argument(
            "-f", "--fff", 
            help="a dummy argument to fool ipython", 
            default="1"
        )
        parser.add_argument(
            "-i", "--ip", 
            help="a dummy argument to fool ipython", 
            default="1"
        )
        parser.add_argument(
            "-s", "--stdin", 
            help="a dummy argument to fool ipython", 
            default="1"
        )
        parser.add_argument(
            "-c", "--control", 
            help="a dummy argument to fool ipython", 
            default="1"
        )
        parser.add_argument(
            "--hb", 
            help="a dummy argument to fool ipython", 
            default="1"
        )
        parser.add_argument(
            "-S", "--Session.signature_scheme", 
            help="a dummy argument to fool ipython", 
            default="1"
        )
        parser.add_argument(
            "-K", "--Session.key", 
            help="a dummy argument to fool ipython", 
            default="1"
        )
        parser.add_argument(
            "--shell", 
            help="a dummy argument to fool ipython", 
            default="1"
        )
        parser.add_argument(
            "--transport", 
            help="a dummy argument to fool ipython", 
            default="1"
        )
        parser.add_argument(
            "--iopub", 
            help="a dummy argument to fool ipython", 
            default="1"
        )
        self.initialized = True
        return parser

    def list_view_idx(self, opt, arg):
        d = vars(opt)
        if arg in d.keys():
            d[arg] = [int(s.strip()) for s in d[arg][0].split(",")]

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False
            )
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        """Parse our options"""
        opt = self.gather_options()
        self.opt = opt
        self.list_view_idx(opt=self.opt, arg="val_view_idx")
        return self.opt
