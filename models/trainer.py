"""
    Trainer for ONIX
"""

import sys
import os
import itertools
from random import randint
from collections import OrderedDict
from abc import ABC
import numpy as np
try:
    import cupy as cp 
except:
    print('Cannot import cupy, only numpy is used...')
import h5py
import torch
import torch.nn.functional as F
from models.mat_calc import get_world_mat
from models.encoder import SpatialEncoder
from models.discriminator import Discriminator
from models.resnetfc import ResnetFC
from models.DatasetCustom import CustomDataset, MyRandomBatchSampler
from models.DatasetTest import TestDataset
from models.sampler import FlexGridRaySampler, ImgToPatch
from models.utils import (
    init_weights,
    get_embedding_function,
    get_minibatches,
    get_ray_bundle,
    repeat_interleave,
    sample_pdf,
    cumprod_exclusive,
    save_tensor_plot,
    grad,
    divergence,
    equation_droplet,
    set_requires_grad,
    nrmse_fn
)


class TrainModel(ABC):
    """ONIX model"""

    def __init__(self, opt):
        self.num_objs = 0
        self.opt = opt
        self.dtype = torch.float
        self.device = torch.device(
            f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        #self.device = 'cpu'
        self.is_train = True
        self.loss_names = ["coarse", "fine", "D_coarse", "G_coarse","pde","u","p","psi","G_total"]
        self.generate_3D = False  # for eval
        #self.attenuation_path = f"{self.opt.run_path}/{self.opt.run_name}/attenuation"
        #self.phase_path = f"{self.opt.run_path}/{self.opt.run_name}/phase"
        self.reshaped_3d_att = []
        self.reshaped_3d_phase = []
        self.seed_torch = 1
        self.seed_rand = 18
        torch.manual_seed(self.seed_torch)
        np.random.seed(self.seed_rand)
        self.val_names = ["val"]

        if opt.in_channel == 1:
            self.with_phase = False
        elif opt.in_channel == 2:
            self.with_phase = True
        else:
            raise ValueError("Only one or two input channels are supported yet!")
        
        if True:  # todo: add option
            self.sampler = FlexGridRaySampler(
                N_samples=self.opt.num_random_rays, orthographic=True
            )
        self.random_proj = opt.random_proj
        

    def save_parameters(self):
        with open(self.save_log, "w+") as f:
            print(self.opt.__dict__, file=f)

    def load_data(self):
        print("start loading data....")
        if self.is_train:
            self.train_dataset = CustomDataset(self.opt)
            self.train_loader = torch.utils.data.DataLoader(
                dataset=self.train_dataset,
                batch_size=self.opt.batch_size,
                shuffle=not self.opt.no_shuffle_input,  # TODO: set to false for 4d
                sampler=MyRandomBatchSampler(self.train_dataset, batch_size=self.opt.batch_size)
                if self.opt.no_shuffle_input
                else None,
                num_workers=8,
                pin_memory=True,
            )
        else:
            self.train_loader = []
        self.test_dataset = TestDataset(self.opt)
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        print("finish loading data")

    def modify_time_stamp(self):
        time_seq = torch.linspace(self.opt.start_time,self.opt.end_time,self.train_dataset.images_pool.shape[0])
        if self.opt.random_time_stamp:
            print("   Using random time stamps....")
            time_diff = time_seq[1] - time_seq[0]
            random_seq = torch.rand(self.train_dataset.images_pool.shape[0]) - 0.5   # -0.5 to 0.5
            random_seq[0] = torch.abs(random_seq[0]) # avoid negative t
            time_seq = time_seq + random_seq*time_diff
        else:
            print('   Using fixed time stamps....')
        print(time_seq)
        self.train_dataset.images_time = time_seq
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.opt.batch_size,
            shuffle=not self.opt.no_shuffle_input,  # TODO: set to false for 4d
            sampler=MyRandomBatchSampler(self.train_dataset, batch_size=self.opt.batch_size)
            if self.opt.no_shuffle_input
            else None,
            num_workers=8,
            pin_memory=True,
        )
            
    def init_in_out(self):
        d_in = self.opt.num_encoding_fn_xyz * 6 if self.opt.enable_encoding_fn_xyz else 0
        if self.opt.include_input_xyz:
            d_in += 3
        if self.opt.use_time:
            if self.opt.enable_encoding_fn_time:
                d_in += self.opt.num_encoding_fn_time * 2
            if self.opt.include_input_time:
                d_in += 1
        self.d_latent = self.latent_size if self.opt.use_encoder else 0
        if not self.opt.use_encoder:
            if self.opt.use_camera_pixel:
                try:
                    d_in += self.train_dataset.total_views # camera info
                except:
                    d_in += self.test_dataset.NV # camera info
        self.d_in = d_in
        self.d_out = self.opt.d_out

    def init_model(self):
        if self.is_train:
            self.lambda_mse = self.opt.lambda_mse
            self.lambda_pde = self.opt.lambda_pde
            self.lambda_u = self.opt.lambda_u
            self.lambda_p = self.opt.lambda_p
            self.lambda_psi = self.opt.lambda_psi # newly added
            self.clip_max = self.opt.clip_max
        if not self.generate_3D:
            self.save_run = f"{self.opt.run_path}/{self.opt.run_name}"
            self.save_log = f"{self.save_run}/log.txt"
            self.save_val = f"{self.save_run}/val.txt"
            self.create_dir_if_not_exist(self.save_run)
        else:
            save_3d_path = f"{self.opt.write_path}/eval"
            self.save_run = save_3d_path
            self.create_dir_if_not_exist(save_3d_path)

        if self.opt.use_encoder:
            self.encoder = SpatialEncoder(
                in_channel=self.opt.in_channel,
                backbone=self.opt.backbone,
                pretrained=self.opt.encoder_pretrain,
                num_layers=self.opt.encoder_num_layers,
                use_first_pool=self.opt.use_first_pool,
            )
            self.latent_size = self.encoder.latent_size
            self.encoder.to(self.device)

        self.load_data()
        if self.is_train:
            self.total_step = len(self.train_loader)
        
        self.init_in_out()
        self.model_coarse = ResnetFC(
            d_in=self.d_in,
            d_latent=self.d_latent,
            d_out=self.opt.d_out,
            n_blocks=self.opt.n_blocks,
            d_hidden=self.opt.d_hidden,
            combine_layer=self.opt.combine_layer,
            combine_type=self.opt.combine_type,
            active=self.opt.activation,
        )
        self.model_coarse.to(self.device)
        if (self.opt.allow_parallel) & (self.opt.gpu_id == 0) & (torch.cuda.device_count()>1):
            #torch.distributed.init_process_group(backend='nccl',world_size=torch.cuda.device_count())
            #torch.nn.parallel.DistributedDataParallel(self.model_coarse)
            self.model_coarse = torch.nn.DataParallel(self.model_coarse)
            print('Multi-GPU mode...')
        else:
            print(f'Single-GPU mode, GPU no. {self.opt.gpu_id}...')
        init_weights(self.model_coarse, "normal", init_gain=0.02)
        trainable_parameters = list(self.model_coarse.parameters())
        if self.opt.use_encoder:
            trainable_parameters += list(self.encoder.parameters())

        if self.opt.num_fine > 0:
            self.model_fine = ResnetFC(
                d_in=self.d_in,
                d_latent=self.d_latent,
                d_out=self.opt.d_out,
                n_blocks=self.opt.n_blocks,
                d_hidden=self.opt.d_hidden,
                combine_layer=self.opt.combine_layer,
                combine_type=self.opt.combine_type,
                active=self.opt.activation,
            )
            self.model_fine.to(self.device)
            init_weights(self.model_fine, "normal", init_gain=0.02)
            trainable_parameters += list(self.model_fine.parameters())
        else:
            self.model_fine = None

        self.init_embedding()
        if self.is_train:
            self.optimizer = torch.optim.Adam(
                trainable_parameters, lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
            )
            self.netD = Discriminator(
                nc=self.opt.in_channel, imsize=int(np.sqrt(self.opt.num_random_rays))
            ).to(self.device)
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=self.opt.lr_d, betas=(self.opt.beta1, 0.999)
            )
        
        if self.opt.enable_v_p:    
            self.init_u_p_psi()  

    def init_embedding(self):
        self.encode_position_fn = get_embedding_function(
            num_encoding_functions=self.opt.num_encoding_fn_xyz,
            include_input=self.opt.include_input_xyz,
            log_sampling=self.opt.log_sampling_xyz,
        )
        self.encode_time_fn = None
        if self.opt.use_time:
            self.encode_time_fn = get_embedding_function(
                num_encoding_functions=self.opt.num_encoding_fn_time,
                include_input=self.opt.include_input_time,
                log_sampling=self.opt.log_sampling_time,
            )

    def run_onix(
        self,
        mode="train",
    ):
        batches = get_minibatches(self.rays, chunksize=self.opt.chunksize)
        
        if self.is_train:
            self.pde_loss = 0   # do not evaluate pde_loss for validation
            self.u_loss = 0
            self.p_loss = 0
            self.psi_loss = 0
        
        pred = [self.predict_with_rendering(batch) for (i, batch) in enumerate(batches)]
        
        if self.is_train:
            self.pde_loss /= len(batches)
            self.u_loss /= len(batches)
            self.p_loss /= len(batches)
            self.psi_loss /= len(batches)
            
        synthesized_images = list(zip(*pred))
        synthesized_images = [
            torch.cat(image, dim=-1) if image[0] is not None else (None)
            for image in synthesized_images
        ]
        if mode == "validation":
            restore_shapes = [
                self.ray_directions.shape[:-1],
                self.ray_directions.shape[:-1],
                self.ray_directions.shape[:-1],
            ]
            if self.model_fine:
                restore_shapes += restore_shapes
            synthesized_images = [
                image.view(shape) if image is not None else None
                for (image, shape) in zip(synthesized_images, restore_shapes)
            ]
            if self.model_fine:
                return tuple(synthesized_images)
            else:
                if self.with_phase:
                    return tuple(synthesized_images + [None])
                else:
                    return tuple(synthesized_images)

        return tuple(synthesized_images)

    def set_input(self, input_data):
        """images: [batch_size, n_views, (2), H, W]; poses: [batch_size, n_views, 4, 4]"""
        self.all_images = (
            input_data[0].to(self.device, dtype=self.dtype)
        )  # 2, 8, 2, 256, 256
        self.all_poses = (
            input_data[1].to(self.device, dtype=self.dtype)
        )  # 2, 8, 4, 4
        if self.opt.use_time:
            self.all_times = (
                input_data[2].to(self.device, dtype=self.dtype) #.squeeze(0) do not squeeze, in case batch size is 1
            )  # 2
            #print(f"self.all_times : {self.all_times}")
        self.SB, self.NV, self.CN, self.H, self.W = self.all_images.shape
        # print(f'images: {self.all_images.all_imagesshape}, poses: {self.all_poses.shape}')

    def get_encode_views(self):
        """Generate"""
        
        if self.opt.n_views > self.NV:
            raise ValueError(
                f"Selected number of input should be less than the number of available input! Only {self.NV} inputs are available!"
            )
        # Choose encode views
        try:
            self.r_index = torch.tensor(self.opt.input_view_idx)
        except:
            self.r_index = torch.randperm(self.NV)[: self.opt.n_views].sort()[
                0
            ]  # select n views from the input
        
        #print(self.r_index)    # Check which views are selected
        self.encode_images = self.all_images[:, self.r_index]
        self.encode_pose = self.all_poses[:, self.r_index]
        self.encode_pose = self.encode_pose.reshape(-1, 4, 4)

    def generate_random_poses(self):
        """Generate random poses for the single-projection training"""
        ProjAngles = np.array(np.random.rand(self.all_images.shape[1]) * 180).reshape(
            -1, 1
        )
        #print(ProjAngles)
        theta_x = 0
        theta_z = 0
        theta_y = ProjAngles
        world_mat = []
        for i in range(ProjAngles.shape[0]):
            world_mat.append(get_world_mat(theta_x, theta_y[i], theta_z))
        world_matrix = np.asarray(world_mat).astype(float)
        return torch.from_numpy(world_matrix).float()  # [batch_size,4,4]

    def get_ray_function(self):
        """Get ray origin and ray direction"""

        if not self.random_proj:
            pose_target = self.all_poses[..., :3, :4].to(self.device)  # [B, 8, 3, 4]
        else:
            pose_target = self.generate_random_poses().unsqueeze(
                0
            )  # [B, num_views, 4, 4]

        self.ray_origins, self.ray_directions = get_ray_bundle(
            self.H,
            self.W,
            pose_target[0],  # all batch objects have the same projection angles
        )

    def sample_grid(self, data, grid):
        tmp = F.grid_sample(
            data,
            grid,  # B,N,1,2
            align_corners=True,
            mode=self.opt.index_interp,
            padding_mode=self.opt.index_padding,
        )
        return tmp[:, :, :, 0]

    def get_ray_sample_grid(self):
        self.get_encode_views()
        self.get_ray_function()
        self.ray_origins = self.ray_origins.permute(0, 3, 1, 2).to(
            self.device
        )  # B,H,W,3 -> B,3,H,W
        self.ray_directions = self.ray_directions.permute(0, 3, 1, 2).to(
            self.device
        )  # B,H,W,3 -> B,3,H,W
        n_ray = int(np.sqrt(self.opt.num_random_rays))

        # sample =FlexGridRaySampler(N_samples=self.opt.num_random_rays).sample_rays(n_ray,n_ray) # 32, 32, 2
        sample = self.sampler.sample_rays(n_ray, n_ray)  # 32, 32, 2
        sample = sample.reshape(-1, 2)
        uv = torch.repeat_interleave(
            sample[None, :, None, :], self.all_images.shape[0], dim=0
        ).to(self.device)
        
        uv_r = torch.repeat_interleave(sample[None, :, None, :], self.NV, dim=0).to(
                self.device
            )
            
        self.ray_origins = (
            self.sample_grid(self.ray_origins, uv_r).permute(0, 2, 1).reshape(-1, 3)
        )  # Bx1024,3
        self.ray_directions = (
            self.sample_grid(self.ray_directions, uv_r).permute(0, 2, 1).reshape(-1, 3)
        )  # Bx1024,3
        all_att_gt = []
        all_rays = []
        if self.with_phase:
            all_ph_gt = []
        if self.with_phase:
            att_gt = (
                self.sample_grid(self.all_images[:, :, 0], uv)
                .to(self.device)
                .reshape(-1)
            )
            ph_gt = (
                self.sample_grid(self.all_images[:, :, 1], uv)
                .to(self.device)
                .reshape(-1)
            )
        else:
            att_gt = self.sample_grid(self.all_images, uv).to(self.device).reshape(-1)
        for obj_idx in range(self.SB):
            if self.opt.use_time:
                self.time_frame = self.all_times[obj_idx]
            rays = self.get_ray_batches()
            all_rays.append(rays)

        self.target_att = att_gt
        if self.with_phase:
            self.target_ph = ph_gt
        self.rays = torch.stack(all_rays)

    def get_ray_sample_new(self):
        self.get_encode_views()
        self.get_ray_function()
        pix_inds = torch.randint(
            0, self.NV * self.H * self.W, (self.opt.num_random_rays,)
        )  # [1024]
        self.ray_origins = self.ray_origins.reshape(-1, self.ray_origins.shape[-1])[
            pix_inds
        ].to(self.device)
        self.ray_directions = self.ray_directions.reshape(
            -1, self.ray_directions.shape[-1]
        )[pix_inds].to(self.device)
        all_att_gt = []
        all_rays = []
        if self.with_phase:
            all_ph_gt = []
        for obj_idx in range(self.SB):
            if self.with_phase:
                att_gt = (
                    self.all_images[obj_idx, :, 0].reshape(-1)[pix_inds].to(self.device)
                )
                ph_gt = (
                    self.all_images[obj_idx, :, 1].reshape(-1)[pix_inds].to(self.device)
                )
            else:
                att_gt = self.all_images[obj_idx].reshape(-1)[pix_inds].to(self.device)
            if self.opt.use_time:
                self.time_frame = self.all_times[obj_idx]
            rays = self.get_ray_batches()
            all_att_gt.append(att_gt)
            if self.with_phase:
                all_ph_gt.append(ph_gt)
            all_rays.append(rays)
        all_att_gt = torch.stack(all_att_gt)  # (SB, ray_batch_size, 2)
        all_rays = torch.stack(all_rays)
        self.target_att = all_att_gt.reshape(-1)
        if self.with_phase:
            all_ph_gt = torch.stack(all_ph_gt)
            self.target_ph = all_ph_gt.reshape(-1)
        self.rays = all_rays
        self.select_inds = pix_inds

    def get_ray_sample_gradient(self):
        """Sample rays by gradient. Different gradient are used for different object"""
        self.get_encode_views()
        # Gradient sampling
        all_att_gt = []
        if self.with_phase:
            all_ph_gt = []
        all_rays = []
        all_prob = []
        for obj_idx in range(self.SB):
            self.get_ray_function()
            all_views = self.all_images[obj_idx].squeeze()
            if obj_idx == 0 or not self.opt.share_gradient:
                if self.with_phase:
                    all_gradient = self.get_gradient_batch(
                        all_views[:, self.opt.gradient_sampling_channel]
                    )
                else:
                    all_gradient = self.get_gradient_batch(all_views)
                gradient_flat = all_gradient.reshape(-1)
                sampling_weighted_gradient = torch.multinomial(
                    gradient_flat, self.opt.num_random_rays
                )
                print(f"calculating gradient for object {obj_idx}")
            self.ray_origins = self.ray_origins.reshape(-1, self.ray_origins.shape[-1])[
                sampling_weighted_gradient
            ].to(self.device)
            self.ray_directions = self.ray_directions.reshape(
                -1, self.ray_directions.shape[-1]
            )[sampling_weighted_gradient].to(self.device)
            if self.opt.use_time:
                self.time_frame = self.all_times[obj_idx]
            gradient_sample_prob = gradient_flat[sampling_weighted_gradient]
            if self.with_phase:
                att_gt = (
                    self.all_images[obj_idx, :, 0]
                    .reshape(-1)[sampling_weighted_gradient]
                    .to(self.device)
                )
                ph_gt = (
                    self.all_images[obj_idx, :, 1]
                    .reshape(-1)[sampling_weighted_gradient]
                    .to(self.device)
                )
            else:
                att_gt = (
                    self.all_images[obj_idx]
                    .reshape(-1)[sampling_weighted_gradient]
                    .to(self.device)
                )
            rays = self.get_ray_batches()
            all_att_gt.append(att_gt)
            if self.with_phase:
                all_ph_gt.append(ph_gt)
            all_rays.append(rays)
            all_prob.append(gradient_sample_prob)
        all_att_gt = torch.stack(all_att_gt)  # (SB, ray_batch_size, 2)
        all_rays = torch.stack(all_rays)
        all_prob = torch.stack(all_prob)
        self.rays = all_rays
        self.select_inds = sampling_weighted_gradient
        self.gradient_sample_prob = all_prob.reshape(-1)
        self.target_att = all_att_gt.reshape(-1)
        if self.with_phase:
            all_ph_gt = torch.stack(all_ph_gt)
            self.target_ph = all_ph_gt.reshape(-1)

    @staticmethod
    def get_gradient_numpy(img):
        """Get gradient using numpy"""
        sx, sy = [np.gradient(img)[i] + 1e-8 for i in range(2)]
        # Get square root of sum of squares
        gradient = np.hypot(sx, sy)
        return gradient

    def get_gradient_batch(self, images):
        """input (C,H,W) tensor, output gradient (C,H,W)"""
        images = images.cpu().numpy()
        gradient = np.empty_like(images)
        for i in range(images.shape[0]):
            gradient[i] = self.get_gradient_numpy(images[i])
        # gradient = torch.tensor(gradient)
        gradient = torch.from_numpy(gradient).to(dtype=self.dtype, device=self.device)
        return gradient

    def optimization(self):
        """The optimization function. Could be biased with gradient sampling"""
        self.is_train = True
        if self.with_phase:
            self.phase_o = []
        self.attenuation_o = []
        if self.opt.use_gradient_sampling:
            self.get_ray_sample_grid()
        else:
            self.get_ray_sample_new()
        self.optimizer.zero_grad()
        self.model_coarse.train()
        if self.model_fine:
            self.model_fine.train()
        if self.opt.use_encoder:
            self.encoder.train()
         
            
        if self.with_phase:
            self.atten_coarse, self.atten_fine, self.ph_coarse, self.ph_fine = (
                None,
                None,
                None,
                None,
            )
            (
                self.atten_coarse,
                self.ph_coarse,
                self.atten_fine,
                self.ph_fine,
            ) = self.run_onix(mode="train")
            self.target_complex = torch.cat((self.target_att, self.target_ph), -1)
            self.complex_coarse = torch.cat((self.atten_coarse, self.ph_coarse), -1)
            self.coarse_loss = torch.nn.functional.mse_loss(
                self.complex_coarse, self.target_complex
            )
        else:
            self.atten_coarse, self.atten_fine = None, None
            self.atten_coarse, self.atten_fine = self.run_onix(
                mode="train",
            )
            self.coarse_loss = torch.nn.functional.mse_loss(
                self.atten_coarse, self.target_att
            )
        self.fine_loss = 0.0
        if self.atten_fine is not None:
            # NOT TESTED! We only used coarse model.
            if self.with_phase:
                self.complex_fine = torch.cat((self.atten_fine, self.ph_fine), -1)
                self.fine_loss = torch.nn.functional.mse_loss(
                    self.complex_fine, self.target_complex
                )
            else:
                self.fine_loss = torch.nn.functional.mse_loss(
                    self.atten_fine, self.target_att
                )
        self.ray_loss = self.coarse_loss + self.fine_loss
            
        self.total_mse_loss = self.ray_loss * self.lambda_mse

        #self.total_loss = (
        #    self.ray_loss * self.lambda_mse + self.pde_loss * self.lambda_pde
        #)
        self.update_gan()

    def validation(self):
        self.is_train = False
        if self.with_phase:
            self.phase_o = []
        self.attenuation_o = []

        if self.opt.val_idx >= 0:
            self.val_idx = self.opt.val_idx  # todo
        else:
            if self.opt.num_val_views != 0:
                self.val_idx = randint(
                    self.all_images.shape[1] - self.opt.num_val_views,
                    self.all_images.shape[1] - 1,
                )
            else:
                self.val_idx = randint(0, self.all_images.shape[1] - 1)
        self.res_H = self.opt.eval_H if self.opt.eval_H else self.all_images.shape[-2]
        self.res_W = self.opt.eval_W if self.opt.eval_W else self.all_images.shape[-1]
        self.get_ray_validation()
        self.model_coarse.eval()
        if self.model_fine:
            self.model_fine.eval()
        if self.opt.use_encoder:
            self.encoder.eval()

        if self.with_phase:
            self.atten_coarse, self.atten_fine, self.ph_coarse, self.ph_fine = (
                None,
                None,
                None,
                None,
            )
            # target_ray_values = None
            (
                self.atten_coarse,
                self.ph_coarse,
                self.atten_fine,
                self.ph_fine,
            ) = self.run_onix(
                mode="validation",
            )

            self.image_coarse_att = self.atten_coarse.reshape([self.res_H, self.res_W])[
                None, None, :, :
            ]
            self.image_coarse_ph = self.ph_coarse.reshape([self.res_H, self.res_W])[
                None, None, :, :
            ]

            self.image_coarse = torch.cat(
                (self.image_coarse_att, self.image_coarse_ph), 1
            )

            if self.atten_fine is not None:
                # NOT TESTED! We only used coarse model.
                self.image_fine_att = self.atten_fine.reshape([self.res_H, self.res_W])[
                    None, None, :, :
                ]
                self.image_fine_ph = self.ph_fine.reshape([self.res_H, self.res_W])[
                    None, None, :, :
                ]
                self.image_fine = torch.cat(
                    (self.image_fine_att, self.image_fine_ph), 1
                )

            self.gt = self.gt.squeeze()[None, :, :, :]
        else:
            self.atten_coarse, self.atten_fine = None, None
            self.atten_coarse, self.atten_fine = self.run_onix(
                mode="validation",
            )
            self.image_coarse = self.atten_coarse.reshape([self.res_H, self.res_W])[
                None, None, :, :
            ]
            if self.atten_fine is not None:
                self.image_fine = self.atten_fine.reshape([self.res_H, self.res_W])[
                    None, None, :, :
                ]

            self.gt = self.gt.squeeze()[None, None, :, :]

        # save validation loss
        if (self.res_H, self.res_W) == (
            self.all_images.shape[-2],
            self.all_images.shape[-1],
        ):
            self.val_loss = torch.nn.functional.mse_loss(self.gt, self.image_coarse)
        if self.generate_3D:
            print(f"Validation loss is: {self.val_loss:.6f}")
            
            self.attenuation_o = torch.cat(self.attenuation_o, 0)
            x = self.reshape_3d(
                self.attenuation_o, self.attenuation_path
            )  # Save 3D 
            self.reshaped_3d_att.append(x)
            
            #if self.with_phase:
            #    self.phase_o = torch.cat(self.phase_o, 0)
            #    self.reshape_3d(self.phase_o, self.phase_path)

    def reshape_3d(self, x, savename):
        x = x.detach().cpu().numpy()
        x = x.reshape(-1, x.shape[-1])
        x = x.reshape(self.res_H, self.res_W, self.opt.num_coarse)
        return x
        #np.save(savename, x)  # todo: need to save npy?

    def get_ray_validation(self):
        SB, NV, _, H, W = self.all_images.shape
        if self.opt.n_views > NV:
            raise ValueError(
                f"Selected number of input should be less than the number of available input! Only {NV} inputs are available!"
            )
        self.image_shape = torch.tensor([H, W])
        try:
            self.r_index = torch.tensor(self.opt.val_view_idx)
        except:
            self.r_index = torch.randperm(NV - self.opt.num_val_views)[
                : self.opt.n_views
            ].sort()[0]
        print(f"Validation view index: {self.r_index}")

        self.encode_images = self.all_images[:, self.r_index]
        self.all_poses = self.all_poses.squeeze(0)
        self.encode_pose = self.all_poses[self.r_index]
        pose_target = self.all_poses[self.val_idx : self.val_idx + 1, :3, :4].to(
            self.device
        )
        self.encode_pose = self.encode_pose[None, ...].repeat(SB, 1, 1, 1)
        self.encode_pose = self.encode_pose.reshape(-1, 4, 4)
        self.gt = self.all_images[:, self.val_idx : self.val_idx + 1]
        self.ray_origins, self.ray_directions = get_ray_bundle(
            self.res_H, self.res_W, pose_target
        )
        if self.opt.use_time:
            self.time_frame = self.all_times
        pix_inds = torch.arange(self.res_H * self.res_W)
        self.ray_origins = self.ray_origins.reshape(-1, self.ray_origins.shape[-1])[
            pix_inds
        ].to(self.device)
        self.ray_directions = self.ray_directions.reshape(
            -1, self.ray_directions.shape[-1]
        )[pix_inds].to(self.device)
        all_rays_gt = []
        all_rays = []
        for obj_idx in range(SB):
            if self.with_phase:
                rays_gt = (
                    self.all_images[obj_idx]
                    .permute(0, 2, 3, 1)
                    .contiguous()
                    .reshape(-1, 2)[pix_inds]
                    .to(self.device)
                )
            else:
                rays_gt = self.all_images[obj_idx].reshape(-1)[pix_inds].to(self.device)
            rays = self.get_ray_batches()

            all_rays_gt.append(rays_gt)
            all_rays.append(rays)
        all_rays_gt = torch.stack(all_rays_gt)  # (SB, ray_batch_size, 3)
        all_rays = torch.stack(all_rays)
        self.target_s = all_rays_gt.reshape(-1)
        self.rays = all_rays
        self.target_s = self.gt.reshape(-1, 1)[pix_inds].to(self.device)
        self.select_inds = pix_inds
        self.SB = SB

    def get_discriminator(self):

        disc_kwargs = {
            "nc": 3,  # channels for patch discriminator
            "ndf": 64,
            "imsize": 256,
            "hflip": False,
        }

        discriminator = Discriminator(**disc_kwargs)

    def encoder_index(self, uv, latent):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param uv (B, N, 2) image points (x,y)
        :return (B, L, N) L is latent size
        """
        uv = uv * 2
        uv = uv.unsqueeze(2)  # (B, N, 1, 2) (8, 2048, 1, 2)
        samples = F.grid_sample(
            latent,
            uv,
            align_corners=True,
            mode=self.opt.index_interp,
            padding_mode=self.opt.index_padding,
        )
        return samples[:, :, :, 0]  # (B, C, N)

    def encode(self, pts, images):
        self.num_objs = images.size(0)
        if len(images.shape) == 5:
            self.num_views_per_obj = images.size(1)
            images = images.reshape(-1, *images.shape[2:])
        else:
            self.num_views_per_obj = 1

        latent = self.encoder(images)  

        if not self.opt.use_camera_space_pnts:
            uv = -pts[:, :, :2] / (pts[:, :, 2:] + 1.5)
        else:
            uv = -pts[:, :, :2]
            uv[:, :, :1] = -uv[:, :, :1]
        uv = uv.to(self.dtype)
        latent = self.encoder_index(uv, latent)
        latent = latent.transpose(1, 2).reshape(-1, self.latent_size)
        return latent
    
    def get_camera_pixel(self,pts,images):
        """
        Newly added
        Add information about the camera pixel, instead of encoding them.
        """
        
        self.num_objs = images.size(0)
        if len(images.shape) == 5:
            self.num_views_per_obj = images.size(1)
            images = images.reshape(-1, *images.shape[2:]) # [4,2,128,128]
        else:
            self.num_views_per_obj = 1
            
        if not self.opt.use_camera_space_pnts:
            uv = -pts[:, :, :2] / (pts[:, :, 2:] + 1.5)
        else:
            uv = -pts[:, :, :2]
            uv[:, :, :1] = -uv[:, :, :1]
        uv = uv.to(self.dtype)
        uv = uv.unsqueeze(2)  # (B, N, 1, 2) (8, 2048, 1, 2)
        samples = F.grid_sample(
            images,
            uv,
            align_corners=True,
            mode=self.opt.index_interp,
            padding_mode=self.opt.index_padding,
        ) 
        #try:      
        return samples[:, (0,), :, 0].transpose(0, 2).reshape(-1, self.NV)  # change n_views to self.NV, should always match the number of cameras 
        #except:
        #   return samples[:, :, :, 0].transpose(1, 2).reshape(-1,self.test_dataset.NV) # during validation, self.NV is not defined.
    
    def get_ray_batches(self):
        z_near = self.opt.z_near * torch.ones_like(self.ray_directions[..., :1])
        z_far = self.opt.z_far * torch.ones_like(self.ray_directions[..., :1])
        if self.opt.use_time:
            time_stamp = self.time_frame * torch.ones_like(self.ray_directions[..., :1])
        self.cat_dim = 8 if not self.opt.use_time else 9
        self.rays = torch.cat(
            (self.ray_origins, self.ray_directions, z_near, z_far), dim=-1
        )  # remove 1st dimension
        if self.opt.use_time:
            self.rays = torch.cat((self.rays, time_stamp), dim=-1)
        self.rays = self.rays.view((-1, self.cat_dim))
        return self.rays

    def get_ray_pts(self, batch):
        ro, rd = batch[..., :3], batch[..., 3:6]
        bounds = batch[..., 6 : self.cat_dim].view(
            (batch.shape[0], -1, 1, self.cat_dim - 6)
        )  # [2048, 1, 2]
        near, far = bounds[..., 0], bounds[..., 1]  # [2048,1]
        if self.opt.use_time:
            time = bounds[..., 2]
        t_vals = torch.linspace(
            0.0,
            1.0,
            self.opt.num_coarse,
            dtype=self.dtype,
            device=self.device,
        )
        if not self.opt.lindisp:
            self.z_vals = near * (1.0 - t_vals) + far * t_vals
        else:
            self.z_vals = 1.0 / (
                1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals
            )  # [2048,64]
        if self.opt.perturb:
            # Get intervals between samples.
            mids = 0.5 * (self.z_vals[..., 1:] + self.z_vals[..., :-1])
            upper = torch.cat((mids, self.z_vals[..., -1:]), dim=-1)
            lower = torch.cat((self.z_vals[..., :1], mids), dim=-1)
            # Stratified samples in those intervals.
            t_rand = torch.rand(self.z_vals.shape, dtype=self.dtype, device=self.device)
            self.z_vals = lower + (upper - lower) * t_rand
        pts = ro[..., None, :] + rd[..., None, :] * self.z_vals[..., :, None]
        return pts

    def main_network(self, pts, batch, coarse=True):        
        # Convert camera pose matrix to a "transform matrix"
        if self.opt.use_camera_space_pnts:
            rot = self.encode_pose[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
            trans = -torch.bmm(rot, self.encode_pose[:, :3, 3:])  # (B, 3, 1)
            poses = torch.cat((rot, trans), dim=-1)  # (B, 3, 4)
            
        pts_flat = pts.reshape((self.SB, -1, pts.shape[-1]))
        self.xyz_pde = pts_flat.reshape((-1,pts.shape[-1]))  # world coordinate
        if self.opt.enable_pinn:
            self.xyz_pde.requires_grad = True 
            
        self.num_onix_pts = self.xyz_pde.size(0)  
        
        xyz = repeat_interleave(self.xyz_pde.reshape((self.SB, -1, pts.shape[-1])), self.opt.n_views)  # (SB*NS, B, 3)
        
        if self.opt.use_camera_space_pnts:
            # Transform query points into the camera spaces of the input views
            xyz_rot = torch.matmul(poses[:, None, :3, :3], xyz.unsqueeze(-1))[..., 0]
            xyz = xyz_rot + poses[:, None, :3, 3]
        
        if self.opt.use_encoder:
            latent = self.encode(xyz.detach(), self.encode_images)  # [2048,512] 
                
        elif self.opt.use_camera_pixel:
            if (self.NV > self.opt.n_views) and (self.opt.n_views == 1):  # Should include info from all cams!
                xyz_to_cam = repeat_interleave(pts_flat, self.NV)
                if self.opt.use_camera_space_pnts:
                    rot_all = self.all_poses.reshape(-1, 4, 4)[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
                    trans_all = -torch.bmm(rot_all, self.all_poses.reshape(-1, 4, 4)[:, :3, 3:])  # (B, 3, 1)
                    poses_all = torch.cat((rot_all, trans_all), dim=-1)  # (B, 3, 4)
                    xyz_to_cam_rot = torch.matmul(poses_all[:, None, :3, :3], xyz_to_cam.unsqueeze(-1))[..., 0]
                    xyz_to_cam = xyz_to_cam_rot + poses_all[:, None, :3, 3]                   
                latent = self.get_camera_pixel(xyz_to_cam.detach(),self.all_images) 
            else:    
                latent = self.get_camera_pixel(xyz.detach(),self.encode_images) 
                latent = repeat_interleave(latent,self.NV)
                
        embedded = xyz.reshape(-1,3)
        
        '''
        self.num_onix_pts = embedded.size(0)   
        if self.opt.enable_pinn:
            self.num_pde_pts = self.num_onix_pts // self.opt.num_pts_ratio  # reduced number of points as onix
            if (self.opt.pinn_sampling_mode == "dynamic") & (self.epoch % self.opt.epoch_pde_sampling == 0) & (self.FLAG_pde == True):
                self.xyz_pde_torch,self.t_pde_torch = self.sample_pde_points(self.num_pde_pts)
                self.FLAG_pde = False 
            embedded = torch.cat((embedded,self.xyz_pde_torch.to(self.device)),dim=0)   #[xxx,3]

        if self.opt.enable_v_p:
            if (self.epoch % self.opt.epoch_vp_sampling == 0) & (self.FLAG_vp == True):
                self.num_vp_pts = self.num_onix_pts // self.opt.num_pts_ratio
                self.xyz_vp_torch,self.t_vp_torch,self.gt_vp_torch = self.sample_vppsi_points(self.num_vp_pts)
                self.FLAG_vp = False
                self.gt_vp_torch = self.gt_vp_torch.to(self.device)
            embedded = torch.cat((embedded,self.xyz_vp_torch.to(self.device)),dim=0) #[xxx,3]
        '''
            
        if self.opt.enable_encoding_fn_xyz:  
            embedded = self.encode_position_fn(embedded)  # [2x132072,63]  
             
        if self.opt.use_time:
            times = batch[..., None, -1:]  # [2, 1024, 1, 1]
            input_time = times.expand([*pts.shape[:-1], 1])  # [2, 1024, 256, 1]
            self.t_pde = input_time.reshape(-1,1) # For calculation of pde loss
            
            if self.opt.enable_pinn:
                self.t_pde.requires_grad = True   
                self.pde_index = ((torch.abs(self.xyz_pde[:,0])<=0.5) & (torch.abs(self.xyz_pde[:,1])<=0.5) & \
                            (torch.abs(self.xyz_pde[:,2])<=0.5) & \
                            (self.t_pde[:,0]>=self.opt.pde_start_time) & (self.t_pde[:,0]<=self.opt.pde_end_time)).nonzero()   # Select the points inside the cube to calculate pde loss 
                #print(self.pde_index.shape)
            #if self.opt.use_camera_space_pnts:
            input_time = repeat_interleave(
                self.t_pde.reshape((*pts.shape[:-1], 1)), self.opt.n_views
            )  # [8, 1024, 256, 1]
            input_time_flat = input_time.reshape(-1, 1)

            '''
            if self.opt.enable_pinn:
                input_time_flat = torch.cat((input_time_flat,self.t_pde_torch.to(self.device)),dim=0)

            if self.opt.enable_v_p:
                input_time_flat = torch.cat((input_time_flat,self.t_vp_torch.to(self.device)),dim=0)
            '''
            
            if self.opt.enable_encoding_fn_time:
                embedded_time = self.encode_time_fn(input_time_flat)  
            else:
                embedded_time = input_time_flat
                
            embedded = torch.cat((embedded, embedded_time), dim=-1)  # [2048x64x2,90]
        
        mlp_input = embedded  # [2048x64x2,90] --> output [2x1024x64,2]

        if (self.opt.use_encoder or self.opt.use_camera_pixel):
            mlp_input = torch.cat((latent, mlp_input), dim=-1)  # [2048x64x2,512+90]
          
        _, B, _ = pts_flat.shape
        NS = self.opt.n_views
        # Run main NeRF network
        if coarse or self.model_fine is None:
            if (NS > 1):      #self.opt.use_encoder:
                mlp_output = self.model_coarse(
                    mlp_input,
                    combine_inner_dims=(NS, B),
                )  # [1024x64x2,2]
            else:
                mlp_output = self.model_coarse(
                    mlp_input,
                    #combine_inner_dims=(NS, B),
                )  # [1024x64x2,2]               
        else:
            mlp_output = self.model_fine(
                mlp_input,
                combine_inner_dims=(NS, B),
            )  # [1024x64x2,2]

        mlp_output = mlp_output.reshape(-1, mlp_output.shape[-1])  #
        return mlp_output

    def predict_with_rendering(self, batch):
        pts = self.get_ray_pts(batch)
        radiance_field = self.main_network(pts, batch, coarse=True)
        
        if self.is_train:
            if (self.opt.enable_pinn) and (self.pde_index.shape[0] > 0) and (self.lambda_pde > 0):
                res_pde = equation_droplet(self.xyz_pde,self.t_pde,radiance_field[:])[self.pde_index] 
                res_div = divergence(self.xyz_pde, radiance_field[:])[self.pde_index]             
                self.pde_loss += torch.nn.functional.mse_loss(
                    res_pde[:], torch.zeros_like(res_pde[:],dtype=torch.float32))  # self.pde_index
                self.pde_loss += torch.nn.functional.mse_loss(
                    res_div[:], torch.zeros_like(res_div[:],dtype=torch.float32)
                )
            
            if self.opt.enable_v_p:
                # temporarily added
                temp = torch.nn.functional.relu(radiance_field[-self.num_vp_pts:,0])
                if torch.max(temp) < 1e-8:
                    scale = 1
                else:
                    scale = torch.max(self.gt_vp_torch[:,4]) / torch.max(temp)
                
                self.psi_loss += torch.nn.functional.mse_loss(
                    temp*scale, self.gt_vp_torch[:,4]
                )
                
                self.u_loss += nrmse_fn(                                    #torch.nn.functional.mse_loss(
                    radiance_field[-self.num_vp_pts:,2:5], self.gt_vp_torch[:,0:3]
                )
                self.p_loss += nrmse_fn(                                    #torch.nn.functional.mse_loss(
                    radiance_field[-self.num_vp_pts:,5], self.gt_vp_torch[:,3]
                )
               
        if self.is_train:
            radiance_field_noise_std = self.radiance_field_noise_std
        else:
            radiance_field_noise_std = 0.0
        if self.with_phase:
            (att_coarse, ph_coarse, weights,) = self.volume_render_xray_propagation(
                radiance_field[:self.num_onix_pts],
                batch[0, ..., 3:6],
                self.z_vals,
                radiance_field_noise_std=radiance_field_noise_std,
            )
            att_fine, ph_fine = None, None
            #duda = grad(att_coarse,self.mlp_input) #[:,[0,1,2,63]]
        else:
            (att_coarse, weights,) = self.volume_render_xray_propagation(
                radiance_field[:self.num_onix_pts],
                batch[0, ..., 3:6],
                self.z_vals,
                radiance_field_noise_std=radiance_field_noise_std,
            )
            att_fine = None
        if self.opt.num_fine > 0:
            SB, R, K = self.z_vals.shape
            z_vals_mid = 0.5 * (self.z_vals[..., 1:] + self.z_vals[..., :-1])
            weights = weights[..., 1:-1].reshape([SB, R, K - 2])
            z_vals = torch.zeros([SB, R, self.opt.num_fine + self.opt.num_coarse]).to(
                self.z_vals
            )
            for obj_idx in range(SB):
                z_samples = sample_pdf(
                    z_vals_mid[obj_idx],
                    weights[obj_idx],
                    self.opt.num_fine,
                    det=(self.opt.perturb == 0.0),
                )
                z_samples = z_samples.detach()
                z_vals[obj_idx], _ = torch.sort(
                    torch.cat((self.z_vals[obj_idx], z_samples), dim=-1), dim=-1
                )
            ro, rd = batch[..., :3], batch[..., 3:6]
            pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]
            radiance_field = self.main_network(pts, batch, coarse=False)
            if self.with_phase:
                att_fine, ph_fine, _, = self.volume_render_xray_propagation(
                    radiance_field[:self.num_onix_pts],
                    batch[0, ..., 3:6],
                    z_vals,
                    radiance_field_noise_std=radiance_field_noise_std,
                )
            else:
                att_fine, _, = self.volume_render_xray_propagation(
                    radiance_field[:self.num_onix_pts],
                    batch[0, ..., 3:6],
                    z_vals,
                    radiance_field_noise_std=radiance_field_noise_std,
                )
        if self.with_phase:
            return att_coarse, ph_coarse, att_fine, ph_fine
        else:
            return att_coarse, att_fine

    def volume_render_xray_propagation(
        self,
        radiance_field: torch.Tensor,
        ray_directions: torch.Tensor,
        depth_values: torch.Tensor,
        radiance_field_noise_std=0.0,
    ):
        """
        Render the image using the law of X-ray propagation
        """
        one_e_10 = torch.tensor([1e-10], dtype=self.dtype, device=self.device)
        dists = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                one_e_10.expand(depth_values[..., :1].shape),
            ),
            dim=-1,
        )  # [2048,64]
        radiance_field = radiance_field.reshape(
            -1, dists.shape[-1], radiance_field.shape[-1]
        )
        dists = dists * ray_directions[..., None, :].norm(p=2, dim=-1)  # [2048,64]
        dists = dists.reshape(-1, dists.shape[-1])

        noise = 0.0
        if radiance_field_noise_std > 0.0:
            noise = (
                torch.randn(
                    radiance_field[..., 0].shape,
                    dtype=self.dtype,
                    device=self.device,
                )
                * radiance_field_noise_std
            )
        sigma_a = torch.nn.functional.relu(
            radiance_field[..., 0] + noise
        )  # (width, height, num_samples) # [2048, 64]
        if self.with_phase:
            phase = torch.nn.functional.relu(radiance_field[..., 1] + noise)
        alpha = 1.0 - torch.exp(-sigma_a * dists)
        weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

        if self.generate_3D:
            self.attenuation_o.append(sigma_a * dists)
            if self.with_phase:
                self.phase_o.append(phase * dists)
            
        attenuation_map = (sigma_a * dists).sum(
            dim=-1
        )  # Compute attenuation of each sample along each ray
        if self.with_phase:
            phase_map = (phase * dists).sum(dim=-1)
            return attenuation_map, phase_map, weights
        else:
            return attenuation_map, weights

    def get_losses(self, loss_list):
        errors_list = OrderedDict()
        for name in loss_list:
            if isinstance(name, str):
                errors_list[name] = float(getattr(self, name + "_loss"))
        return errors_list

    def print_losses(self, logfile, epoch, iters, losses):
        message = "Epoch [{}/{}], Step [{}/{}]".format(
            epoch + 1, self.opt.num_epochs, iters + 1, self.total_step
        )
        for name, loss in losses.items():
            message += ", {:s}: {:.5f}".format(name, loss)
        print(message)
        with open(
            logfile,
            "a",
            encoding="utf-8",
        ) as f:
            print(message, file=f)

    def get_val_losses(self):
        return self.get_losses(self.val_names)

    def get_current_losses(self):
        return self.get_losses(self.loss_names)

    def print_current_losses(self, **kwargs):
        self.print_losses(self.save_log, **kwargs)

    def print_val_losses(self, **kwargs):
        self.print_losses(self.save_val, **kwargs)

    @staticmethod
    def create_dir_if_not_exist(path):
        if os.path.exists(path):
            print("Warning: Overwriting folder: {}".format(path))
        if not os.path.exists(path):
            os.makedirs(path)

    def save_inputs(self):
        for i in range(self.all_images.shape[0]):
            save_tensor_plot(self.all_images[i], self.save_run, f"input_images_{i}")

    def visual_iter(self, epoch, iteration, obj_id):
        save_name = "{:03d}epoch_{:04d}step".format(epoch + 1, iteration + 1)
        gt_name = (
            f"{save_name}_gt_obj{obj_id}_view{self.val_idx}_enc{self.r_index.tolist()}"
        )
        if self.opt.use_time:
            gt_name += f"_time{self.all_times.item()}"
        save_tensor_plot(
            self.atten_coarse.detach().reshape([self.res_H, self.res_W]),
            self.save_run,
            f"{save_name}_coarse_att_obj{obj_id}",
        )
        if self.with_phase:
            save_tensor_plot(
                self.ph_coarse.detach().reshape([self.res_H, self.res_W]),
                self.save_run,
                f"{save_name}_coarse_ph_obj{obj_id}",
            )
            save_tensor_plot(
                self.gt.squeeze()[0],
                self.save_run,
                f"{gt_name}_course_att",
            )
            save_tensor_plot(
                self.gt.squeeze()[1],
                self.save_run,
                f"{gt_name}_course_ph",
            )
        else:
            save_tensor_plot(
                self.gt.squeeze(),
                self.save_run,
                f"{gt_name}_course",
            )

        if self.atten_fine is not None:
            save_tensor_plot(
                self.atten_fine.detach().reshape([self.res_H, self.res_W]),
                self.save_run,
                f"{save_name}_fine_att",
            )
            if self.with_phase:
                save_tensor_plot(
                    self.ph_fine.detach().reshape(
                        [self.all_images.shape[-2], self.all_images.shape[-1]]
                    ),
                    self.save_run,
                    f"{save_name}_fine_ph",
                )

    def save_net(self, name, epoch, net, optimizer, loss):
        model_save_name = f"{name}_{epoch}ep.pt"
        path = f"{self.save_run}/save"
        if not os.path.exists(path):
            os.makedirs(path)
        print("Saving trained model {}".format(model_save_name))
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            path + f"/{model_save_name}",
        )

    def save_models(self, epoch):
        self.save_net(
            "coarse", epoch + 1, self.model_coarse, self.optimizer, self.coarse_loss
        )
        if self.atten_fine is not None:
            self.save_net(
                "fine", epoch + 1, self.model_fine, self.optimizer, self.fine_loss
            )
        if self.opt.use_encoder:
            self.save_net(
                "encoder", epoch + 1, self.encoder, self.optimizer, self.coarse_loss
            )
        self.save_net(
            "discriminator", epoch + 1, self.netD, self.optimizer_D, self.D_coarse_loss
        )

    def adjust_learning_rate(self, epoch, optimizer, lr_start):
        """Learning rate decay"""
        lr_new = lr_start * (
            self.opt.lr_decay_factor ** (epoch // self.opt.lr_decay_steps)
        )
        print(f'learning rate is {lr_new}')
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_new

    def adjust_noise_factor(self, epoch):
        self.radiance_field_noise_std = self.opt.radiance_field_noise_std * (
            self.opt.noise_decay_factor ** (epoch / self.opt.noise_decay_steps)
        )

    def update_parameters(self, epoch):
        self.adjust_learning_rate(epoch, self.optimizer, lr_start=self.opt.lr)
        self.adjust_learning_rate(epoch, self.optimizer_D, lr_start=self.opt.lr_d)
        self.adjust_noise_factor(epoch)  # Adjust the noise factor if needed
        if self.opt.enable_v_p:      # Enable sampling for GT v,p field points
            self.FLAG_vp = True 
        if self.opt.enable_pinn:      # Enable sampling for pde points
            self.FLAG_pde = True 
        if self.opt.use_encoder:
            if epoch >= self.opt.num_encoder_epochs:
                set_requires_grad(self.encoder, False)
                print('   fix encoder')   
            else:
                print('   train encoder')     
   
    def load_trained_models(self, load_path, load_epoch):
        # fine_path = f"{load_path}/save/fine_{load_epoch}ep.pt"
        coarse_path = f"{load_path}/save/coarse_{load_epoch}ep.pt"
        
        coarse_checkpoint = torch.load(coarse_path, map_location=self.device)
        print("Loading model from ", coarse_path)
        
        if self.opt.use_encoder:
            encoder_path = f"{load_path}/save/encoder_{load_epoch}ep.pt"
            encoder_checkpoint = torch.load(encoder_path, map_location=self.device)
            self.encoder.load_state_dict(
                encoder_checkpoint["model_state_dict"], strict=False
            )
            print(f"Finish loading {encoder_path}")
            
        self.model_coarse.load_state_dict(
            coarse_checkpoint["model_state_dict"], strict=False
        )
        if self.opt.num_fine > 0:
            self.model_fine.load_state_dict(
                coarse_checkpoint["model_state_dict"], strict=False
            )     
    '''
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    '''      
                  
    def update_gan(self):
        set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        x_real = self.all_images.reshape(-1, *self.all_images.shape[-3:])
        hwfr = [self.all_images.shape[-2], self.all_images.shape[-1], (1, 1)]

        img_to_patch = ImgToPatch(self.sampler, hwfr[:3])

        x_real = img_to_patch(x_real.to(self.device))  # N_samples x C
        x_fake = self.complex_coarse.reshape(2, -1).permute(1, 0)
        self.D_coarse_loss = self.backward_D(self.netD, x_real, x_fake)
        self.D_fine_loss = 0.0
        if self.atten_fine is not None:
            self.D_fine_loss = self.backward_D(self.netD, x_real, x_fake)
        self.D_total_loss = self.D_coarse_loss + self.D_fine_loss
        self.optimizer_D.step()
        
        set_requires_grad(self.netD, False)
        self.optimizer.zero_grad()
        self.G_coarse_loss = (
            self.compute_loss(self.netD(x_fake), 1) * self.opt.lambda_generator
        )

        self.G_fine_loss = 0.0
        self.G_total_loss =  self.G_fine_loss + self.total_mse_loss + self.G_coarse_loss
        if self.opt.enable_pinn:
            self.G_total_loss += self.pde_loss * self.lambda_pde
        if self.opt.enable_v_p:
            self.G_total_loss += (self.u_loss * self.lambda_u + self.p_loss * self.lambda_p + self.psi_loss * self.lambda_psi)
        self.G_total_loss.backward()
        if self.clip_max != 0:
            if self.opt.use_encoder:
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(
                        self.model_coarse.parameters(), self.encoder.parameters()
                    ),
                    self.clip_max,
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(
                        self.model_coarse.parameters()),
                    self.clip_max,
                )

        self.optimizer.step()

    def backward_D(self, netD, real, fake):
        y = torch.zeros(self.opt.batch_size)
        pred_real = netD(real, y)
        loss_D_real = self.compute_loss(pred_real, 1)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.compute_loss(pred_fake, 0)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        # print(f'loss_D_real: {loss_D_real}, loss_D_fake: {loss_D_fake}')
        return loss_D

    def compute_loss(self, d_outs, target):

        d_outs = [d_outs] if not isinstance(d_outs, list) else d_outs
        loss = 0

        for d_out in d_outs:

            targets = d_out.new_full(size=d_out.size(), fill_value=target)
            self.gan_type = "standard"
            if self.gan_type == "standard":
                loss += F.binary_cross_entropy_with_logits(d_out, targets)
            elif self.gan_type == "wgan":
                loss += (2 * target - 1) * d_out.mean()
            else:
                raise NotImplementedError

        return loss / len(d_outs)   
    
            
     # Sample points for pde loss
    def sample_pde_points(self,num_pde_pts):
        print('          Selecting points for PDE calculation...')
        x_cor = np.random.rand(num_pde_pts,1) - 0.5
        y_cor = np.random.rand(num_pde_pts,1) - 0.5
        z_cor = np.random.rand(num_pde_pts,1) - 0.5
        t_pde_np = self.opt.pde_start_time + 1.0*(self.opt.pde_end_time-self.opt.pde_start_time)*np.random.rand(num_pde_pts,1)
        t_pde_np = t_pde_np.astype('float32')
        xyz_pde_np = np.hstack((z_cor,x_cor,y_cor)).astype('float32')   #(xxx,3)
        xyz_pde_torch = torch.from_numpy(xyz_pde_np)
        t_pde_torch = torch.from_numpy(t_pde_np)
        return xyz_pde_torch,t_pde_torch
    
    def init_u_p_psi(self):
        '''
            Sampling initial condition points
        '''
        
        gt_up_all = h5py.File(self.opt.load_path,'r')['/vp'][:][0][None,...]  #[1,4,128,128,128]
        
        # add psi_h 
        gt_psi_all = h5py.File(self.opt.load_path,'r')['/H_psi'][:][(0,),None,:,:,:] #[1,1,128,128,128]
        gt_up_all = np.concatenate((gt_up_all,gt_psi_all),axis=1)  #[1,5,128,128,128]
        print(gt_up_all.shape)
        
        self.H =  gt_up_all.shape[-1]
        self.num_t = 1    # only select the initial point, instead of gt_up_all.shape[0]
        node_x = np.linspace(-0.5, 0.5, self.H, dtype=np.float32, endpoint=True) # Notice: -0.5 first or 0.5 first
        node_y = np.linspace(0.5, -0.5, self.H, dtype=np.float32, endpoint=True) # Notice: -0.5 first or 0.5 first
        node_z = np.linspace(0.5, -0.5, self.H, dtype=np.float32, endpoint=True) # Notice: -0.5 first or 0.5 first
        node_t = np.linspace(self.opt.start_time,self.opt.end_time,self.num_t,dtype=np.float32, endpoint=True) # only select the first time point
        
        tt,xx,yy,zz = np.meshgrid(node_t,node_x,node_y,node_z,indexing='ij')  # [1,128,128,128]
        xyz_np_test = np.concatenate((xx[None,:],yy[None,:],zz[None,:]),axis=0)    #(3,1,num_pixel,num_pixel,num_pixel)
        self.xyz_np_test = xyz_np_test.reshape(3,-1).T   #(1*num_pixel**3,3)
        self.t_np_test = tt.reshape(-1, 1)   # (1*num_pixel**3,1)
        #self.gt_all = gt_up_all.reshape(self.num_t,4,-1).transpose(0,2,1).reshape(-1,4)  #[10*num_pixel**3,1]
        
        # add psi_h:
        self.gt_all = gt_up_all.reshape(self.num_t,5,-1).transpose(0,2,1).reshape(-1,5)  #[1*num_pixel**3,1]
    
        
    def sample_vppsi_points(self,num_vp_pts):
        '''
        # Sampling points for calculating v,p supervised loss
        # output: xyz_vp_torch, t_vp_torch, gt_vp_torch
        '''
        
        print('          Selecting points for u,p and psi field')
        
        try:
            index_selected = cp.asnumpy(cp.random.choice(self.num_t*self.H**3,num_vp_pts,replace=False))
        except:
            index_selected = np.random.choice(self.num_t*self.H**3,num_vp_pts,replace=False)
        xyz_selected = self.xyz_np_test[index_selected].astype('float32')
        t_selected = self.t_np_test[index_selected].astype('float32')
        gt_selected = self.gt_all[index_selected].astype('float32')
        
        return torch.from_numpy(xyz_selected), torch.from_numpy(t_selected), torch.from_numpy(gt_selected)




