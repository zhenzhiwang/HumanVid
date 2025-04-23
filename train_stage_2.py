import argparse
import copy
import logging
import math
import os
import os.path as osp
import random
import time
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import diffusers
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from collections import defaultdict
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection

from src.dataset.dance_video_h_v_camera import HumanDanceCameraVideoDataset
from src.dataset.dance_image_h_v_camera import OrientationBatchSampler, OrientationSampler
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid import Pose2VideoPipeline
from src.cameractrl.pose_adaptor import CameraPoseEncoder
from src.utils.util import (delete_additional_ckpt, import_filename, read_frames, seed_everything, save_checkpoint)
import wandb
warnings.filterwarnings("ignore")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")


class Net(nn.Module):
    def __init__(self, reference_unet: UNet2DConditionModel, 
                denoising_unet: UNet3DConditionModel, 
                pose_guider: PoseGuider, 
                reference_control_writer, 
                reference_control_reader, 
                camera_pose_encoder: CameraPoseEncoder,):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader
        self.camera_pose_encoder = camera_pose_encoder

    def forward(self, noisy_latents, timesteps, ref_image_latents, clip_image_embeds, pose_img, camera_embedding, uncond_fwd: bool = False,):
        # camera pose
        assert camera_embedding.ndim == 5 # b c f h w
        b = camera_embedding.shape[0] 
        camera_embedding_feature = self.camera_pose_encoder(camera_embedding)      # bf c h w
        assert len(camera_embedding_feature) == 1, "length of camera_embedding_feature should be 1, got {}".format(len(camera_embedding_feature))
        camera_embedding_feature = camera_embedding_feature[0]
        camera_embedding_feature = rearrange(camera_embedding_feature, '(b f) c h w -> b c f h w', b=b)
        
        # human pose
        pose_cond_tensor = pose_img.to(device="cuda")
        pose_fea = self.pose_guider(pose_cond_tensor)
        
        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_unet(ref_image_latents, ref_timesteps, encoder_hidden_states=clip_image_embeds, return_dict=False, )
            self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.denoising_unet(noisy_latents, timesteps, pose_cond_fea = pose_fea + camera_embedding_feature, encoder_hidden_states=clip_image_embeds, ).sample

        return model_pred


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps, mixed_precision=cfg.solver.mixed_precision, log_with="mlflow", project_dir="./mlruns", kwargs_handlers=[kwargs],)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    exp_name = cfg.exp_name
    save_dir = f"{cfg.output_dir}/{exp_name}"
    if accelerator.is_main_process:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    inference_config_path = "./configs/inference/inference_v2.yaml"
    infer_config = OmegaConf.load(inference_config_path)
    if accelerator.is_main_process:
        wandb.init(project="Animate-with-camera train stage 2")

    if cfg.solver.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif cfg.solver.mixed_precision == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(f"Do not support weight dtype: {cfg.weight_dtype} during training")

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(rescale_betas_zero_snr=True, timestep_spacing="trailing", prediction_type="v_prediction",)
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)

    image_enc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path,).to(dtype=weight_dtype, device="cuda")
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to("cuda", dtype=weight_dtype)
    reference_unet = UNet2DConditionModel.from_pretrained(cfg.base_model_path, subfolder="unet",).to(device="cuda", dtype=weight_dtype)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(cfg.base_model_path, cfg.mm_path, subfolder="unet",
                                                              unet_additional_kwargs=OmegaConf.to_container(infer_config.unet_additional_kwargs),).to(device="cuda")

    # human pose
    pose_guider = PoseGuider(conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)).to(device="cuda")  # dtype=weight_dtype

    # camera pose
    camera_pose_encoder_kwargs = OmegaConf.to_container(cfg.pose_encoder_kwargs)
    camera_pose_encoder = CameraPoseEncoder(**camera_pose_encoder_kwargs)
    
    # load checkpoint
    stage1_ckpt_dir = cfg.stage1_ckpt_dir
    stage1_ckpt_step = cfg.stage1_ckpt_step
    denoising_unet.load_state_dict(torch.load(os.path.join(stage1_ckpt_dir, f"denoising_unet-{stage1_ckpt_step}.pth"), map_location="cpu",), strict=False,)
    reference_unet.load_state_dict(torch.load(os.path.join(stage1_ckpt_dir, f"reference_unet-{stage1_ckpt_step}.pth"), map_location="cpu",), strict=False,)
    pose_guider.load_state_dict(torch.load(os.path.join(stage1_ckpt_dir, f"pose_guider-{stage1_ckpt_step}.pth"), map_location="cpu",), strict=False,)
    keys_to_remove = ("encoder_down_attention_blocks.0.0.attention_blocks.0.pos_encoder.pe", "encoder_down_attention_blocks.0.1.attention_blocks.0.pos_encoder.pe")
    state_dict = torch.load(os.path.join(stage1_ckpt_dir, f"camera_pose_encoder-{stage1_ckpt_step}.pth"), map_location="cpu")
    for key in keys_to_remove:
        if key in state_dict:
            del state_dict[key]
    camera_pose_encoder.load_state_dict(state_dict, strict=False,)

    if cfg.resume_from_checkpoint != "":
        stage2_ckpt_dir = cfg.resume_from_checkpoint
        stage2_ckpt_step = cfg.resume_step
        denoising_unet.load_state_dict(torch.load(os.path.join(stage2_ckpt_dir, f"motion_module-{stage2_ckpt_step}.pth"),map_location="cpu",),strict=False,)
        camera_pose_encoder.load_state_dict(torch.load(os.path.join(stage2_ckpt_dir, f"camera_pose_encoder-{stage2_ckpt_step}.pth"),map_location="cpu",),strict=False,)

    # Freeze
    vae.requires_grad_(False)
    image_enc.requires_grad_(False)
    reference_unet.requires_grad_(False)
    denoising_unet.requires_grad_(False)
    pose_guider.requires_grad_(False)
    # Set motion module and camera encoder learnable
    
    camera_pose_encoder.requires_grad_(True)
    for name, module in denoising_unet.named_modules():
        if "motion_modules" in name:
            for params in module.parameters():
                params.requires_grad = True

    reference_control_writer = ReferenceAttentionControl(reference_unet, do_classifier_free_guidance=False, mode="write", fusion_blocks="full",)
    reference_control_reader = ReferenceAttentionControl(denoising_unet, do_classifier_free_guidance=False, mode="read", fusion_blocks="full",)

    net = Net(reference_unet, denoising_unet, pose_guider, reference_control_writer, reference_control_reader, camera_pose_encoder,)

    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

    if cfg.solver.scale_lr:
        learning_rate = (cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes)
    else:
        learning_rate = cfg.solver.learning_rate #* np.sqrt(cfg.solver.gradient_accumulation_steps)

    # Initialize the optimizer
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, net.parameters()))
    # print trainable_params
    if accelerator.is_main_process:
        trainable_names = list(filter(lambda p: p[1].requires_grad, net.named_parameters()))

        def get_prefix(name, k):
            parts = name.split('.')
            prefix = '.'.join(parts[:k])
            return prefix

        prefix_params = defaultdict(list)
        prefix_param_sizes = defaultdict(int)

        for name, param in trainable_names:
            prefix = get_prefix(name, 2)  # Get the prefix by joining the first k elements
            prefix_params[prefix].append(name)
            prefix_param_sizes[prefix] += param.numel()

        print("Trainable parameters: ")
        for prefix, param_names in prefix_params.items():
            total_size = prefix_param_sizes[prefix] / 1e6
            print(f"    {prefix} (Size: {total_size} M)")

    logger.info(f"Total trainable params {len(trainable_params)}")
    optimizer = optimizer_cls(trainable_params, lr=learning_rate, betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2), weight_decay=cfg.solver.adam_weight_decay, eps=cfg.solver.adam_epsilon,)

    # Scheduler
    lr_scheduler = get_scheduler(cfg.solver.lr_scheduler, optimizer=optimizer, num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps, num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,)

    train_dataset = HumanDanceCameraVideoDataset(train_size_large=cfg.data.train_size_large, 
                                                 train_size_small=cfg.data.train_size_small, 
                                                 img_scale=(1.0, 1.1), 
                                                 n_sample_frames=cfg.data.n_sample_frames, 
                                                 sample_rate=cfg.data.sample_rate, 
                                                 horizontal_meta=cfg.data.horizontal_meta, 
                                                 vertical_meta = cfg.data.vertical_meta, )
    # Create the custom batch sampler
    horizontal_indices = list(range(len(train_dataset.horizontal_meta)))
    vertical_indices = list(range(len(train_dataset.horizontal_meta), len(train_dataset)))
    orientation_sampler = OrientationSampler(horizontal_indices, vertical_indices)
    dataset_seed = random.randint(0, 10000)
    batch_sampler = OrientationBatchSampler(orientation_sampler, batch_size=cfg.data.train_bs, rank = accelerator.process_index, seed = dataset_seed)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=2,)

    # Prepare everything with our `accelerator`.
    (net, optimizer, train_dataloader, lr_scheduler,) = accelerator.prepare(net, optimizer, train_dataloader, lr_scheduler,)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.solver.gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(cfg.solver.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(exp_name, init_kwargs={"mlflow": {"run_name": run_time}},)
        # dump config file
        mlflow.log_dict(OmegaConf.to_container(cfg), "config.yaml")

    # Train!
    total_batch_size = (cfg.data.train_bs
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.data.train_bs}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")
    global_step = 0
    first_epoch = 0
    
    if cfg.resume_from_checkpoint != "":
        global_step = cfg.resume_step + 1

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, cfg.solver.max_train_steps), disable=not accelerator.is_local_main_process,)
    progress_bar.set_description("Steps")

    while global_step < cfg.solver.max_train_steps:
        train_loss = 0.0
        t_data_start = time.time()
        for step, batch in enumerate(train_dataloader):
            t_data = time.time() - t_data_start
            with accelerator.accumulate(net):
                # Convert videos to latent space
                pixel_values_vid = batch["pixel_values_vid"].to(weight_dtype)
                with torch.no_grad():
                    video_length = pixel_values_vid.shape[1]
                    pixel_values_vid = rearrange(pixel_values_vid, "b f c h w -> (b f) c h w")
                    latents = vae.encode(pixel_values_vid).latent_dist.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                    latents = latents * 0.18215

                noise = torch.randn_like(latents)
                if cfg.noise_offset > 0:
                    noise += cfg.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1, 1), device=latents.device, )
                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(0, train_noise_scheduler.num_train_timesteps, (bsz, ), device=latents.device, )
                timesteps = timesteps.long()

                pixel_values_pose = batch["pixel_values_pose"]  # (bs, f, c, H, W)
                pixel_values_pose = pixel_values_pose.transpose(1, 2)  # (bs, c, f, H, W)

                camera_pose = batch['plucker_embedding']  # (bs, f, c, H, W)
                camera_pose = camera_pose.transpose(1, 2)  # (bs, c, f, H, W)

                uncond_fwd = random.random() < cfg.uncond_ratio
                clip_image_list = []
                ref_image_list = []
                for batch_idx, (ref_img, clip_img) in enumerate(zip(batch["pixel_values_ref_img"], batch["clip_ref_img"],)):
                    if uncond_fwd:
                        clip_image_list.append(torch.zeros_like(clip_img))
                    else:
                        clip_image_list.append(clip_img)
                    ref_image_list.append(ref_img)

                with torch.no_grad():
                    ref_img = torch.stack(ref_image_list, dim=0).to(dtype=vae.dtype, device=vae.device)
                    ref_image_latents = vae.encode(ref_img).latent_dist.sample()  # (bs, d, 64, 64)
                    ref_image_latents = ref_image_latents * 0.18215
                    clip_img = torch.stack(clip_image_list, dim=0).to(dtype=image_enc.dtype, device=image_enc.device)
                    clip_img = clip_img.to(device="cuda", dtype=weight_dtype)
                    clip_image_embeds = image_enc(clip_img.to("cuda", dtype=weight_dtype)).image_embeds
                    clip_image_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d)

                # add noise
                noisy_latents = train_noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the target for loss depending on the prediction type
                if train_noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif train_noise_scheduler.prediction_type == "v_prediction":
                    target = train_noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {train_noise_scheduler.prediction_type}")

                # ---- Forward!!! -----
                model_pred = net(noisy_latents, timesteps, ref_image_latents, clip_image_embeds, pixel_values_pose, camera_pose, uncond_fwd=uncond_fwd, )

                if cfg.snr_gamma == 0:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    snr = compute_snr(train_noise_scheduler, timesteps)
                    if train_noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (torch.stack([snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr)
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = (loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights)
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, cfg.solver.max_grad_norm,)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    wandb.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                if global_step %  cfg.checkpointing_steps == 0:
                    # save model after each epoch
                    if accelerator.is_main_process:
                        #save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                        #delete_additional_ckpt(save_dir, 1)
                        #accelerator.save_state(save_path)
                        # save motion module only
                        unwrap_net = accelerator.unwrap_model(net)
                        save_checkpoint(unwrap_net.denoising_unet, save_dir, "motion_module", global_step, total_limit=10,)
                        #save_checkpoint(    unwrap_net.pose_guider, save_dir, "pose_guider", global_step, total_limit=3,)
                        save_checkpoint(unwrap_net.camera_pose_encoder, save_dir, "camera_pose_encoder", global_step, total_limit=10,)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "td": f"{t_data:.2f}s", }
            t_data_start = time.time()
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break
        
        
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/training/stage2.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    main(config)