import argparse
import logging
import math
import os
import os.path as osp
import random
import warnings
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import diffusers
import mlflow
import numpy as np
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
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection
from einops import rearrange
from collections import defaultdict

from src.dataset.dance_image_h_v_camera_depth_normal import HumanDanceCameraDataset, OrientationBatchSampler, OrientationSampler
from src.dwpose import DWposeDetector
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.cameractrl.pose_adaptor import CameraPoseEncoder
from src.pipelines.pipeline_pose2img import Pose2ImagePipeline
from src.utils.util import delete_additional_ckpt, import_filename, seed_everything, save_checkpoint
import wandb
import time
from datetime import timedelta
warnings.filterwarnings("ignore")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")


class Net(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
        camera_pose_encoder: CameraPoseEncoder,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader
        self.camera_pose_encoder = camera_pose_encoder

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        pose_img,
        camera_embedding,
        uncond_fwd: bool = False,
    ):
        # camera pose
        assert camera_embedding.ndim == 5
        b = camera_embedding.shape[0]
        camera_embedding_feature = self.camera_pose_encoder(camera_embedding)[0]      # bf c h w
        camera_embedding_feature = rearrange(camera_embedding_feature, '(b f) c h w -> b c f h w', b=b)
        
        # human pose
        pose_cond_tensor = pose_img.to(device="cuda")
        pose_fea = self.pose_guider(pose_cond_tensor)
        
        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            pose_cond_fea = pose_fea + camera_embedding_feature,
            encoder_hidden_states=clip_image_embeds,
        ).sample

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
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def log_validation(
    vae,
    image_enc,
    net,
    scheduler,
    accelerator,
    width,
    height,
):
    logger.info("Running validation... ")

    ori_net = accelerator.unwrap_model(net)
    reference_unet = ori_net.reference_unet
    denoising_unet = ori_net.denoising_unet
    pose_guider = ori_net.pose_guider

    generator = torch.manual_seed(42)
    # cast unet dtype
    vae = vae.to(dtype=torch.float32)
    image_enc = image_enc.to(dtype=torch.float32)

    pose_detector = DWposeDetector()
    pose_detector.to(accelerator.device)

    pipe = Pose2ImagePipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to(accelerator.device)

    ref_image_paths = [
        "./configs/inference/ref_images/anyone-2.png",
        "./configs/inference/ref_images/anyone-3.png",
    ]
    pose_image_paths = [
        "./configs/inference/pose_images/pose-1.png",
        "./configs/inference/pose_images/pose-1.png",
    ]

    pil_images = []
    for ref_image_path in ref_image_paths:
        for pose_image_path in pose_image_paths:
            pose_name = pose_image_path.split("/")[-1].replace(".png", "")
            ref_name = ref_image_path.split("/")[-1].replace(".png", "")
            ref_image_pil = Image.open(ref_image_path).convert("RGB")
            pose_image_pil = Image.open(pose_image_path).convert("RGB")

            image = pipe(
                ref_image_pil,
                pose_image_pil,
                width,
                height,
                20,
                3.5,
                generator=generator,
            ).images
            image = image[0, :, 0].permute(1, 2, 0).cpu().numpy()  # (3, 512, 512)
            res_image_pil = Image.fromarray((image * 255).astype(np.uint8))
            # Save ref_image, src_image and the generated_image
            w, h = res_image_pil.size
            canvas = Image.new("RGB", (w * 3, h), "white")
            ref_image_pil = ref_image_pil.resize((w, h))
            pose_image_pil = pose_image_pil.resize((w, h))
            canvas.paste(ref_image_pil, (0, 0))
            canvas.paste(pose_image_pil, (w, 0))
            canvas.paste(res_image_pil, (w * 2, 0))

            pil_images.append({"name": f"{ref_name}_{pose_name}", "img": canvas})

    vae = vae.to(dtype=torch.float16)
    image_enc = image_enc.to(dtype=torch.float16)

    del pipe
    torch.cuda.empty_cache()

    return pil_images


def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with="mlflow",
        project_dir="./mlruns",
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
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
    if accelerator.is_main_process and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if accelerator.is_main_process:
        wandb.init(project="Animate-with-camera train stage 1")

    if cfg.solver.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif cfg.solver.mixed_precision == "no":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)

    # base model
    vae = AutoencoderKL.from_pretrained(
        cfg.vae_model_path,
    ).to("cuda", dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device="cuda") # dtype=weight_dtype,

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
        },
    ).to(device="cuda")  # dtype=weight_dtype,

    # human pose
    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(device="cuda") # dtype=weight_dtype,

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path
    ).to(dtype=weight_dtype, device="cuda")

    # camera pose
    camera_pose_encoder_kwargs = OmegaConf.to_container(cfg.pose_encoder_kwargs)
    camera_pose_encoder = CameraPoseEncoder(**camera_pose_encoder_kwargs)
    if cfg.camera_pose_encoder_path != "":
        if "CameraCtrl.ckpt" in cfg.camera_pose_encoder_path:
            camera_pose_encoder.load_state_dict(
                torch.load(cfg.camera_pose_encoder_path, map_location="cpu")['pose_encoder_state_dict'],
                strict=False,
            )
        else:
            camera_pose_encoder.load_state_dict(
                torch.load(cfg.camera_pose_encoder_path, map_location="cpu"),
                strict=False,
            )
    
    if cfg.denoising_unet_path != "":
        # load pretrained weights
        denoising_unet.load_state_dict(
            torch.load(cfg.denoising_unet_path, map_location="cpu"),
            strict=False,
        )
    if cfg.reference_unet_path != "":
        reference_unet.load_state_dict(
            torch.load(cfg.reference_unet_path, map_location="cpu"),
        )
        
    pose_guider.load_state_dict(
        torch.load(cfg.pose_guider_path, map_location="cpu"),
    )


    if cfg.resume_from_checkpoint != "":
        stage1_ckpt_dir = cfg.resume_from_checkpoint
        stage1_ckpt_step = cfg.resume_step
        denoising_unet.load_state_dict(torch.load(os.path.join(stage1_ckpt_dir, f"denoising_unet-{stage1_ckpt_step}.pth"),map_location="cpu",),strict=False,)
        camera_pose_encoder.load_state_dict(torch.load(os.path.join(stage1_ckpt_dir, f"camera_pose_encoder-{stage1_ckpt_step}.pth"),map_location="cpu",),strict=False,)
        reference_unet.load_state_dict(torch.load(os.path.join(stage1_ckpt_dir, f"reference_unet-{stage1_ckpt_step}.pth"),map_location="cpu",),strict=True,)
        pose_guider.load_state_dict(torch.load(os.path.join(stage1_ckpt_dir, f"pose_guider-{stage1_ckpt_step}.pth"),map_location="cpu",),strict=True,)

    # Freeze
    vae.requires_grad_(False)
    image_enc.requires_grad_(False)

    # Explictly declare training models
    denoising_unet.requires_grad_(True)
    #  Some top layer parames of reference_unet don't need grad
    for name, param in reference_unet.named_parameters():
        if "up_blocks.3" in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

    pose_guider.requires_grad_(True)
    camera_pose_encoder.requires_grad_(True)

    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )

    net = Net(
        reference_unet,
        denoising_unet,
        pose_guider,
        reference_control_writer,
        reference_control_reader,
        camera_pose_encoder,
    )

    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

    # Initialize the optimizer
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

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

    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )

    train_dataset = HumanDanceCameraDataset(
        train_size_large=cfg.data.train_size_large,
        train_size_small=cfg.data.train_size_small,
        img_scale=(1.0, 1.1),
        horizontal_meta=cfg.data.horizontal_meta,
        vertical_meta = cfg.data.vertical_meta,
        sample_margin=cfg.data.sample_margin,
    )

    # Create the custom batch sampler
    horizontal_indices = list(range(len(train_dataset.horizontal_meta)))
    vertical_indices = list(range(len(train_dataset.horizontal_meta), len(train_dataset)))
    orientation_sampler = OrientationSampler(horizontal_indices, vertical_indices)
    batch_sampler = OrientationBatchSampler(orientation_sampler, batch_size=cfg.data.train_bs, rank = accelerator.process_index)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=4,  # Adjust the number of workers as needed
    )

    # Prepare everything with our `accelerator`.
    (
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            cfg.exp_name,
            init_kwargs={"mlflow": {"run_name": run_time}},
        )
        # dump config file
        mlflow.log_dict(OmegaConf.to_container(cfg), "config.yaml")

    # Train!
    total_batch_size = (
        cfg.data.train_bs
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.data.train_bs}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")
    # Initialize timers and logging variables
    timers = defaultdict(float)
    iteration_count = 0
    train_loss = 0.0
    log_step = 20  # Log every n steps

    global_step = 0
    if cfg.resume_from_checkpoint != "":
        global_step = cfg.resume_step + 1

    data_start_time = time.time()

    while global_step < cfg.solver.max_train_steps:
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(net):
                # Data Loading Timer
                pixel_values = batch["tgt_img"].to(weight_dtype)
                data_loading_end = time.time()
                timers['data'] += data_loading_end - data_start_time

                # VAE Inference Timer
                vae_start = time.time()
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents.unsqueeze(2)  # (b, c, 1, h, w)
                    latents = latents * 0.18215

                noise = torch.randn_like(latents)
                if cfg.noise_offset > 0.0:
                    noise += cfg.noise_offset * torch.randn(
                        (noise.shape[0], noise.shape[1], 1, 1, 1),
                        device=noise.device,
                    )

                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    train_noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                vae_end = time.time()
                timers['vae'] += vae_end - vae_start

                # Forward Timer
                forward_start = time.time()
                tgt_pose_img = batch["tgt_pose"].unsqueeze(2)  # (bs, 3, 1, 512, 512)
                camera_pose = batch['plucker_embedding'].unsqueeze(2)  # (bs, 3, 1, 512, 512)

                uncond_fwd = random.random() < cfg.uncond_ratio
                clip_image_list = []
                ref_image_list = []
                for ref_img, clip_img in zip(batch["ref_img"], batch["clip_images"]):
                    if uncond_fwd:
                        clip_image_list.append(torch.zeros_like(clip_img))
                    else:
                        clip_image_list.append(clip_img)
                    ref_image_list.append(ref_img)

                with torch.no_grad():
                    ref_img = torch.stack(ref_image_list, dim=0).to(
                        dtype=vae.dtype, device=vae.device
                    )
                    ref_image_latents = vae.encode(ref_img).latent_dist.sample()  # (bs, d, 64, 64)
                    ref_image_latents = ref_image_latents * 0.18215

                    clip_img = torch.stack(clip_image_list, dim=0).to(
                        dtype=image_enc.dtype, device=image_enc.device
                    )
                    clip_image_embeds = image_enc(
                        clip_img.to("cuda", dtype=weight_dtype)
                    ).image_embeds
                    image_prompt_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d)

                noisy_latents = train_noise_scheduler.add_noise(
                    latents, noise, timesteps
                )

                if train_noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif train_noise_scheduler.prediction_type == "v_prediction":
                    target = train_noise_scheduler.get_velocity(
                        latents, noise, timesteps
                    )
                else:
                    raise ValueError(
                        f"Unknown prediction type {train_noise_scheduler.prediction_type}"
                    )

                model_pred = net(
                    noisy_latents,
                    timesteps,
                    ref_image_latents,
                    image_prompt_embeds,
                    tgt_pose_img,
                    camera_pose,
                    uncond_fwd,
                )

                if cfg.snr_gamma == 0:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    snr = compute_snr(train_noise_scheduler, timesteps)
                    if train_noise_scheduler.config.prediction_type == "v_prediction":
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                    )
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()

                

                # Gather losses for distributed training
                avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                forward_end = time.time()
                timers['forward'] += forward_end - forward_start

                # Backpropagation Timer
                backward_start = time.time()

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        cfg.solver.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                backward_end = time.time()
                timers['backward'] += backward_end - backward_start

            # End of gradient accumulation
            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                global_step += 1
                iteration_count += 1

                # Logging every `log_step` steps
                if global_step % log_step == 0 and accelerator.is_main_process:
                    if iteration_count > 0:
                        avg_data_loading = timers['data'] / iteration_count
                        avg_vae_inference = timers['vae'] / iteration_count
                        avg_forward = timers['forward'] / iteration_count
                        avg_backward = timers['backward'] / iteration_count
                        # Add other timers if necessary
                        train_loss /= iteration_count
                    else:
                        avg_data_loading = avg_vae_inference = avg_forward = avg_backward = 0.0

                    current_lr = optimizer.param_groups[0]['lr']
                    avg_step_time = avg_data_loading + avg_vae_inference + avg_forward + avg_backward

                    # Calculate ETA
                    remaining_steps = cfg.solver.max_train_steps - global_step
                    eta_seconds = remaining_steps * avg_step_time

                    # Format ETA
                    eta = str(timedelta(seconds=int(eta_seconds)))
                    logger.info(
                        f"Step: {global_step}/{cfg.solver.max_train_steps} "
                        f"Loss: {train_loss:.4f} "
                        # Add other loss components if available
                        f"Lr: {current_lr:.2e} "
                        f"ETA: {eta} "
                        f"[Time]: Data: {avg_data_loading:.3f}s, "
                        f"VAE: {avg_vae_inference:.3f}s, "
                        f"Forward: {avg_forward:.3f}s, "
                        f"Back: {avg_backward:.3f}s"
                    )

                    # Log to wandb
                    wandb.log({
                        "train_loss": train_loss,
                        # Add other loss components if available
                        "learning_rate": current_lr,
                        "avg_data_time": avg_data_loading,
                        "avg_vae_time": avg_vae_inference,
                        "avg_forward_time": avg_forward,
                        "avg_backward_time": avg_backward,
                        "eta": eta,
                    }, step=global_step)

                    # Reset timers and training loss
                    timers = defaultdict(float)
                    iteration_count = 0
                    train_loss = 0.0

                # Checkpointing
                if global_step % cfg.checkpointing_steps == 0 and accelerator.is_main_process:
                    unwrap_net = accelerator.unwrap_model(net)
                    save_checkpoint(unwrap_net.reference_unet, save_dir, "reference_unet", global_step, total_limit=3, logger=logger)
                    save_checkpoint(unwrap_net.denoising_unet, save_dir, "denoising_unet", global_step, total_limit=3, logger=logger)
                    save_checkpoint(unwrap_net.pose_guider, save_dir, "pose_guider", global_step, total_limit=3, logger=logger)
                    save_checkpoint(unwrap_net.camera_pose_encoder, save_dir, "camera_pose_encoder", global_step, total_limit=3, logger=logger)

                # Validation
                if global_step % cfg.val.validation_steps == 0 and accelerator.is_main_process:
                    generator = torch.Generator(device=accelerator.device)
                    generator.manual_seed(cfg.seed)

                    sample_dicts = log_validation(
                        vae=vae,
                        image_enc=image_enc,
                        net=net,
                        scheduler=val_noise_scheduler,
                        accelerator=accelerator,
                        width=cfg.data.train_width,
                        height=cfg.data.train_height,
                    )

                    for sample_id, sample_dict in enumerate(sample_dicts):
                        sample_name = sample_dict["name"]
                        img = sample_dict["tgt_img"]
                        with TemporaryDirectory() as temp_dir:
                            out_file = Path(
                                f"{temp_dir}/{global_step:06d}-{sample_name}.gif"
                            )
                            img.save(out_file)
                            mlflow.log_artifact(out_file)

            data_start_time = time.time()
            if global_step >= cfg.solver.max_train_steps:
                break

    # Finalize training
    accelerator.wait_for_everyone()
    accelerator.end_training()
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/training/stage1.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    main(config)
