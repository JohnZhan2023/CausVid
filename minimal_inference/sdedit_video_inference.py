from causvid.models.wan.causal_inference import InferencePipeline
from diffusers.utils import export_to_video
import numpy as np
import torch
import cv2
import os
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
from typing import List, Optional
def resize_video(video, target_height=480):
    """Resize video to target height while maintaining aspect ratio."""
    target_width = 832  # 固定宽度为832
    
    resized_frames = []
    for frame in video:
        resized_frame = cv2.resize(frame, (target_width, target_height))
        resized_frames.append(resized_frame)
    
    return np.stack(resized_frames)

def load_video(video_path):
    """Load video and convert to tensor format."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    original_size = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if original_size is None:
            original_size = (frame.shape[1], frame.shape[0])  # (width, height)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    # Convert to numpy array and resize to 480p
    video = np.stack(frames)
    video = resize_video(video, target_height=480)
    
    # Convert to tensor [T, H, W, C]
    video = torch.from_numpy(video).float() / 255.0
    video = video * 2 - 1
    return video, original_size


def add_noise(pipeline, video, t):
    """Add noise to video using flow matching schedule."""
    noise = torch.randn_like(video)
    t = torch.ones(video.shape[0], device=video.device, dtype=torch.int64) * t
    noisy_video = pipeline.scheduler.add_noise(video, noise, t)
    return noisy_video

class SdeditVideoInference(InferencePipeline):
    def inference(self, noise: torch.Tensor, text_prompts: List[str], start_latents: Optional[torch.Tensor] = None, return_latents: bool = False, denoise_steps_list: Optional[List[int]] = None) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if denoise_steps_list is not None:
            self.denoising_step_list = denoise_steps_list

        output = torch.zeros(
            [batch_size, num_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )

            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False

        num_input_blocks = start_latents.shape[1] // self.num_frame_per_block if start_latents is not None else 0

        # Step 2: Temporal denoising loop
        num_blocks = num_frames // self.num_frame_per_block
        for block_index in range(num_blocks):
            noisy_input = noise[:, block_index *
                                self.num_frame_per_block:(block_index + 1) * self.num_frame_per_block]

            if start_latents is not None and block_index < num_input_blocks:
                timestep = torch.ones(
                    [batch_size, self.num_frame_per_block], device=noise.device, dtype=torch.int64) * 0

                current_ref_latents = start_latents[:, block_index * self.num_frame_per_block:(
                    block_index + 1) * self.num_frame_per_block]
                output[:, block_index * self.num_frame_per_block:(
                    block_index + 1) * self.num_frame_per_block] = current_ref_latents

                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                    current_end=(block_index + 1) *
                    self.num_frame_per_block * self.frame_seq_length
                )
                continue

            # Step 2.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                # set current timestep
                timestep = torch.ones(
                    [batch_size, self.num_frame_per_block], device=noise.device, dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                        current_end=(
                            block_index + 1) * self.num_frame_per_block * self.frame_seq_length
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep *
                        torch.ones([batch_size], device="cuda",
                                   dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                        current_end=(
                            block_index + 1) * self.num_frame_per_block * self.frame_seq_length
                    )

            # Step 2.2: rerun with timestep zero to update the cache
            output[:, block_index * self.num_frame_per_block:(
                block_index + 1) * self.num_frame_per_block] = denoised_pred

            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                current_end=(block_index + 1) *
                self.num_frame_per_block * self.frame_seq_length
            )

        # Step 3: Decode the output
        print("[Debug] VAE解码前的output值范围:", output.min().item(), output.max().item())
        video = self.vae.decode_to_pixel(output)
        print("[Debug] VAE解码后的video值范围:", video.min().item(), video.max().item())
        video = (video * 0.5 + 0.5).clamp(0, 1)
        print("[Debug] 归一化后的video值范围:", video.min().item(), video.max().item())

        if return_latents:
            return video, output
        else:
            return video

def retrieve_timesteps(sdedit_strength, denoise_steps_list):
    """
    Retrieve the timesteps for the denoising steps.
    """
    # 找到最接近sdedit_strength的值的索引
    denoise_steps = torch.tensor(denoise_steps_list)
    index = (denoise_steps - sdedit_strength).abs().argmin()
    # turn to tensor
    sdedit_strength = torch.tensor(sdedit_strength/1000.0)
    return sdedit_strength, denoise_steps_list[index:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--reference_video", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--noise_level", type=float, default=0.5)
    
    args = parser.parse_args()
    
    # Disable gradient computation
    torch.set_grad_enabled(False)
    
    # Load config and setup pipeline
    config = OmegaConf.load(args.config_path)
    pipeline = SdeditVideoInference(config, device="cuda")
    pipeline.to(device="cuda", dtype=torch.bfloat16)
    
    # Load checkpoint
    state_dict = torch.load(os.path.join(args.checkpoint_folder, "model.pt"), map_location="cpu")['generator']
    pipeline.generator.load_state_dict(state_dict, strict=True)
    
    # Load reference video
    video, original_size = load_video(args.reference_video)
    print("[Debug] 原始视频加载后的值范围:", video.min().item(), video.max().item())
    
    # Move video to CUDA and convert to [B, C, T, H, W] format
    video = video.permute(0, 3, 1, 2).unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)
    print("[Debug] 转换格式后的值范围:", video.min().item(), video.max().item())
    
    # Use VAE to encode video
    with torch.no_grad():
        device, dtype = video.device, video.dtype
        video = video.permute(0, 2, 1, 3, 4)
        scale = [pipeline.vae.mean.to(device=device, dtype=dtype),
                 1.0 / pipeline.vae.std.to(device=device, dtype=dtype)]
        print("[Debug] VAE scale值:", scale)
        latents = pipeline.vae.model.encode(video, scale).float()
        print("[Debug] VAE编码后latents的值范围:", latents.min().item(), latents.max().item())
        latents = latents.permute(0, 2, 1, 3, 4)

    # Assert latent video has 12 frames
    assert latents.shape[1] == 12, f"Expected 12 frames in latent space, but got {latents.shape[1]}"
    
    # Add noise using flow matching schedule
    t, denoise_steps_list = retrieve_timesteps(args.noise_level, pipeline.denoising_step_list)
    noisy_latents = add_noise(pipeline, latents, t).to(device="cuda", dtype=torch.bfloat16)
    print("[Debug] 添加噪声后的值范围:", noisy_latents.min().item(), noisy_latents.max().item())
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Run inference
    prompts = [args.prompt]
    import time
    start_time = time.time()
    print(denoise_steps_list)
    video = pipeline.inference(
        noise=noisy_latents,
        text_prompts=prompts,
        denoise_steps_list=denoise_steps_list
    )[0]
    end_time = time.time()
    print(f"Time taken for inference: {end_time - start_time:.2f} seconds")
    print("[Debug] 推理后的视频值范围:", video.min().item(), video.max().item())
    
    # Convert to correct format for saving
    video = video.permute(0, 2, 3, 1).cpu().numpy()
    print("[Debug] 转换格式后的视频值范围:", video.min(), video.max())
    
    # Resize back to original dimensions
    resized_video = []
    for frame in video:
        resized_frame = cv2.resize(frame, original_size)
        resized_video.append(resized_frame)
    video = np.stack(resized_video)
    print("[Debug] 最终保存前的视频值范围:", video.min(), video.max())
    
    # Export video
    export_to_video(video, os.path.join(args.output_folder, "output.mp4"), fps=8)

if __name__ == "__main__":
    main() 