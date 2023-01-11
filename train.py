import os
import math

import wandb
import torch
from torch import nn
from tqdm.auto import tqdm
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

from data import train_val_test_dataloaders
from output import array_to_plot, array_to_midi, midi_to_mp3
from util import define_args, get_args

with define_args():
    from data import BATCH_SIZE, WIDTH, HEIGHT

    EVAL_BATCH_SIZE = 4
    MIXED_PRECISION = "no"
    LEARNING_RATE = 1e-4
    ADAM_BETA1 = 0.95
    ADAM_BETA2 = 0.999
    ADAM_WEIGHT_DECAY = 1e-6
    ADAM_EPSILON = 1e-08
    LR_SCHEDULER = "cosine"
    LR_WARMUP_STEPS = 500
    NUM_EPOCHS = 100000
    GRADIENT_ACCUMULATION_STEPS = 10
    USE_EMA = True
    EMA_INV_GAMMA = 1.0
    EMA_POWER = 3 / 4
    EMA_MAX_DECAY = 0.9999
    SAVE_MEDIA_EPOCHS = 100
    SAVE_MODEL_EPOCHS = 200
    OUTPUT_DIR = "checkpoints"
    RESUME_FROM = "checkpoints-570000"
    TRAIN_INPAINTER = True
    SAMPLE_COUNT = 4


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if RESUME_FROM:
        model = UNet2DModel.from_pretrained(RESUME_FROM + "/unet").to("cuda")

        if model.conv_in.weight.shape[1] == 4 and TRAIN_INPAINTER:
            new_conv = nn.Conv2d(12, 128, kernel_size=3, padding=(1, 1)).to("cuda")
            new_conv.weight.data[:, :4, :, :] = model.conv_in.weight
            new_conv.bias.data = model.conv_in.bias
            model.conv_in = new_conv
    else:
        model = UNet2DModel(
            sample_size=64,
            in_channels=12 if TRAIN_INPAINTER else 4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        ).to("cuda")

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(ADAM_BETA1, ADAM_BETA2),
        weight_decay=ADAM_WEIGHT_DECAY,
        eps=ADAM_EPSILON,
    )

    train_dl, test_dl, val_dl = train_val_test_dataloaders()

    lr_scheduler = get_scheduler(
        LR_SCHEDULER,
        optimizer=optimizer,
        num_warmup_steps=LR_WARMUP_STEPS,
        num_training_steps=(len(train_dl) * NUM_EPOCHS) // GRADIENT_ACCUMULATION_STEPS,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dl) / GRADIENT_ACCUMULATION_STEPS)

    ema_model = EMAModel(
        model, inv_gamma=EMA_INV_GAMMA, power=EMA_POWER, max_value=EMA_MAX_DECAY
    )

    print("args", get_args())
    wandb_run = wandb.init(project="cantable-diffuguesion", config=get_args())

    pipeline = DDPMPipeline(
        unet=ema_model.averaged_model if USE_EMA else model,
        scheduler=noise_scheduler,
    ).to("cuda")

    global_step = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
        )
        progress_bar.set_description(f"Epoch {epoch}")

        # Generate sample images for visual inspection
        if epoch % SAVE_MEDIA_EPOCHS == 0:
            log_media(
                model,
                noise_scheduler,
                global_step,
                val_dl,
            )

        if epoch % SAVE_MODEL_EPOCHS == 0:
            # save the model
            pipeline.save_pretrained(OUTPUT_DIR)
            artifact = wandb.Artifact("checkpoints", type="checkpoints")
            artifact.add_dir("checkpoints")  # Adds multiple files to artifact
            wandb_run.log_artifact(artifact)

        for step, batch in enumerate(train_dl):
            clean_images = batch.to("cuda")

            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bsz = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=clean_images.device,
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            if TRAIN_INPAINTER:
                mask = random_mask(clean_images.shape[0])
                masked_images = clean_images.clone()
                masked_images[~mask] *= 0
                model_input = torch.cat([noisy_images, mask, masked_images], dim=1)
            else:
                model_input = noisy_images

            # Predict the noise residual
            noise_pred = model(model_input, timesteps).sample
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            if USE_EMA:
                ema_model.step(model)
            optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1

            logs = {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
            if USE_EMA:
                logs["ema_decay"] = ema_model.decay

            if global_step % 10 == 0:
                wandb.log(logs, step=global_step)

            progress_bar.set_postfix(**logs)

    progress_bar.close()
    log_media(
        ema_model.averaged_model if USE_EMA else model,
        noise_scheduler,
        global_step,
        val_dl,
    )
    pipeline.save_pretrained(OUTPUT_DIR)
    wandb.save(OUTPUT_DIR)


@torch.no_grad()
def sample(model, noise_scheduler, mask=None, masked_images=None):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    num_inference_steps = 1000
    noise_scheduler = DDPMScheduler(num_train_timesteps=num_inference_steps)
    image = torch.randn(
        (SAMPLE_COUNT, model.in_channels, WIDTH, model.sample_size),
        generator=generator,
        device="cuda",
    )
    noise_scheduler.set_timesteps(num_inference_steps)

    for t in noise_scheduler.timesteps:
        # 1. predict noise model_output
        if TRAIN_INPAINTER:
            model_input = torch.cat([image.to("cuda"), mask.to("cuda"), masked_images.to("cuda")], dim=1).to("cuda")
        else:
            model_input = image
        model_output = model(model_input, t).sample

        # 2. compute previous image: x_t -> x_t-1
        image = noise_scheduler.step(model_output, t, image).prev_sample

    return (image / 2 + 0.5).clamp(0, 1)


def random_mask(count):
    mask = torch.zeros([count, 4, WIDTH, HEIGHT], dtype=bool).to("cuda")
    for i in range(count):
        for c in range(4):
            start_index, end_index = torch.randint(0, WIDTH, (2,))
            if start_index > end_index:
                start_index, end_index = end_index, start_index
            mask[i, c, start_index:end_index] = True
    return mask


def log_media(model, noise_scheduler, global_step, val_dl=None):
    logs = {}
    # run pipeline in inference (sample random noise and denoise)
    if TRAIN_INPAINTER:
        mask = random_mask(SAMPLE_COUNT)
        masked_images = torch.zeros_like(mask, dtype=torch.float32).to("cuda")
        for i, batch in enumerate(val_dl):
            if i == SAMPLE_COUNT:
                break
            masked_images[i] = batch[0, :, :WIDTH]
        masked_images[~mask] *= 0
        logs["masked_inputs"] = [wandb.Image(array_to_plot(i)) for i in masked_images]

        arrays = sample(model, noise_scheduler, mask.to("cuda"), masked_images.to("cuda"))
    else:
        arrays = sample(model, noise_scheduler)
    #arrays = torch.from_numpy(arrays).permute([0, 3, 1, 2])

    plots = [array_to_plot(a) for a in arrays]
    midis = [array_to_midi(a, f"/tmp/midi-output-{i}.mid") for i, a in enumerate(arrays)]
    audios = [
        midi_to_mp3(midi, f"/tmp/audio-output-{i}.mp3") for i, midi in enumerate(midis)
    ]

    logs["plots"] = [wandb.Image(plot) for plot in plots]
    logs["audios"] = [wandb.Audio(audio) for audio in audios]

    wandb.log(logs, step=global_step)


if __name__ == "__main__":
    main()
