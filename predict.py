import random
from typing import Optional
import sys
import torch
import torch.nn.functional as F
import music21
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from cog import BaseModel, BasePredictor, Input, Path

from data import RESOLUTION, MIN_PITCH, MAX_PITCH
from output import array_to_midi, midi_to_score, midi_to_mp3


class Output(BaseModel):
    mp3: Optional[Path]
    #score: Optional[Path]
    midi: Optional[Path]


class Predictor(BasePredictor):
    def setup(self):
        self.model = model = UNet2DModel.from_pretrained("checkpoints/unet").to("cuda")

    def predict(
        self,
        duration: int = Input(
            description="Duration in quarter notes",
            choices=(
                64 // RESOLUTION,
                128 // RESOLUTION,
                256 // RESOLUTION,
                512 // RESOLUTION,
                1024 // RESOLUTION,
            ),
            default=128 // RESOLUTION,
        ),
        tempo: float = Input(
            description="Tempo in quarter notes per minute", default=90, ge=40, le=200
        ),
        melody: str = Input(
            description="Melody in tinyNotation format. Accepts ? for inpainting a single note, and ?* for inpainting between two melodic parts",
            default="",
        ),
        # not working :(
        # return_score: bool = Input(
        #     description="Return sheet music score", default=True
        # ),
        return_mp3: bool = Input(description="Return mp3 audio", default=True),
        return_midi: bool = Input(description="Return midi", default=True),
        seed: int = Input(description="Random seed. Random if seed == -1", default=-1),
    ) -> Output:
        num_outputs = 1

        #if not return_score and not return_mp3 and not return_midi:
        if not return_mp3 and not return_midi:
            raise Exception(
                "At least one of return_score, return_mp3, return_midi must be true"
            )

        if seed == -1:
            seed = random.randint(0, 100000)

        length = duration * RESOLUTION

        if melody:
            mel_inputs, mel_mask, length = parse_melody(melody, length, num_outputs)
        else:
            mel_inputs = torch.zeros([num_outputs, 4, length, 64]).to("cuda")
            mel_mask = torch.zeros_like(mel_inputs, dtype=torch.bool).to("cuda")

        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        array = sample(self.model, generator, mel_inputs, mel_mask, 1000, length)[0]
        midi = array_to_midi(array, "/tmp/midi.mid", tempo=tempo)

        output = Output(
            midi=Path(midi) if return_midi else None,
            #score=Path(midi_to_score(midi, "/tmp/score.png")) if return_score else None,
            mp3=Path(midi_to_mp3(midi, "/tmp/audio.mp3")) if return_mp3 else None,
        )

        return output


def parse_melody(text, length, num_outputs):
    text = text.replace("|", "")
    if "?*" in text:
        if text.count("?*") > 1:
            raise Exception("Can only have on '?*' in the input")
        text1, text2 = text.split("?*")
        mel_inputs1, mel_mask1 = parse_notes(text1, num_outputs)
        mel_inputs2, mel_mask2 = parse_notes(text2, num_outputs)

        notes_length = mel_inputs1.shape[2] + mel_inputs2.shape[2]
        if notes_length > length:
            length = notes_length
        mel_inputs = torch.zeros([num_outputs, 4, length, 64]).to("cuda")
        mel_mask = torch.zeros_like(mel_inputs, dtype=torch.bool).to("cuda")

        mel_inputs[:, :, : mel_inputs1.shape[2]] = mel_inputs1
        mel_mask[:, :, : mel_mask1.shape[2]] = mel_mask1
        mel_inputs[:, :, -mel_inputs2.shape[2] :] = mel_inputs2
        mel_mask[:, :, -mel_mask2.shape[2] :] = mel_mask2
    else:
        mel_inputs1, mel_mask1 = parse_notes(text, num_outputs)

        notes_length = mel_inputs1.shape[2]
        if notes_length > length:
            length = notes_length
        mel_inputs = torch.zeros([num_outputs, 4, length, 64]).to("cuda")
        mel_mask = torch.zeros_like(mel_inputs, dtype=torch.bool).to("cuda")

        mel_inputs[:, :, : mel_inputs1.shape[2]] = mel_inputs1
        mel_mask[:, :, : mel_mask1.shape[2]] = mel_mask1

    if length % 64 != 0:
        new_length = length - (length % 64) + 64
        pad = new_length - length
        mel_inputs = F.pad(mel_inputs, (0, 0, 0, pad), "constant", 0)
        mel_mask = F.pad(mel_mask, (0, 0, 0, pad), "constant", True)

    return mel_inputs, mel_mask, length


def parse_notes(text, num_outputs):
    text = text.replace("?", "CC")  # hack for inpainting masks

    notes = music21.converter.parse("tinyNotation: 4/4 " + text).flat.notes
    if len(notes) > 0:
        mel_length = int((notes[-1].offset + notes[-1].duration.quarterLength) * RESOLUTION)
    else:
        mel_length = 0

    mel_inputs = torch.zeros([num_outputs, 4, mel_length, 64]).to("cuda")
    mel_mask = torch.zeros_like(mel_inputs, dtype=torch.bool).to("cuda")

    for note in notes:
        if note.pitch.midi != 36:  # == CC == "?"
            pitch = note.pitch.midi + 12
            if pitch < MIN_PITCH:
                raise Exception(f"Pitch is too low: {note}")
            if pitch > MAX_PITCH:
                raise Exception(f"Pitch is too high: {note}")
            start_index = int(note.offset * RESOLUTION)
            end_index = int(
                note.offset * RESOLUTION + note.duration.quarterLength * RESOLUTION
            )
            mel_inputs[0, 0, start_index:end_index, pitch - MIN_PITCH] = 1
            mel_mask[:, 0, start_index:end_index] = True

            # staccato
            mel_inputs[0, 0, start_index - 1, pitch - MIN_PITCH] = 0

    return mel_inputs, mel_mask


@torch.no_grad()
def sample(model, generator, inputs, mask, num_inference_steps, length):
    num_outputs = inputs.shape[0]
    length = inputs.shape[2]
    noise_scheduler = DDPMScheduler(num_train_timesteps=num_inference_steps)
    image = torch.randn(
        (num_outputs, 4, length, model.sample_size),
        generator=generator,
        device="cuda",
    )
    noise_scheduler.set_timesteps(num_inference_steps)

    for t in noise_scheduler.timesteps:
        model_input = torch.cat([image, mask, inputs], dim=1)
        model_output = model(model_input, t).sample

        image = noise_scheduler.step(
            model_output, t, image, generator=generator
        ).prev_sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image[mask] = inputs[mask]
    return image[:, :, :length]
