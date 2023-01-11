import os
from pathlib import Path
import subprocess
import tempfile
from PIL import Image
import torch
import pretty_midi as pm
import matplotlib.pyplot as plt
from midiSynth.synth import MidiSynth

from data import RESOLUTION, MIN_PITCH


PALETTE = torch.tensor(
    [
        [0, 0, 0],  # black
        [255, 0, 0],  # red
        [0, 255, 0],  # green
        [0, 0, 255],  # blue
        [255, 0, 255],  # magenta
        [255, 255, 0],  # yellow
        [255, 255, 255],  # white
    ]
)


def array_to_plot(arr):
    color_matrix = PALETTE[
        ((arr.cpu() > 0.75).permute([1, 2, 0]) * torch.arange(1, 5))
        .max(axis=2)[0]
        .T.to(int)
    ]
    resize = 0.1
    height, width, _ = color_matrix.shape
    figsize = width * resize, height * resize

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.imshow(color_matrix, origin="lower")
    fig.canvas.draw()
    print(fig.canvas.get_width_height())

    img = Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )
    plt.close(fig)
    return img


def array_to_plots(arr):
    fig, axs = plt.subplots(2, 2, figsize=[8, 8])
    for i in range(2):
        for j in range(2):
            ax = axs[i][j]
            ax.imshow(arr[i * 2 + j].T, origin="lower")
    fig.tight_layout()
    fig.canvas.draw()

    img = Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )
    plt.close(fig)
    return img


def midi_to_wav(midi_path, wav_path):
    MidiSynth().midi2audio(str(midi_path), str(wav_path))
    return str(wav_path)


def midi_to_mp3(midi_path, mp3_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = temp_dir + "/audio.wav"
        MidiSynth().midi2audio(str(midi_path), str(wav_path))
        subprocess.check_output(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(wav_path),
                "-af",
                "silenceremove=1:0:-50dB,aformat=dblp,areverse,silenceremove=1:0:-50dB,aformat=dblp,areverse",  # strip silence
                str(mp3_path),
            ],
        )
    return mp3_path


def midi_to_score(midi_path, score_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        lilypond_path = temp_dir + "/score.ly"
        subprocess.check_output(
            ["midi2ly", str(midi_path), "--output", str(lilypond_path)],
        )
        subprocess.check_output(
            [
                "lilypond",
                "-fpng",
                "-dresolution=300",
                '-dpaper-size="a5landscape"',
                "-dcrop",
                "-o",
                str(Path(score_path).with_suffix("")),
                str(lilypond_path),
            ]
        )
    cropped_path = str(Path(score_path).with_suffix("")) + ".cropped." + str(Path(score_path).suffix)
    if Path(cropped_path).exists():
        os.rename(cropped_path, str(score_path))
    return score_path


def array_to_midi(
    arr, midi_path, instrument_name="Lead 6 (voice)", tempo=90, time_sig=4
):
    sec_per_beat = 60 / tempo
    track = pm.PrettyMIDI(initial_tempo=tempo)
    track.time_signature_changes.append(pm.TimeSignature(time_sig, 4, 0))

    for mat in arr:
        instrument = pm.Instrument(pm.instrument_name_to_program(instrument_name))
        write_notes(instrument, mat, sec_per_beat)
        track.instruments.append(instrument)

    track.write(str(midi_path))
    return midi_path


def write_notes(instrument, mat, sec_per_beat):
    def append_note(pitch, start_beat, end_beat):
        note = pm.Note(
            pitch=pitch,
            velocity=120,
            start=start_beat * sec_per_beat,
            end=end_beat * sec_per_beat,
        )
        instrument.notes.append(note)

    cur_pitch = None
    start_beat = None
    for t, vec in enumerate(mat):
        beat = t / RESOLUTION

        max_i = int(torch.argmax(vec).item())
        if vec[max_i] > 0.75:
            pitch = max_i + MIN_PITCH
            if pitch != cur_pitch:
                if cur_pitch:
                    append_note(cur_pitch, start_beat, beat)
                cur_pitch = pitch
                start_beat = beat
            if start_beat is None:
                start_beat = beat
                cur_pitch = pitch
        else:
            if cur_pitch is not None:
                append_note(cur_pitch, start_beat, beat)
                start_beat = None
                cur_pitch = None

    if cur_pitch is not None:
        append_note(cur_pitch, start_beat, mat.shape[1] / RESOLUTION)
