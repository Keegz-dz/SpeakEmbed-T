""" This script demonstrates how to perform speaker diarization using the SpeechEncoder toolkit.
It extracts reference segments for known speakers from an audio interview, computes continuous
embeddings for the entire interview, and then compares the continuous embeddings with the
reference speaker embeddings using a cosine similarity.

The final output is an interactive animation that displays the similarity curves over time,
while playing the audio in sync with the plot.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import librosa
import torchaudio
import numpy as np
import matplotlib
from pathlib import Path
from typing import Union, List, Tuple

# Import necessary modules
from scripts.speech_encoder_v2 import SpeechEncoderV2
from scripts.params import *
from utils import *
from visualisations import *
from scripts.embed import Embed
from data_preprocessing import audio_preprocessing
from speaker_diarisation_utils import interactive_diarization

def prompt_audio_path(default_path: str) -> str:
    print("\n=== Speaker Diarization Tool ===\n")
    print(f"Default audio path: {default_path}")
    use_default = input("Do you want to use the default audio file? (Y/n): ").strip().lower()
    if use_default in ('', 'y', 'yes'):
        return default_path
    else:
        while True:
            user_path = input("Please enter the path to your audio file: ").strip()
            if os.path.isfile(user_path):
                return user_path
            else:
                print("File not found. Please enter a valid file path.")

def prompt_speaker_segments() -> Tuple[List[Tuple[float, float]], List[str]]:
    print("\nHow many speakers do you want to diarize?")
    while True:
        try:
            num_speakers = int(input("Enter the number of speakers: ").strip())
            if num_speakers < 1:
                print("Please enter a positive integer.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
    print("\nFor each speaker, you will be asked for their name and a reference segment from the audio.")
    print("Example: Start time: 0, End time: 5.5 (for a segment from 0s to 5.5s)")
    print("Please make sure the segment contains only that speaker's voice.\n")
    segments = []
    speaker_names = []
    for idx in range(1, num_speakers + 1):
        name = input(f"Enter name for speaker #{idx}: ").strip()
        while True:
            try:
                start = float(input(f"  Start time for {name} (in seconds, e.g., 0): ").strip())
                end = float(input(f"  End time for {name} (in seconds, e.g., 5.5): ").strip())
                if end <= start:
                    print("    End time must be greater than start time. Please try again.")
                    continue
                break
            except ValueError:
                print("    Invalid input. Please enter numeric values for start and end times.")
        segments.append((start, end))
        speaker_names.append(name)
    return segments, speaker_names

def load_audio_file(audio_path: str) -> Tuple[torch.Tensor, int]:
    """
    Load an audio file and return waveform and sample rate.
    """
    if not os.path.isfile(audio_path):
        print(f"Error: Audio file '{audio_path}' not found.")
        sys.exit(1)
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        print(f"Audio loaded successfully. Sample rate: {sample_rate}Hz, Duration: {waveform.shape[1]/sample_rate:.2f}s")
        return waveform, sample_rate
    except Exception as e:
        print(f"Error loading audio file: {e}")
        sys.exit(1)

def preprocess_waveform(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Convert to mono, resample, and apply audio preprocessing.
    """
    if waveform.shape[0] > 1:
        print("Converting stereo to mono...")
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate != sampling_rate:
        print(f"Resampling from {sample_rate}Hz to {sampling_rate}Hz...")
        waveform = torchaudio.transforms.Resample(sample_rate, sampling_rate)(waveform)
    print("Applying audio preprocessing...")
    wav = audio_preprocessing.preprocess_audio(waveform, sampling_rate)
    return wav

def extract_reference_segments(wav: torch.Tensor, segments: List[Tuple[float, float]], sr: int) -> List[torch.Tensor]:
    """
    Extract reference segments from the waveform.
    """
    speaker_wavs = []
    for i, (start, end) in enumerate(segments):
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        if start_sample >= wav.shape[0] or end_sample > wav.shape[0]:
            print(f"Warning: Segment {i+1} ({start}-{end}s) exceeds audio length. Adjusting...")
            end_sample = min(end_sample, wav.shape[0])
            start_sample = min(start_sample, end_sample-1)
        speaker_wavs.append(wav[start_sample:end_sample])
    return speaker_wavs

def load_speech_encoder(device: torch.device) -> SpeechEncoderV2:
    """
    Load the speech encoder model from the default path.
    """
    print("Loading speech encoder model...")
    encoder = SpeechEncoderV2(device, torch.device("cpu"))
    try:
        checkpoints = torch.load(
            "models/speech_encoder_transformer/encoder(0.096).pt",
            map_location=device
        )
        encoder.load_state_dict(checkpoints['model_state'])
        encoder.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    return encoder

def compute_embeddings(embedder: Embed, wav: torch.Tensor, sr: int) -> Tuple[torch.Tensor, List[slice]]:
    """
    Compute continuous embeddings for the full audio.
    """
    print("Computing continuous embeddings for the full audio...")
    embedder.encoder.use_nested_tensor = False
    if len(wav) > 10 * sr * 60:  # If longer than 10 minutes
        print("Audio is long, processing in chunks...")
        chunk_size = 5 * 60 * sr  # 5-minute chunks
        embeddings_list = []
        splits_list = []
        for i in range(0, len(wav), chunk_size):
            chunk_end = min(i + chunk_size, len(wav))
            print(f"Processing chunk {i/sr/60:.1f}m - {chunk_end/sr/60:.1f}m...")
            wav_chunk = wav[i:chunk_end]
            _, chunk_embeds, chunk_splits = embedder.embed_utterance(wav_chunk, return_partials=True)
            adjusted_splits = [slice(s.start + i, s.stop + i) for s in chunk_splits]
            embeddings_list.append(chunk_embeds)
            splits_list.extend(adjusted_splits)
        cont_embeds = torch.cat(embeddings_list, dim=0)
        wav_splits = splits_list
    else:
        _, cont_embeds, wav_splits = embedder.embed_utterance(wav, return_partials=True)
    print(f"Created {len(wav_splits)} embedding segments.")
    return cont_embeds, wav_splits

def compute_reference_embeddings(embedder: Embed, speaker_wavs: List[torch.Tensor], speaker_names: List[str], device: torch.device) -> List[torch.Tensor]:
    """
    Compute reference speaker embeddings.
    """
    print("Computing reference speaker embeddings...")
    speaker_embeds = []
    for i, speaker_wav in enumerate(speaker_wavs):
        print(f"  Processing speaker: {speaker_names[i]}")
        speaker_embed = torch.tensor(embedder.embed_utterance(speaker_wav), device=device)
        speaker_embeds.append(speaker_embed)
    return speaker_embeds

def compute_similarities(cont_embeds: torch.Tensor, speaker_embeds: List[torch.Tensor], speaker_names: List[str]) -> dict:
    """
    Compute cosine similarities between continuous and reference embeddings.
    """
    print("Computing similarities...")
    similarity_dict = {}
    for name, speaker_embed in zip(speaker_names, speaker_embeds):
        similarity = (cont_embeds @ speaker_embed).detach().cpu().numpy()
        similarity_dict[name] = similarity
    return similarity_dict

def run_interactive_visualization(similarity_dict, wav, wav_splits):
    print("Starting interactive visualization...")
    interactive_diarization(similarity_dict, wav, wav_splits, x_crop=7, show_time=True)

def main():
    default_audio_path = "visualisations/demo_speaker_diarisation.mp3"
    default_segments = [(0, 5.5), (6.5, 12), (17, 25)]
    default_speaker_names = ["Kyle Gass", "Sean Evans", "Jack Black"]
    audio_path = prompt_audio_path(default_audio_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    waveform, sample_rate = load_audio_file(audio_path)
    wav = preprocess_waveform(waveform, sample_rate)
    if audio_path == default_audio_path:
        segments, speaker_names = default_segments, default_speaker_names
        print("\nUsing default speaker segments and names:")
        for i, (name, (start, end)) in enumerate(zip(speaker_names, segments), 1):
            print(f"  Speaker #{i}: {name}, Segment: {start}-{end}s")
    else:
        segments, speaker_names = prompt_speaker_segments()
    speaker_wavs = extract_reference_segments(wav, segments, sampling_rate)
    encoder = load_speech_encoder(device)
    embedder = Embed(encoder)
    cont_embeds, wav_splits = compute_embeddings(embedder, wav, sampling_rate)
    speaker_embeds = compute_reference_embeddings(embedder, speaker_wavs, speaker_names, device)
    similarity_dict = compute_similarities(cont_embeds, speaker_embeds, speaker_names)
    run_interactive_visualization(similarity_dict, wav, wav_splits)

if __name__ == "__main__":
    main()