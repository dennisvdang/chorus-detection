"""
Utility functions for chorus detection.
"""

import os
import shutil
from functools import reduce

from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from pytube import YouTube

# Constants
AUDIO_TEMP_PATH = "output/temp"


def extract_audio(url, output_path=AUDIO_TEMP_PATH):
    """Downloads audio from YouTube URL and saves as MP3."""
    try:
        yt = YouTube(url)
        video_title = yt.title
        audio_stream = yt.streams.filter(only_audio=True).first()
        if not audio_stream:
            print("No audio stream found")
            return None, None

        os.makedirs(output_path, exist_ok=True)
        out_file = audio_stream.download(output_path)
        base, _ = os.path.splitext(out_file)
        audio_file = base + '.mp3'
        if os.path.exists(audio_file):
            os.remove(audio_file)
        os.rename(out_file, audio_file)
        return audio_file, video_title
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


def strip_silence(audio_path):
    """Removes silent parts from an audio file."""
    sound = AudioSegment.from_file(audio_path)
    nonsilent_ranges = detect_nonsilent(
        sound, min_silence_len=500, silence_thresh=-50)
    stripped = reduce(lambda acc, val: acc + sound[val[0]:val[1]],
                      nonsilent_ranges, AudioSegment.empty())
    stripped.export(audio_path, format='mp3')


def cleanup_temp_files():
    """Clean up temporary files after processing."""
    if os.path.exists(AUDIO_TEMP_PATH):
        try:
            shutil.rmtree(AUDIO_TEMP_PATH)
        except Exception as e:
            print(f"Warning: Could not clear temporary files: {e}")
