"""
Utility functions for chorus detection.
"""

import os
import shutil
from functools import reduce

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

# Constants
AUDIO_TEMP_PATH = "output/temp"


def extract_audio(url, output_path=AUDIO_TEMP_PATH):
    """Downloads audio from a YouTube URL and saves it as MP3.

    Returns:
        Tuple of (path to the mp3 file, video title), or (None, None) on failure.
    """
    # Imported here, not at module scope: strip_silence() below is pulled in by
    # the feature pipeline, and that must not require a YouTube downloader.
    import yt_dlp

    try:
        os.makedirs(output_path, exist_ok=True)
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(output_path, "%(id)s.%(ext)s"),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            # Fail fast when YouTube is unreachable or blocking the request
            # instead of cycling through yt-dlp's default retry backoff.
            "socket_timeout": 10,
            "retries": 2,
            "fragment_retries": 2,
            "extractor_retries": 1,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        audio_file = os.path.join(output_path, f"{info['id']}.mp3")
        if not os.path.exists(audio_file):
            print("Audio download did not produce an mp3 file")
            return None, None
        return audio_file, info.get("title")
    except Exception as e:
        print(f"An error occurred while downloading audio: {e}")
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
