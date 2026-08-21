"""Run manifests: one JSON record per run stating what ran, on what, with which versions.

A manifest holds the full command, the git commit of the code that ran, the
checkpoint file and its hash, the resolved versions of the beat tracker and
its dependencies, the seed, the dataset and split paths, and the start and
end times. Standard library only.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from typing import Optional, Sequence

# Packages whose resolved versions every manifest records. beat_this installs
# from the tip of its GitHub main branch, so its version can change per run.
MANIFEST_PACKAGES = ("beat_this", "torch", "torchaudio", "librosa", "numpy",
                     "scipy", "scikit-learn", "soundfile", "soxr",
                     "chorus-detection")


def utc_now_iso() -> str:
    """The current UTC time, second precision, ISO 8601."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: str, chunk_bytes: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed so checkpoints do not load into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def git_state(repo_dir: str) -> dict:
    """The commit the given repository is at, and whether its tree is dirty.

    Dirty counts only modifications to tracked files. Never raises: when git
    is absent or the directory is not a repository, the failure is recorded
    as fields.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir,
            capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_dir, capture_output=True, text=True, check=True).stdout
        return {"repo_dir": os.path.abspath(repo_dir),
                "commit": commit, "dirty": bool(status.strip())}
    except Exception as exc:  # git missing, not a repo, or repo unreadable
        return {"repo_dir": os.path.abspath(repo_dir),
                "commit": None, "dirty": None, "error": str(exc)}


def package_versions(names: Sequence[str] = MANIFEST_PACKAGES) -> dict:
    """Resolved version of each named package; None when not installed."""
    versions = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def checkpoint_record(path: Optional[str]) -> Optional[dict]:
    """Path, size, and SHA-256 of a checkpoint file; None when no checkpoint."""
    if not path:
        return None
    record = {"path": os.path.abspath(path)}
    if os.path.exists(path):
        record["bytes"] = os.path.getsize(path)
        record["sha256"] = file_sha256(path)
    else:
        record["error"] = "file not found at manifest time"
    return record


def write_manifest(path: str, *,
                   command: Optional[Sequence[str]] = None,
                   repo_dir: Optional[str] = None,
                   checkpoint: Optional[str] = None,
                   seed: Optional[int] = None,
                   dataset_paths: Optional[dict] = None,
                   split_files: Optional[dict] = None,
                   started_at: Optional[str] = None,
                   finished_at: Optional[str] = None,
                   extra: Optional[dict] = None) -> dict:
    """Write one run manifest as JSON and return it as a dict.

    Args:
        command: The full invocation; defaults to sys.argv.
        repo_dir: Repository whose commit identifies the code that ran;
            defaults to the directory of the running script.
        checkpoint: Model checkpoint file the run read, hashed into the record.
        seed: The run's random seed, or None when the run takes none.
        dataset_paths: Named input paths, e.g. {"audio_dir": ..., "labels": ...}.
        split_files: Named split lists, e.g. {"test": ".../test_songs.txt"}.
        started_at / finished_at: ISO times from utc_now_iso(); finished_at
            defaults to now, so calling this at the end of a run records both.
        extra: Run-specific fields (arm name, decoder, grid, counts).
    """
    if repo_dir is None:
        main_file = getattr(sys.modules.get("__main__"), "__file__", None)
        repo_dir = os.path.dirname(os.path.abspath(main_file)) if main_file else os.getcwd()
    manifest = {
        "manifest_version": 1,
        "command": list(command) if command is not None else list(sys.argv),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git": git_state(repo_dir),
        "checkpoint": checkpoint_record(checkpoint),
        "packages": package_versions(),
        "seed": seed,
        "dataset_paths": dataset_paths or {},
        "split_files": split_files or {},
        "started_at": started_at,
        "finished_at": finished_at or utc_now_iso(),
    }
    if extra:
        manifest["extra"] = extra
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
