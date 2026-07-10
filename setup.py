#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="chorus-detection",
    version="0.2.0",
    description="A tool for detecting chorus sections in audio files",
    author="Dennis Dang",
    url="https://github.com/dennisvdang/chorus-detection",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.24.0,<1.25.0",
        "scipy>=1.10.0,<1.11.0",
        "scikit-learn>=1.3.0,<1.4.0",
        "librosa>=0.10.0,<0.11.0",
        "soundfile>=0.12.1,<0.13.0",
        "pydub>=0.25.1,<0.26.0",
        "matplotlib>=3.7.0,<3.8.0",
        "pandas>=2.0.0,<2.1.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0.0",
        "torch>=2.0.0",
        "streamlit>=1.22.0,<1.23.0",
        "pytube>=15.0.0,<16.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "chorus-detection-web=web.app:run_web_app",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
