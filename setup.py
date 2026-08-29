"""
AXIOM Package Setup
Copyright (c) 2026 Next Gen AI / Rob Kucher
"""

from setuptools import setup, find_packages

setup(
    name="axiom-llm",
    version="0.1.0",
    description="AXIOM: Adaptive eXpanded Intelligence with Orthogonal Memory",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    author="Rob Kucher / Next Gen AI",
    author_email="rob@nextgenai.com.mx",
    url="https://github.com/nextgenai/axiom",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "tokenizers>=0.19.0",
        "accelerate>=0.29.0",
        "sentencepiece>=0.2.0",
        "tqdm>=4.66.0",
        "numpy>=1.26.0",
    ],
    extras_require={
        "train": [
            "wandb>=0.16.0",
            "deepspeed>=0.14.0",
        ],
        "quant": [
            "bitsandbytes>=0.43.0",
        ],
        "fast": [
            "flash-attn>=2.5.0",
            "xformers>=0.0.25",
        ],
        "dev": [
            "pytest>=8.0.0",
            "black>=24.0.0",
            "isort>=5.13.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
