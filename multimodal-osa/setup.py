from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="multimodal-osa",
    version="1.0.0",
    description="Dual contrastive learning with clinically guided multimodal fusion "
                "for speech-based OSA severity estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Wang et al.",
    url="https://github.com/your-username/multimodal-osa",
    license="MIT",
    packages=find_packages(exclude=["tests*", "experiments*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "soundfile>=0.12.0",
        "librosa>=0.10.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
