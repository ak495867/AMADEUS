from setuptools import setup, find_packages

found = find_packages()
packages = ["amadeus"] + [f"amadeus.{p}" for p in found]
package_dir = {"amadeus": "."}
for p in found:
    package_dir[f"amadeus.{p}"] = p.replace(".", "/")

setup(
    name="amadeus",
    version="1.0.0",
    packages=packages,
    package_dir=package_dir,
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "tqdm>=4.65.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "websockets>=11.0",
    ],
    entry_points={
        "console_scripts": [
            "amadeus=amadeus.sdk.cli:main",
        ],
    },
)
