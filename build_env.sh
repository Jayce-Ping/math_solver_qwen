#!/bin/bash

# conda activate pytorch-env

pip install torch torchvision torchaudio

pip install vllm --extra-index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt

