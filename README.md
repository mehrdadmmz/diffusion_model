# Diffusion Models on MNIST

This repository contains a Jupyter Notebook implementation of a diffusion model trained on the MNIST dataset using PyTorch and Hugging Face's [diffusers](https://github.com/huggingface/diffusers) library. The model uses a UNet backbone along with two noise schedulers: DDPM for stochastic sampling and DDIM for deterministic sampling.

## Overview

- **Dataset:** MNIST  
- **Model:** UNet2DModel  
- **Schedulers:**  
  - **DDPM Scheduler:** For stochastic training and sampling  
  - **DDIM Scheduler:** For deterministic training and sampling  
- **Loss Function:** Mean Squared Error (MSE)  
- **Optimizer:** AdamW

The notebook includes steps for:
- Loading and visualizing the MNIST dataset.
- Plotting the noise schedule.
- Training the model with both DDPM and DDIM schedulers.
- Plotting the DDPM loss curve.
- Generating samples using both schedulers.

## Installation

Install the required dependencies with:

```bash
pip install torch torchvision diffusers matplotlib
```

## Usage
- Clone the repository:
```bash
git clone https://github.com/mehrdadmmz/diffusion_model.git
cd diffusion_model

```
- Run the notebook:
  jupyter notebook
- Execute the cells in the notebook to train the model, visualize the noise schedule, plot the loss curve, and generate sample images.


## Results
- Noise Schedule
    - <img width="429" alt="image" src="https://github.com/user-attachments/assets/e03145df-12ea-44be-86d8-2b6f47e9b8fa" />
- DDPM Loss Curve
    - <img width="442" alt="image" src="https://github.com/user-attachments/assets/326a370f-001a-484e-a1d0-efe920066a59" />
- Generated Samples
    - DDPM Samples (Stochastic Sampling)
        - <img width="397" alt="image" src="https://github.com/user-attachments/assets/f0ca235a-6f68-4b65-a219-ae881e0fd786" />
    - DDIM Samples (Deterministic Sampling)
        - <img width="396" alt="image" src="https://github.com/user-attachments/assets/d3f78c43-e275-4fa1-980a-ffa6c4cd0895" />



