# gsoc-2025-denoising-diffusion-project
A Diffusion-Based Deep Learning Framework for Denoising Protoplanetary Disk Observations

---

##  How to Run the Pipeline

Follow the steps below to generate your data, create hyperparameters, train the diffusion model, and validate the results. All commands should be executed from the project’s root directory on a Linux system.

### 1️ **Generate the Dataset**

Run the data creation script to prepare all required training/validation files:

```bash
python create_data.py
```

### 2️ **Generate Hyperparameters**

This script creates or updates the configuration file used by the diffusion model:

```bash
python create_hyperparameters.py
```

### 3️ **Train the Diffusion Model**

Start training using the generated dataset and hyperparameters:

```bash
python train_diffusion.py
```

### 4️ **Validate the Model**

After training completes, run validation to evaluate performance:

```bash
python validate.py
```

---
