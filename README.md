# Animation Frame Interpolation (AFI) Project

## Overview

This project aims to develop a high-quality frame interpolation model specifically tailored for animation and cartoon-style videos. The primary objective is to enhance video fluidity by increasing frames per second (FPS) while meticulously preserving the original motion dynamics and artistic nuances. This is a solo endeavor by an engineering student, focusing on practical application and learning within a resource-constrained environment.

## Project Goal

To create a robust deep learning model capable of:
*   Interpolating frames in video clips of animations/cartoons.
*   Maintaining high visual fidelity and consistency of motion.
*   Operating under the assumption of no scene cuts within processed video segments.

## Methodology

The planned workflow is as follows:
1.  **Video Preprocessing:** Segment videos into clips, ensuring each clip does not contain scene cuts.
2.  **Frame Generation:** Utilize a deep learning model to generate intermediate frames for each clip, effectively increasing its FPS.
3.  **Video Postprocessing:** Reassemble the processed clips to form the final high-frame-rate video.

## Technical Stack & Constraints

### Core Technologies:
*   **Programming Language:** Python
*   **Deep Learning Framework:** PyTorch
*   **Development Environment:** Visual Studio Code

### Constraints:
*   **Hardware:** Limited local disk space; no dedicated CUDA-enabled GPU.
*   **Budget:** Project budget is capped at $200.

## Current Skills & Learning Focus

### Current Proficiencies:
*   Solid understanding of Machine Learning and Deep Learning fundamentals, including underlying mathematics.
*   Proficient in PyTorch for model development.
*   Familiarity with MySQL.

### Areas for Development:
*   Designing and training large-scale deep learning models.
*   Adopting professional methodologies for model architecture and training pipelines.
*   Exploring tools and best practices for efficient model development and experimentation.
*   Techniques for creating and managing large-scale datasets.

## Project Status

### Completed:
- [x] Initial training set-up for VAE (Variational Autoencoder) and VQ-VAE (Vector Quantized Variational Autoencoder).
- [x] Integration with TensorBoard for experiment tracking and visualization.
- [x] Implementation of Adversarial Loss component.
- [x] Integrate Perceptual Loss (e.g., using VGG16 features).

### In-Progress:
- [ ] Determine promising hyperparameters for the VAE/VQVAE

### To-Do:

- [ ] Experiment with U-Net architecture for the core generation task.
- [ ] Develop and apply methods for dataset filtering and curation.
- [ ] Implement an advanced discriminator (e.g., "SN-ResNet PatchGAN + SA").
- [ ] Explore IS-Net as an alternative to or enhancement of U-Net.
- [ ] Investigate and implement Perceptual Loss based on recent research (e.g., [arXiv:2312.01943](https://arxiv.org/pdf/2312.01943)).
- [ ] Conduct small-scale training and refinement using a subset of the dataset at 128x128 resolution.
- [ ] Scale up and refine training for 256x256 resolution.
- [ ] Further scale up and refine training for 512x128 resolution.
- [ ] Implement and optimize for large-scale training with the full 512x512 target resolution.

## Concepts to Explore & Learn

To enhance this project and personal skill set, the following concepts are targeted for learning and potential integration:

*   **Advanced GAN Architectures:** Deeper dive into PatchGAN, and understanding architectures like StyleGAN or CycleGAN for inspiration, even if direct application is complex.
*   **Perceptual Loss Functions:** Beyond VGG-based, explore LPIPS and other metrics that correlate well with human perception of image quality.
*   **Attention Mechanisms & Transformers:** For capturing long-range dependencies in sequences, potentially beneficial for video tasks.
*   **Flow Estimation Techniques:** Optical flow can be a powerful input or supervisory signal for frame interpolation.
*   **Data Augmentation for Video:** Techniques specific to video data to increase dataset robustness.
*   **Efficient Training Strategies:**
    *   Gradient Accumulation: To simulate larger batch sizes with limited memory.
    *   Mixed-Precision Training: If adaptable to non-CUDA environments or future cloud use.
    *   Transfer Learning: Leveraging pre-trained models or components.
*   **Model Optimization & Quantization:** For eventual efficient inference, though not an immediate priority.
*   **Cloud Computing Resources:** Exploring cost-effective options for occasional intensive training (e.g., Google Colab Pro, Kaggle Kernels, AWS/Azure spot instances within budget).
*   **Version Control:** Consistent use of Git and GitHub for project management and collaboration (even for solo projects).
*   **Automated Hyperparameter Tuning:** Tools like Optuna or Ray Tune for efficient experimentation.
