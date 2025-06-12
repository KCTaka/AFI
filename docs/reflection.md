# **Project Comments and Reflection**

## **Intention**
I had this grand idea. I occasionally create edit/videos with cartoons/anime and realized its incompatibility with slow-motion. There are methods online to use an algorithm (**Twixtor**) to extrapolate and increase frame rate; however, there was two huge problem with that:  
1. **Twixtor bad job with interpolating between two frames (lacks understanding of the object in motion)**  
2. **the video frame rate does not match with the true animation frame rate**  

With the ever-growing advancement of generation models, and the passion/intention to develop my skills as an engineer, I decided to build my own model to approach the proposed two challenges. This is a large leap for me, and I must acquire - along with many others - the following skills:  

- Design an appropriate model architecture  
- Manage and organize datasets  
- Train efficiently with appropriate techniques  

I've had basic skills with **PyTorch** and distributed training with **torch run** but I struggled on finding the right hyperparameter configuration which often led to large time consumption trying to figure out why it seemingly didn't work when the code itself was fine. I decided to use sweeping (**hyperparameter optimization**) and proper modern **MLOps** standards to improve my workflow for the big project.

## **What I liked to do**
I've always loved and have been doing things from scratch to a point where I am incapable of learning something without understanding its seemingly unnecessary basics. Many languages, libraries, and dependencies are a **black bock** in the sense that you as the user do not need to know how it happens but only what it does. However, when I plan what to do with this dependency, I tend to need to know **how it does what it does** so that it does not conflict with what I plan to make out of it. This may sound proper, but I get overly cautious and am not able to proceed without researching and understanding how it works. **This project is a large step** towards finding that balance and making progress despite not fully understanding what happens under the hood. I believe this is necessary to some extent when working on a large project where it's near impossible to grasp everything before proceeding, leading to huge time loss. 

## **In the beginning**
At the start, planned to do this project with **PyTorch** + **torch run** + **tensorboard** + **Optuna**. Which went okay but the code looked really messy. That's when I discovered using something like **PyTorch lightning** and **W&B** is the current standard in the ML workflow. I had to revamp the entire project to accommodate the lightning structure and W&B logging. However, with that, it helped me understand Optuna, tensorboard, and torch run much more than before.  
This was my vague realization (although Optuna offers more in sweeping):  

- **W&B** = **Optuna** + **Tensorboard**  
- **PyTorch Lightning** = **PyTorch** + **torch run**  

## **Problems**
There were so many problems that just wouldn't go my way:  

- **Discovering PyTorch lightning.**  
  I discovered that people use lightning instead of what I had for simplicity, and I knew about it after going through hell to write a messy yet functional distributed training code. I think the majority of my time was wasted from this.  

- **W%B sweep and agent does not programmatically work well with lightning**  
  The problem was multiple GPU runs the agent and sweep code making many sweeps as there is GPU. I could not find solution for this, I just had to avoid it with command line sweep method.  

- **Time-embedding required a convolutional layer in two stages before adding it into the model flow.**  
  One for in general and the other for each block/layer in the model architecture. I initially had just one for each which led to inconsistent and irregular generation patterns.  

- **W%B Model logging/storing stored ALL model after every epoch**  
  I was surprised I couldn't find a single solution to this simple and intuitively common problem. I ended up just giving up and just storing the latest model only.  

- **U-net architecture had to be built in a certain way than what I initially thought.**  
  As the paper and videos I was watching had interesting blocks, the way to skip connection in U-net was different from my previous understanding of how u-net looked like. The concatenation happened after down-sampling, whereas the standard u-net structure had before down-sampling. Also downsampling was done with **Conv2d** instead of **MaxPooling**, which probably allowed the model to pick up on more details but introduced more parameters to train (since this happens only 6 times in total, I deem it insignificant).  

Each of the above problems (and many more) just led to me not being able to progress for so long. 

Another problem I just encountered is that the model didn't save after completion on W&B for some reason and I lost progress on 100 epochs. At least I had the logging saved, but I would have to train from scratch again if I wanteed the model file. Also, I forgot to log the quantized latent image after all denoising process during validation phase.

## **Comments**
Although I hated how even the simple things I couldn't progress, the outcome was better than what I followed on paper. **This is definitely a large step** towards the big project
