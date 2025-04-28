from models import *

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_time_with_cpu_dml():
    import time
    cpu = torch.device('cpu')
    dml = torch_directml.device()
    
    x = torch.randn(32, 3, 128, 128).to(cpu)
    vae_cpu = VAE(128).to(cpu)
    vae_dml = VAE(128).to(dml)
    
    # test CPU
    total_time_cpu = 0
    for i in range(10):
        start = time.time()
        x_reconst, mu, logvar = vae_cpu(x)
        total_time_cpu += time.time() - start
    print(f"Average time on CPU: {total_time_cpu/10:.4f} seconds")
    
    # test DML
    x = x.to(dml)
    total_time_dml = 0
    for i in range(10):
        start = time.time()
        x_reconst, mu, logvar = vae_dml(x)
        total_time_dml += time.time() - start
    print(f"Average time on DML: {total_time_dml/10:.4f} seconds")
    
def test_time_with_cpu_dml_VQVAE():
    import time
    cpu = torch.device('cpu')
    dml = torch_directml.device()
    
    x = torch.randn(32, 3, 128, 128).to(cpu)
    vqvae_cpu = VQVAE(128, 512).to(cpu)
    vqvae_dml = VQVAE(128, 512).to(dml)
    
    # test CPU
    total_time_cpu = 0
    for i in range(10):
        start = time.time()
        x_reconst, q_loss = vqvae_cpu(x)
        total_time_cpu += time.time() - start
    print(f"Average time on CPU: {total_time_cpu/10:.4f} seconds")
    
    # test DML
    x = x.to(dml)
    total_time_dml = 0
    for i in range(10):
        start = time.time()
        x_reconst, q_loss = vqvae_dml(x)
        total_time_dml += time.time() - start
    print(f"Average time on DML: {total_time_dml/10:.4f} seconds")