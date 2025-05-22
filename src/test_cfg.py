import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="train_ddpm")
def test_cfg(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    print(OmegaConf.to_yaml(cfg))
    
if __name__ == "__main__":
    test_cfg()