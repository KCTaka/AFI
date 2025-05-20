
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from typing import Any, Type, List, Optional

def instantiate_list(configs: List[DictConfig], 
                     cls: Type[Any] = object, 
                     *args, **kwargs) -> List[Any]:
    """
    Instantiate a list of objects from a list of configs.
    Args:
        configs (List[DictConfig]): List of configs to instantiate.
        cls (Optional[Type[Any]]): Class to instantiate. If None, use the class from the config.
        *args: Additional arguments to pass to the constructor.
        **kwargs: Additional keyword arguments to pass to the constructor.
    Returns:
        List[Any]: List of instantiated objects.
    """
    if isinstance(configs, DictConfig):
        configs = [configs]
    
    instances = []
    for config in configs:
        if isinstance(config, DictConfig) and "_target_" in config:
            # If the config has a target, instantiate it
            instance = instantiate(config, *args, **kwargs)
            if isinstance(instance, cls):
                # If the instance is of the correct type, append it to the list
                instances.append(instance)

    return instances