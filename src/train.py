import os, torch, hydra, wandb
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from lightning.pytorch import Trainer, Callback, LightningDataModule, LightningModule
from lightning.pytorch.loggers import Logger, WandbLogger

from src.utils.hydra import instantiate_list, flat_dict_to_nested_dict

# ----------------------------------------------------------------------
def _broadcast_cfg(cfg: DictConfig) -> DictConfig:
    """Send the merged sweep-cfg from rank 0 to every other rank."""
    if dist.is_available() and dist.is_initialized():
        obj_list = [OmegaConf.to_container(cfg, resolve=True)]
        dist.broadcast_object_list(obj_list, src=0)         # send once  :contentReference[oaicite:0]{index=0}
        cfg = OmegaConf.create(obj_list[0])                 # overwrite (all ranks)
    return cfg

# ----------------------------------------------------------------------
def train(cfg: DictConfig, *, is_sweep: bool = False) -> None:
    """One complete Lightning run (single-GPU or DDP)."""

    # ------------------------------------------------------------------ sweep handling
    run = None
    if is_sweep:
        @rank_zero_only                                       # guard: one run only  :contentReference[oaicite:1]{index=1}
        def _safe_init():
            return wandb.init(project=cfg.loggers.wandb.project,
                               job_type="sweep-run")
        run = _safe_init()

        if run is not None:                                   # rank 0 merges config
            cfg = OmegaConf.merge(cfg,
                    flat_dict_to_nested_dict(run.config))
        cfg = _broadcast_cfg(cfg)                             # every rank now equal

    # ------------------------------------------------------------------ build objects
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    system:      LightningModule    = hydra.utils.instantiate(cfg.systems)

    # WandbLogger uses rank-aware DummyExperiment internally  :contentReference[oaicite:2]{index=2}
    loggers: list[Logger] = [
        WandbLogger(experiment=run) if run else
        WandbLogger(project=cfg.loggers.wandb.project)
    ]
    loggers += instantiate_list(
        {k: v for k, v in cfg.loggers.items() if k != "wandb"}.values(), cls=Logger
    )

    callbacks: list[Callback] = instantiate_list(cfg.callbacks.values(), cls=Callback)

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=loggers,
        callbacks=callbacks,
    )

    trainer.fit(system, datamodule=datamodule)

# ----------------------------------------------------------------------
@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Entry-point. Hydra passes the YAML tree in `cfg`."""

    if cfg.get("sweep") is None:                       # plain single run
        train(cfg, is_sweep=False)
        return

    # ------------------------------ sweep-agent: one per machine
    @rank_zero_only
    def _run_agent() -> None:
        sweep_id = wandb.sweep(
            OmegaConf.to_container(cfg.sweep, resolve=True),
            project=cfg.loggers.wandb.project)                    

        # Every call below spins up *one* training run
        def _wrapped_train():
            train(cfg, is_sweep=True)

        wandb.agent(sweep_id,
                    function=_wrapped_train,
                    count=cfg.sweep.agent.count,
                    project=cfg.loggers.wandb.project)

    _run_agent()

if __name__ == "__main__":
    main()
