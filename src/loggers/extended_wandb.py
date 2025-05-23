# src/loggers/extended_wandb.py

import os
import logging
from typing import Any, Dict, Optional, Union # Added for type hints

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_only # Ensure logging/API calls only on rank 0

# It's good practice to have a logger for your custom class
logger = logging.getLogger(__name__)
# Suppress wandb's own verbose logging unless necessary for debugging
logging.getLogger("wandb").setLevel(logging.WARNING)


class ExtendedWandbLogger(WandbLogger):
    def __init__(
        self,
        *args: Any,
        # gcs_checkpoint_dir, save_top_k, save_last are removed as they are not
        # used by this logger for W&B pruning. Pruning is based on ModelCheckpoint's state.
        **kwargs: Any,
    ) -> None:
        # Pop unused arguments that might have been in old configs to avoid errors
        super().__init__(*args, **kwargs)
        logger.info("ExtendedWandbLogger initialized. Pruning will be based on ModelCheckpoint callback states.")

    @rank_zero_only # Important: ensure artifact operations run only on main process
    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        # Let the parent class log the artifact first.
        # This uses the checkpoint_callback.dirpath and the new checkpoint file.
        # Ensure WandbLogger's `log_model` param is True or "all" for this to work.
        super().after_save_checkpoint(checkpoint_callback)

        if not self.experiment:
            logger.warning("W&B experiment not initialized (self.experiment is None). Skipping artifact pruning.")
            return

        trainer = checkpoint_callback.trainer 
        if not trainer:
            logger.warning("Trainer not found in checkpoint_callback. Skipping W&B artifact pruning.")
            return

        # Collect all "approved" checkpoint basenames from ALL ModelCheckpoint callbacks
        approved_basenames = set()
        for cb in trainer.checkpoint_callbacks:
            if isinstance(cb, ModelCheckpoint):
                # cb.best_model_path, cb.last_model_path etc. are absolute paths.
                # We need their basenames as these are typically used for W&B artifact names.
                if cb.best_model_path and cb.best_model_path: # Check if path is not empty or None
                    approved_basenames.add(os.path.basename(cb.best_model_path))
                if cb.last_model_path and cb.last_model_path:
                    approved_basenames.add(os.path.basename(cb.last_model_path))
                for path in cb.kth_best_model_paths.values():
                    if path: # Check if path is not empty or None
                        approved_basenames.add(os.path.basename(path))
        
        if not approved_basenames:
            logger.info("No approved checkpoints found from ModelCheckpoint callbacks' state. Skipping W&B artifact pruning.")
            return

        logger.info(f"Approved W&B artifact basenames (from ModelCheckpoints): {approved_basenames}")

        run = self.experiment
        try:
            # List artifacts of type 'model' (default for checkpoints)
            # Note: run.logged_artifacts() returns summaries. Iterating through them.
            artifacts_to_prune_names = []
            api = self._wandb_init.get("wandb_api") # Get the public API for more robust operations if needed

            # Using run.artifacts.filter to find artifacts.
            # This might be more robust or offer better filtering if available and works as expected.
            # The direct `run.logged_artifacts()` approach is also common.
            # Let's stick to simpler `run.logged_artifacts()` if it works for artifact names.

            # The name of the artifact logged by WandbLogger is typically the checkpoint file's basename.
            for art_summary in run.logged_artifacts(): # type: ignore # run.logged_artifacts() exists
                # WandbLogger logs checkpoints with type 'model'
                if art_summary.type == 'model':
                    # art_summary.name is the name given when logging the artifact
                    # e.g., "model-epoch=02-percep_recon_loss=0.34.ckpt"
                    if art_summary.name not in approved_basenames:
                        artifacts_to_prune_names.append(art_summary.name)
            
            if not artifacts_to_prune_names:
                logger.info("No W&B artifacts to prune for this run.")
                return

            logger.info(f"Found W&B artifacts to prune: {artifacts_to_prune_names}")

            for artifact_name in artifacts_to_prune_names:
                try:
                    # Fetch the artifact by name and type to delete it (all versions and aliases)
                    # We need to use the project context for the API if run context is not enough.
                    # Format for use_artifact is typically "entity/project/artifact_name:version_or_alias"
                    # or just "artifact_name:version_or_alias" if project/entity are clear from context.
                    # Since we are in the run context, "artifact_name:latest" with type should work.
                    
                    logger.info(f"Attempting to delete stale W&B artifact '{artifact_name}' (type: model)...")
                    stale_artifact = run.use_artifact(f"{artifact_name}:latest", type='model') # type: ignore
                    stale_artifact.delete(delete_aliases=True)
                    logger.info(f"Successfully deleted stale W&B artifact '{artifact_name}'.")
                except Exception as e:
                    # Common error: "Artifact not found". Could be already deleted or a naming mismatch.
                    logger.error(f"Failed to delete W&B artifact '{artifact_name}': {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Error during W&B artifact pruning process: {e}", exc_info=True)