from typing import List, Optional, Tuple

import hydra
import lightning as L
import pyrootutils
import torch
torch.set_float32_matmul_precision("high")

from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from misalign import utils
from datetime import datetime

log = utils.get_pylogger(__name__)

@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    for key, value in train_metrics.items():
        log.info(f"{key}: {value}")
    
    if cfg.get("test"):
        log.info("Starting testing!") # TODO: Test 할때 여기 조정하기.
        # ckpt_path = "/SSD3_8TB/Daniel/13_misalign_proposed_final/logs/Model_adaconv_Data_SynthRAD_MR_CT_Pelvis_Misalign_X0_Y0_R0_M0_D0/synthRAD_Adaconv_content1_contextual5_style1_cycle0_noTanh/runs/2024-01-27_14-20-06/checkpoints/epoch_epoch=053.ckpt"
        # ckpt_path = "/SSD3_8TB/Daniel/13_misalign_proposed_final/logs/Model_proposed_A_to_B_Data_SynthRAD_MR_CT_Pelvis_Misalign_X0_Y0_R0_M0_D0/synthRAD_Proposed_100_30_100_120_HM_Applied_(HistogramMatching적용된거로해보니까ISMRM보다더성능좋네!)/runs/2023-12-19_06-52-45/checkpoints/epoch_epoch=089.ckpt"
        # ckpt_path = "/SSD3_8TB/Daniel/13_misalign_proposed_final/logs/Model_dam_Data_SynthRAD_MR_CT_Pelvis_Misalign_X0_Y0_R0_M0_D0/synthRAD_DAM_Train_justLowBatch(이게DAM에쓴최종버전)/runs/2023-11-24_06-23-57/checkpoints/epoch_epoch=093.ckpt"
        # ckpt_path = "/SSD3_8TB/Daniel/13_misalign_proposed_final/logs/Model_pgan_Data_SynthRAD_MR_CT_Pelvis_Misalign_X0_Y0_R0_M0_D0/synthRAD_PGAN_lv0/runs/2023-10-27_09-12-22/checkpoints/epoch_epoch_095.ckpt"
        # ckpt_path = "/SSD3_8TB/Daniel/13_misalign_proposed_final/logs/Model_proposed_A_to_B_Data_SynthRAD_MR_CT_Pelvis_Misalign_X0_Y0_R0_M0_D0/synthRAD_Proposed_100_30_100_120_l1WeightOut/runs/2023-11-07_03-25-04/checkpoints/epoch_epoch=098.ckpt" #TODO: test만하는경우 여기에 직접입력하기
        # ckpt_path = "/SSD3_8TB/Daniel/13_misalign_proposed_final/logs/Model_dam_Data_SynthRAD_MR_CT_Pelvis_Misalign_X0_Y0_R0_M0_D0/synthRAD_DAM_Train_cycle1_cycleWithReal/runs/2023-11-22_10-37-27/checkpoints/epoch_epoch=099.ckpt" #TODO: test만하는경우 여기에 직접입력하기
        # ckpt_path = "/SSD3_8TB/Daniel/13_misalign_proposed_final/logs/Model_dam_Data_SynthRAD_MR_CT_Pelvis_Misalign_X0_Y0_R0_M0_D0/synthRAD_DAM_Train_justLowBatch/runs/2023-11-24_06-23-57/checkpoints/epoch_epoch=093.ckpt"
        # ckpt_path = "/SSD3_8TB/Daniel/13_misalign_proposed_final/logs/Model_pgan_Data_SynthRAD_MR_CT_Pelvis_Misalign_X0_Y0_R0_M0_D0/synthRAD_Alignformer_lq_sr_PGAN_final/runs/2024-01-23_16-24-13/checkpoints/epoch_epoch=087.ckpt"
        # ckpt_path = "/SSD3_8TB/Daniel/13_misalign_proposed_final/logs/Model_dam_Data_SynthRAD_MR_CT_Pelvis_Misalign_X0_Y0_R0_M0_D0/synthRAD_Alignformer_lq_sr_DAM_final/runs/2024-01-23_16-24-21/checkpoints/epoch_epoch=093.ckpt"
        # ckpt_path = "/SSD3_8TB/Daniel/13_misalign_proposed_final/logs/Model_proposed_A_to_B_Data_SynthRAD_MR_CT_Pelvis_Misalign_X0_Y0_R0_M0_D0/synthRAD_Proposed_100_30_100_120_final/runs/2023-11-03_21-27-19/checkpoints/epoch_epoch_092.ckpt"
        ckpt_path = trainer.checkpoint_callback.best_model_path # train 할땐 켜고 test할땐 끄고
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics
    
    for key, value in test_metrics.items():
        log.info(f"{key}: {value}")
        
    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="defaults.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)
    
    log.info(f"Starting experiment at {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    
    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )
    
    log.info(f"Finished experiment at {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    return metric_value


if __name__ == "__main__":
    main()
