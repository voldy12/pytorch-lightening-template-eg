from logging import Logger
from typing import Tuple, Dict
import lightning as L
import torch
import hydra
from omegaconf import DictConfig
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from typing import List, Tuple, Dict

from radium import utils

log = utils.get_pylogger(__name__)


#@utils.task_wrapper
def predict(cfg: DictConfig) -> Tuple[dict, dict]:
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.models._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.models)
    
    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))
    
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)


    log.info("Starting Prediction!")
    preds = trainer.predict(model=model, datamodule=datamodule, return_predictions = True,  ckpt_path=cfg.test_ckpt_path)
    log.info(f"Prediction Values <{preds}>")
    print(f'Top-k[{cfg.models.topk}] preds : {preds}')


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig):
    # Run inference on the model
    predict(cfg)


if __name__ == "__main__":
    main()