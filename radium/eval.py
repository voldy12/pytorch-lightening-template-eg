from typing import Tuple, Dict

import lightning as L
import hydra
from omegaconf import DictConfig

from radium import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.models._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.models)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "trainer": trainer,
    }

    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    log.info(f"Best ckpt path: {cfg.ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    # train the model
    metric_dict, _ = train(cfg)

    # this will be used by hydra later for optimization
    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
