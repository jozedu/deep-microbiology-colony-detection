"""
Custom training utilities for Detectron2.

Contains:
- LossEvalHook: Computes validation loss during training.
- MyTrainer: Custom DefaultTrainer subclass with validation loss tracking
             and COCO evaluation.
"""

import os
import time
import logging
import datetime

import numpy as np
import torch

from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import COCOEvaluator, inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm


class LossEvalHook(HookBase):
    """Hook that computes validation loss at regular intervals during training.

    This is needed because Detectron2's DefaultTrainer does not compute
    validation loss by default — it only runs evaluation metrics (mAP).

    Args:
        eval_period: Compute validation loss every `eval_period` iterations.
        model: The model being trained.
        data_loader: A DataLoader for the validation set.
    """

    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):
        """Iterate over the validation set and compute the mean loss."""
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []

        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start

            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (
                    (time.perf_counter() - start_time) / iters_after_start
                )
                eta = datetime.timedelta(
                    seconds=int(total_seconds_per_img * (total - idx - 1))
                )
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Loss on Validation done {idx + 1}/{total}. "
                        f"{seconds_per_img:.4f} s / img. ETA={eta}"
                    ),
                    n=5,
                )

            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)

        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar("validation_loss", mean_loss)
        comm.synchronize()

        return losses

    def _get_loss(self, data):
        """Compute total loss for a single batch (in eval mode)."""
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        return sum(metrics_dict.values())

    def after_step(self):
        """Called after each training step."""
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)


class MyTrainer(DefaultTrainer):
    """Custom Detectron2 trainer with validation loss tracking.

    Extends DefaultTrainer to:
    1. Build a COCOEvaluator for the test/validation set.
    2. Insert a LossEvalHook that computes validation loss at each
       evaluation period, enabling loss curve monitoring.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(
            -1,
            LossEvalHook(
                self.cfg.TEST.EVAL_PERIOD,
                self.model,
                build_detection_test_loader(
                    self.cfg,
                    self.cfg.DATASETS.TEST[0],
                    DatasetMapper(self.cfg, True),
                ),
            ),
        )
        return hooks
