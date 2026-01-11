from plerio.training_utils.metrics_module import BinaryMetrics

import os
import copy
import shutil
from pathlib import Path

from torch import nn
import torch

import optuna
from optuna.integration import PyTorchLightningPruningCallback

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback


import typing as tp
import logging
import warnings



class SingleProtPRILightningModule(pl.LightningModule):
    """
    Lightning trainer for RNA-protein model with binary metrics.
    Expects batch dict: {'x': inputs, 'label': labels}.
    """
    def __init__(
        self,
        model: nn.Module,
        optimization_harness: dict[str, tp.Any],
        loss_fn: nn.Module = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.optimization_harness = optimization_harness
        self.loss_fn = loss_fn if loss_fn is not None else nn.BCEWithLogitsLoss()

        # Metrics
        self.train_metrics = BinaryMetrics()
        self.val_metrics = BinaryMetrics()
        self.test_metrics = BinaryMetrics()

    # ----------------------------
    # Forward pass
    # ----------------------------
    def forward(self, x):
        return self.model(x)

    # ----------------------------
    # Step functions
    # ----------------------------
    def training_step(self, batch, batch_idx):
        x = batch['seq']
        y = batch['label'][:, None].float()

        logits = self(x)
        # print(logits.dtype, y.dtype)
        loss = self.loss_fn(logits, y)

        self.train_metrics.update(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['seq']
        y = batch['label'][:, None].float()

        logits = self(x)            
        loss = self.loss_fn(logits, y)

        self.val_metrics.update(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch['seq']
        y = batch['label'][:, None].float()

        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.test_metrics.update(logits, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    # ----------------------------
    # Epoch-end metric logging
    # ----------------------------
    def on_train_epoch_end(self):
        stats = self.train_metrics.compute()
        for k, v in stats.items():
            self.log(f"train_{k}", v, prog_bar=False)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        stats = self.val_metrics.compute()
        for k, v in stats.items():
            self.log(f"val_{k}", v, prog_bar=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        stats = self.test_metrics.compute()
        for k, v in stats.items():
            self.log(f"test_{k}", v, prog_bar=True)
        self.test_metrics.reset()

    # ------------------------------------------------------------
    # optimizers
    # ------------------------------------------------------------
    def configure_optimizers(self):
        opt_class = getattr(
            torch.optim,
            self.optimization_harness['optimizer'],
        )

        optimizer = opt_class(
            self.parameters(),
            lr=self.optimization_harness['lr']
        )

        # scheduler_class = getattr(
        #     torch.optim.lr_scheduler, 
        #     self.optimization_harness['scheduler']
        # )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=1000, 
            eta_min=1e-6,
            last_epoch=-1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


class _MetricHistory(Callback):
    def __init__(self, monitor: str):
        super().__init__()
        self.monitor = monitor
        self.values: list[float] = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.monitor in trainer.callback_metrics:
            try:
                self.values.append(float(trainer.callback_metrics[self.monitor]))
            except Exception:
                pass


class OptunaLightningModelTuner:
    """
    Tuner that accepts an already-initialized nn.Module and tunes
    training hyperparameters with PyTorch Lightning + Optuna.

    You must provide either:
      - a LightningDataModule (datamodule=...), or
      - a factory that returns a fresh DataModule per trial (datamodule_factory=...).

    Optional model_mutator lets you modify the passed model per trial
    (e.g., set dropout p).
    """
    def __init__(
        self,
        *,
        datamodule: pl.LightningDataModule | None = None,
        datamodule_factory: tp.Callable[[], pl.LightningDataModule] | None = None,

        # search/optimization
        target_metric: str,                 # e.g., "auprc", "auroc", "f1"
        number_of_trials: int,
        epoch_number: tuple[int, int],
        learning_rates: tuple[float, float],
        optimizers: list[str],
        dropout_probability: tuple[float, float] | None = None,
        model_mutator: tp.Callable[[optuna.Trial, nn.Module, float | None], None] | None = None,
        pruner: optuna.pruners.BasePruner | None = None,
        early_stopping_patience: int = 5,

        # IO
        save_path: str | Path,
        save_ckpt_path: str | Path | None = None,

        # Trainer/device
        accelerator: str = "auto",
        devices: int | list[int] | str = 1,
        seed: int | None = 42,
        max_time: str | None = None,
        log_every_n_steps: int = 10,
    ):
                # Data
        assert datamodule is not None or datamodule_factory is not None, \
            "Provide datamodule or datamodule_factory."

        self.datamodule = datamodule
        self.datamodule_factory = datamodule_factory

        # Search
        self.target_metric = target_metric
        self.monitor = f"val_{target_metric}"
        self.number_of_trials = number_of_trials

        assert len(epoch_number) == 2
        assert len(learning_rates) == 2
        if dropout_probability is not None:
            assert len(dropout_probability) == 2

        self.epoch_number = epoch_number
        self.learning_rates = learning_rates
        self.optimizers = optimizers
        self.dropout_probability = dropout_probability
        self.model_mutator = model_mutator
        self.pruner = pruner or optuna.pruners.MedianPruner(n_startup_trials=5)
        self.early_stopping_patience = early_stopping_patience

        # IO
        self.save_path = str(save_path)
        os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)

        self.save_ckpt_path = (
            str(save_ckpt_path) if save_ckpt_path is not None else None
        )
        if self.save_ckpt_path is not None:
            os.makedirs(os.path.dirname(self.save_ckpt_path) or ".", exist_ok=True)

        # Trainer / device
        self.accelerator = accelerator
        self.devices = devices
        self.seed = seed
        self.max_time = max_time
        self.log_every_n_steps = log_every_n_steps

        if seed is not None:
            pl.seed_everything(seed, workers=True)

        # Internals
        self._base_model: nn.Module | None = None
        self._base_state: dict[str, torch.Tensor] | None = None

        self.study = optuna.create_study(
            direction="maximize",
            pruner=self.pruner
        )

        self.best_score: float | None = None
        self.best_params: dict[str, tp.Any] | None = None
        self._best_ckpt_tmp: str | None = None

    def set_model(self, model: nn.Module) -> None:
        """Provide the base model; captures its initial weights."""
        self._base_model = model
        self._base_state = copy.deepcopy(model.state_dict())

    def run(
        self,
        model: nn.Module | None = None,
    ):
        if model is not None:
            self.set_model(model)

        if self._base_model is None:
            raise RuntimeError("No model provided. Use run(model=...) or set_model().")

        optuna.logging.disable_default_handler()
        self.study.optimize(
            self._objective, 
            n_trials=self.number_of_trials,
            show_progress_bar=True,
        )

        best_model = None
        best_ckpt_path = None

        if self._best_ckpt_tmp:
            # Strip "model." prefix saved by LightningModule for the inner nn.Module
            ckpt = torch.load(self._best_ckpt_tmp, map_location="cpu", weights_only=False) 
            state_dict = ckpt["state_dict"]
            inner_sd = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    inner_sd[k[len("model."):]] = v
            if inner_sd:
                state_dict = inner_sd

            # Load into a fresh clone of your base model
            model_for_load = self._clone_fresh_model()
            model_for_load.load_state_dict(state_dict, strict=True)

            best_model = model_for_load
            torch.save(best_model.state_dict(), self.save_path)

            if self.save_ckpt_path is not None:
                shutil.copy(self._best_ckpt_tmp, self.save_ckpt_path)
                best_ckpt_path = self.save_ckpt_path

        return (
            self.study,
            best_model,
            self.best_params,
            self.best_score,
            best_ckpt_path,
        )
    
    def _build_datamodule(self):
        if self.datamodule is not None:
            return self.datamodule
        return self.datamodule_factory()

    def _clone_fresh_model(self):
        m = copy.deepcopy(self._base_model)
        m.load_state_dict(self._base_state, strict=True)
        return m

    def _auto_set_dropout(self, model, p):
        for mod in model.modules():
            if isinstance(mod, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                mod.p = p

    def _objective(self, trial: optuna.Trial) -> float:
        # Hyperparameters
        epochs = trial.suggest_int("epochs", *self.epoch_number)
        lr = trial.suggest_float("lr", *self.learning_rates, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", self.optimizers)

        dropout_p = None
        if self.dropout_probability is not None:
            dropout_p = trial.suggest_float(
                "dropout_probability", *self.dropout_probability
            )

        # Fresh model
        torch_model = self._clone_fresh_model()
        if dropout_p is not None:
            if self.model_mutator is not None:
                self.model_mutator(trial, torch_model, dropout_p)
            else:
                self._auto_set_dropout(torch_model, dropout_p)

        # Lightning module
        lightning_module = SingleProtPRILightningModule(
            model=torch_model,
            optimization_harness={"optimizer": optimizer_name, "lr": lr},
            loss_fn=nn.BCEWithLogitsLoss(),
        )

        # Data + callbacks
        datamodule = self._build_datamodule()

        checkpoint_cb = ModelCheckpoint(
            monitor=self.monitor,
            mode="max",
            save_top_k=1,
            filename="best-{epoch:02d}-{" + self.monitor + ":.4f}",
            verbose=False
        )
        early_stop_cb = EarlyStopping(
            monitor=self.monitor,
            mode="max",
            patience=self.early_stopping_patience,
            verbose=False
        )
        pruning_cb = PyTorchLightningPruningCallback(trial, monitor=self.monitor)
        history_cb = _MetricHistory(self.monitor)

        # Trainer
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator=self.accelerator,
            devices=self.devices,
            callbacks=[checkpoint_cb, early_stop_cb, pruning_cb, history_cb],
            log_every_n_steps=self.log_every_n_steps,
            max_time=self.max_time,
            enable_progress_bar=False,
            logger=False,
            enable_model_summary=False,
        )

        # Fit
        trainer.fit(lightning_module, datamodule=datamodule)

        # Score
        if self.monitor not in trainer.callback_metrics:
            raise optuna.TrialPruned(f"Metric {self.monitor} not available.")

        score = float(trainer.callback_metrics[self.monitor])

        # Track best
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            best = {"epochs": epochs, "lr": lr, "optimizer": optimizer_name}
            if dropout_p is not None:
                best["dropout_probability"] = dropout_p
            self.best_params = best

            best_ckpt = checkpoint_cb.best_model_path
            self._best_ckpt_tmp = (
                best_ckpt if best_ckpt and os.path.exists(best_ckpt) else None
            )

        trial.set_user_attr("val_metric_history", history_cb.values)
        return score


def quiet_mode(): 
    # 1) Silence Lightning loggers 
    for name in (
        "pytorch_lightning", 
        "lightning", 
        "lightning.pytorch", 
        "lightning_fabric", 
        "lightning.pytorch.utilities.seed", 
        "lightning_fabric.utilities.seed", 
    ): 
        lg = logging.getLogger(name) 
        lg.setLevel(logging.CRITICAL) 
        lg.propagate = False 
        lg.handlers.clear() 
        lg.addHandler(logging.NullHandler())

        # 2) Silence Lightning warnings (num_workers, checkpoint dir, etc.)
        warnings.filterwarnings("ignore", module="pytorch_lightning")
        warnings.filterwarnings("ignore", module="lightning")
        warnings.filterwarnings("ignore", message="The '.*_dataloader' does not have many workers.*")
        warnings.filterwarnings("ignore", message="Checkpoint directory .* exists and is not empty.*")

        # 3) Silence Optuna logs but keep its progress bar
        optuna.logging.disable_default_handler()            # remove default stream handler
        optuna.logging.set_verbosity(optuna.logging.WARNING)  # WARNING or ERROR


