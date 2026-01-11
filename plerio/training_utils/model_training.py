from tqdm.auto import tqdm

from scipy.stats import pearsonr
from sklearn.metrics import (accuracy_score,
                             balanced_accuracy_score,
                             f1_score,
                             matthews_corrcoef,
                             precision_score,
                             recall_score,
                             roc_auc_score)
from pathlib import Path

import os
import pickle
import typing as tp
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import random

import optuna
import torch
import torch.nn as nn

import subprocess


def set_seed(seed: int):
    """
    Sets the seed for reproducibility across numpy, random, torch.
    :param seed: random seed.
    """
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # Torch for CPU and device-agnostic
    torch.cuda.manual_seed(seed)  # GPU: CUDA
    torch.cuda.manual_seed_all(seed)  # Multi-GPU
    torch.backends.cudnn.deterministic = True  # Deterministic behavior for CUDA
    torch.backends.cudnn.benchmark = False  # Disable cuDNN benchmarking


set_seed(43)


def cpu_num(array):
    return array.cpu().detach().numpy()


def beautiful_printout(func: tp.Callable) -> tp.Callable:
    def inner(instance, fold: str) -> dict:
        metrics = func(instance, fold)
        print(f"{fold} loss: {metrics['BCE_Loss'][-1]:.04f}")
        print(
            f"{fold} bal_acc: {metrics['balanced_accuracy_score'][-1]:.04f}")
        print(f"{fold} roc_auc: {metrics['roc_auc_score'][-1]:.04f}")
        print(f"{fold} mcc: {metrics['matthews_corrcoef'][-1]:.04f}")
        print()
        return metrics
    return inner


class BinaryClassificationMetricHolder:
    """
    A class for metrics measuring on a single fold (e.g., train or val).
    For each metric `BinaryClassificationMetricHolder` calculates it on
    the every batch and then, at the end of epoch, obtains mean epoch metric
    value through averaging across accumulated values. Note that all metrics
    calculated on the batches during the current epoch will be deleted at its
    end.
    """
    def __init__(self,
                 metrics_list:
                 list[tp.Callable[[np.ndarray, np.ndarray], float]]
                 | None = None
                 ) -> None:
        """
        :param metrics_list: list of sklearn metrics to calculate for each
        batch and epoch.
        """
        if metrics_list is None:
            metrics_list = [
                accuracy_score,
                balanced_accuracy_score,
                f1_score,
                matthews_corrcoef,
                precision_score,
                recall_score,
                roc_auc_score
            ]

        self.name_to_func = dict(zip([metric.__name__ for metric in metrics_list],
                                     metrics_list))
        self.current_epoch_metrics = {metric.__name__: []
                                      for metric in metrics_list}
        self.bce = nn.BCEWithLogitsLoss()
        self.current_epoch_metrics.update({'BCE_Loss': []})
        self.averaged_metrics = {metric.__name__: []
                                 for metric in metrics_list}
        self.averaged_metrics.update({'BCE_Loss': []})
        self.requires_probabilities = {roc_auc_score.__name__,
                                       pearsonr.__name__}

    def add_batch_metrics(self, predictions, y_true) -> None:
        """
        Calculates metrics on the current batch. Since they are needed just
        to average them at the end of epoch, it is possible to store their
        sum instead of all metrics. But the implementation below store all
        the metrics values of current epoch till the end of if because it
        might be useful.
        :param predictions: model predictions on the new batch.
        :param y_true: true labels
        :return: `None`
        """
        y_true = y_true.view(-1, 1)
        # print(predictions, y_true)
        self.current_epoch_metrics['BCE_Loss'].append(
            float(self.bce(predictions, y_true).item())
        )

        predictions = cpu_num(predictions.sigmoid())
        y_true = cpu_num(y_true)
        for metric in self.current_epoch_metrics:
            if metric == 'BCE_Loss':
                continue
            if not metric in self.requires_probabilities:
                self.current_epoch_metrics[metric].append(
                    float(self.name_to_func[metric](
                            y_true, 
                            (predictions >= 0.5).astype(int)
                        )
                    )
                )
            else:
                try:
                    metric_value = float(
                        self.name_to_func[metric](
                            y_true,
                            predictions
                        )
                    )
                except ValueError: # ROC_AUC case
                    metric_value = 0
                self.current_epoch_metrics[metric].append(
                    float(metric_value)
                )

    def add_epoch_metrics(self) -> dict[str, list[float]]:
        """
        This method should be called only at the end of each epoch.
        For each metric, it averages all the metric values calculated
        during this epoch and then delete this values.
        :return:
        """
        for metric in self.current_epoch_metrics:
            mean_value = np.mean(self.current_epoch_metrics[metric])
            # It is necessary to clear current epoch metrics
            self.current_epoch_metrics[metric] = []
            self.averaged_metrics[metric].append(mean_value)
        return self.averaged_metrics


class BatchProcessor:
    """
    A class to process unpickled batch of data according to the mode:
    `BatchProcessor` with `multi_protein_mode=True` expects unpickled batch
    to be tuple of `(rna_batch, protein_batch, label_batch)` and requires
    protein embeddings and frequencies while `BatchProcessor` with
    `multi_protein_mode=False` expects unpickled batch to be tuple of
    `(rna_batch, label_batch)` and doesn't need protein embeddings and
    frequencies dealing with single-protein model.
    """
    def __init__(self, multi_protein_mode: bool = False,
                 rna_input_dim: int = 1088,
                 rna_embedding_size: tp.Optional[int] = None,
                 deterministic_hash: bool = True,
                 protein_embeddings: dict[str, torch.Tensor] | None = None,
                 protein_weights: dict[str, float] | None = None,
                 device: torch.device = 'cuda') -> None:
        """
        Initializes the batch processor.
        :param multi_protein_mode: single or multi-protein mode,
         default: single
        :param rna_input_dim: size of input rna embedding vector,
         default: 1088
        :param rna_embedding_size: output size of RNA embedding vector,
         if passed, should be prime. If no value passed for this argument,
         no hashing procedure will be applied.
        :param deterministic_hash: whether to use deterministic hashing
         or sample a random one.
        :param protein_embeddings: protein vector representations.
        :param protein_weights: weight for each protein.
        :param device: device to put the output batch.
        """

        self.multi_protein_mode = multi_protein_mode
        self.rna_input_dim = rna_input_dim
        self.rna_embedding_size = rna_embedding_size
        self.deterministic_hash = deterministic_hash
        self.protein_embeddings = protein_embeddings
        self.protein_weights = protein_weights
        if self.deterministic_hash:
            self.hash = lambda x: (26855093 * x + 796233790) % (10 ** 9 + 7)
        else:
            self.hash = self.sample_hash_()

        indices = torch.arange(rna_input_dim, dtype=torch.long)
        if self.rna_embedding_size is not None:
            self.ind2hashval = self.hash(indices) % self.rna_embedding_size
        self.device = device

    def __call__(self, unpickled_tuple) -> tuple[torch.Tensor, ...]:
        """
        High-level API to process unpickled batch.
        :param unpickled_tuple: tuple of
         `(rna_batch, protein_batch, label_batch)` or
         `(rna_batch, label_batch)` according to the mode.
        :return: parsed and processed batch with all tensors moved
         to the needed `device`.
        """
        if self.multi_protein_mode:
            return self.process_multi_protein_batch_(unpickled_tuple)
        else:
            return self.process_single_protein_batch_(unpickled_tuple)


    def process_single_protein_batch_(self, unpickled_tuple) -> (
            tuple)[torch.Tensor, torch.Tensor]:
        """
        Process unpickled batch according to the single-protein strategy,
        technically, just moves tensors to the `device`.
        :param unpickled_tuple: tuple of `(rna_batch, label_batch)`
        :return: processed `(rna_batch, label_batch)`
        """
        rna_batch, y_true = unpickled_tuple
        if self.rna_embedding_size is not None:
            rna_batch = self.hash_kmer_batch(rna_batch)
        # print(rna_batch)
        # print(y_true)
        rna_batch = rna_batch.to(self.device)

        y_true = y_true.to(self.device)
        return rna_batch, y_true

    def process_multi_protein_batch_(self, unpickled_tuple) -> (
            tuple)[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process unpickled batch according to the multi-protein strategy.
        It concatenates protein embeddings with RNA embeddings to obtain
        pair embedding and creates a batch of protein weights.
        :param unpickled_tuple: tuple of
         `(rna_batch, protein_name_batch, label_batch)`
        :return: `(pair_emb_batch, protein_weights_batch, label_batch)`
        """
        rna_batch, prot_names, y_true = unpickled_tuple
        weight_batch = torch.Tensor([self.protein_weights[prot_name]
                                     for prot_name in prot_names])
        weight_batch /= sum(weight_batch)
        weight_batch = weight_batch.view(-1, 1)

        prot_batch = torch.cat([self.protein_embeddings[prot_name]
                                for prot_name in prot_names], dim=0)

        pair_batch = torch.cat([rna_batch, prot_batch], dim=1)

        pair_batch = pair_batch.to(self.device)
        weight_batch = weight_batch.to(self.device)
        y_true = y_true.to(self.device)
        return pair_batch, weight_batch, y_true

    def hash_kmer_batch(self, rna_batch: torch.Tensor) -> torch.Tensor:
        hashed_rna_batch = torch.zeros(rna_batch.shape[0],
                                       self.rna_embedding_size)
        for i in range(rna_batch.shape[1]):
            hashed_rna_batch[:, self.ind2hashval[i]] += rna_batch[:, i]
        return hashed_rna_batch

    @staticmethod
    def sample_hash_(seed: float = 42,
                     max_int: int = 10 ** 9 + 7
                     ) -> tp.Callable[[int], int]:
        random.seed(seed)
        a_coeff = random.randint(1, max_int)
        b_coeff = random.randint(0, max_int)

        def hash_(x: torch.Tensor | int) -> torch.Tensor | int:
            hash_value = (a_coeff * x + b_coeff) % max_int
            return hash_value

        return hash_


class ModelTrainer(nn.Module):
    """
    A class to train single- or multiprotein model given all the
    hyperparameters.
    """
    def __init__(self,
                 train_path: Path | str,
                 val_path: Path | str,
                 test_path: Path | str,
                 save_path: Path | str,
                 model: nn.Module,
                 loss_function:
                 tp.Callable[[torch.Tensor, ...], torch.Tensor],
                 optimizer: tp.Any,
                 lr_scheduler: tp.Any,
                 number_of_epochs: int,
                 multi_protein_mode: bool = False,
                 rna_input_dim: int = 1088,
                 rna_embedding_size: tp.Optional[int] = None,
                 metrics_to_measure:
                 list[tp.Callable[[np.ndarray, np.ndarray], float]]
                 | None = None,
                 device: str | torch.device = 'cuda',
                 protein_embeddings: dict[str, torch.Tensor] = None,
                 protein_weights: dict[str, float] = None
                 ) -> None:
        """
        :param multi_protein_mode: single or multi-protein model will be
         trained.
        :param train_path: path to train data.
        :param val_path: path to validation data.
        :param protein_embeddings: dictionary from protein names to embeddingss.
        :param protein_weights: dictionary from protein names to weights.
        :param model: model, supports models that takes concatenated protein
         embedding and rna embedding as input.
        :param loss_function: custom loss function that supports protein
         weights.
        :param optimizer: initialized optimizer.
        :param lr_scheduler: initialized learning rate scheduler.
        :param number_of_epochs: number of epochs to train the model.
        :param metrics_to_measure: list of metrics to measure, expect
         list of functions from sklearn.metrics.
        :param device: device, 'cpu', 'cuda' or like that.
        """
        super().__init__()
        self.multi_protein_mode = multi_protein_mode
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.save_path = save_path
        self.device = device

        self.protein_embeddings = protein_embeddings
        self.protein_weights = protein_weights

        self.rna_embedding_size = rna_embedding_size
        self.rna_input_dim = rna_input_dim
        self.batch_processor = BatchProcessor(
            multi_protein_mode,
            rna_input_dim,
            rna_embedding_size,
            protein_embeddings,
            protein_weights,
            self.device
        )

        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.number_of_epochs = number_of_epochs

        self.train_metrics_holder = (
            BinaryClassificationMetricHolder(metrics_to_measure))
        self.val_metrics_holder = (
            BinaryClassificationMetricHolder(metrics_to_measure))

    # @beautiful_printout
    def run_epoch_(self, fold: str) -> dict[str, list[float]]:
        """

        :param fold: train or val
        :return:
        """
        if fold == 'train':
            data_path = self.train_path
        elif fold == 'val':
            data_path = self.val_path
        else:
            data_path = self.test_path

        files = os.listdir(data_path)
        random.shuffle(files)

        for filename in files:
            with open(os.path.join(data_path, filename), 'rb') as file:
                unpickled_batch = pickle.load(file)
            # print(unpickled_batch)
            batch = self.batch_processor(unpickled_batch)
            inp_for_model = batch[0]
            y_true = batch[-1]

            if fold == 'train':
                self.train()
                y_pred = self.model(inp_for_model)
                loss = self.calculate_loss_(y_true, y_pred, *batch[1:-1])
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            else:
                self.model.eval()
                with torch.no_grad():
                    y_pred = self.model(inp_for_model)
                    loss = self.calculate_loss_(y_true, y_pred, *batch[1:-1])

            # print(f'{fold} Loss: ', loss.item())
            # print('y_pred')
            # print(y_pred)
            # print('y_true')
            # print(y_true)
            # print()
            # # TODO: handle NaN more accurate
            # y_pred = y_pred.nan_to_num()
            # probabilities = cpu_num(y_pred)
            # y_true = cpu_num(y_true)

            if fold == 'train':
                self.train_metrics_holder.add_batch_metrics(y_pred, y_true)
            else:
                self.val_metrics_holder.add_batch_metrics(y_pred, y_true)

        if fold == 'train':
            return self.train_metrics_holder.add_epoch_metrics()
        return self.val_metrics_holder.add_epoch_metrics()
    
    def run_saliency_calculation(self) -> torch.Tensor:
       
        data_path = self.train_path

        files = os.listdir(data_path)
        random.shuffle(files)

        grads = []
        for filename in files:
            with open(os.path.join(data_path, filename), 'rb') as file:
                unpickled_batch = pickle.load(file)
            # print(unpickled_batch)
            batch = self.batch_processor(unpickled_batch)
            inp_for_model = batch[0]
            inp_for_model.requires_grad = True
            y_true = batch[-1]

            y_pred = self.model(inp_for_model)
            loss = self.calculate_loss_(y_true, y_pred, *batch[1:-1])
            loss.backward()

            grads.append(torch.abs(batch[0].grad).mean(dim=0).detach().cpu())
            self.model.zero_grad()

        return torch.stack(grads).mean(dim=0)


    def calculate_loss_(self, *args):
        if self.multi_protein_mode:
            loss = self.loss_function(*args)
        else:
            y_pred = args[1]
            y_true = args[0].view(-1, 1)
            # print(y_pred.shape, y_true.shape)
            loss = self.loss_function(y_pred, y_true)
        return loss

    def train_model(self) -> tuple[dict[str, list[float]],
    dict[str, list[float]]]:
        for epoch in range(self.number_of_epochs):
            self.run_epoch_('val')
            self.run_epoch_('train')
        # subprocess.call(f'touch {self.save_path}', shell=True)
        torch.save(self.model.state_dict(), self.save_path)
        return (self.train_metrics_holder.averaged_metrics,
                self.val_metrics_holder.averaged_metrics)


# TODO пристрой куда-нибудь
def parse_protein_frequencies(file: str) -> dict[str, float]:
        """
        Parse a file with protein names and their frequency weights
        to the dictionary from protein to corresponding weight.
        :param file: path-like, absolute or relative path to a file
        to parse. If the path is relative,
        :return: mapping from protein names to their weights
        """
        prot_to_weight = {}
        with open(file, 'r') as file:
            for line in file:
                prot, weight = line.split()
                prot_to_weight[prot] = float(weight)
        return prot_to_weight


class OptunaModelTrainerWrapper:
    """
    A class that consider a single `ModelTrainer` as a model for
    hyperparameter tuning. By the end of the tuning, the `optuna.study`
    result, metrics and the best model are returned.
    """
    def __init__(self,
                 multi_protein_mode: bool,
                 model_type: str,
                 train_path: Path | str,
                 val_path: Path | str,
                 test_path: Path | str,

                 save_path: Path | str,

                 target_metric: str,
                 number_of_trials: int,

                 rna_emb_dim: int | None,

                 epoch_number: tuple[int, int],
                 dropout_probability: tuple[float, float],
                 learning_rates: tuple[float, float],
                 optimizers: list[str],
                 lr_schedulers: list[str],

                 device: torch.device | str = 'cuda') -> None:
        """
        :param multi_protein_mode: single or multi-protein model will be
         trained.
        :param train_path: path to train data.
        :param val_path: path to validation data.
        :param test_path: path to test data.

        :param save_path: path to save the best model.

        :param target_metric: metric to maximize.
        :param number_of_trials: number of trials for `optuna`.

        :param epoch_number: segment where to search for optimal training
         epoch number.
        :param dropout_probability: segment where to search for optimal
         dropout probability. Note that no log scale will be used.
        :param learning_rates: segment where to search for optimal
         learning rate. Note that the study will be done using log scale.
        :param optimizers: list of optimizers to try. Note that optimizers
         would be obtained from `torch.optim` using `getattr`.
        :param lr_schedulers: initialized learning rate schedulers to try.
         Initialized ones are required because each scheduler have different
         parameters with different meaning.

        :param device: device, 'cpu', 'cuda' or like that.
        """
        assert len(epoch_number) == 2, \
            'integer hyperparameter requires a segment'
        assert len(dropout_probability) == 2,\
            'float hyperparameter requires a segment'
        assert len(learning_rates) == 2, \
            'float hyperparameter requires a segment'

        self.multi_protein_mode = multi_protein_mode

        assert model_type in ['linreg', 'mlp']
        self.model_type = model_type
        self.train_path: Path | str = train_path
        self.val_path: Path | str = val_path
        self.test_path: Path | str = test_path
        self.save_path: Path | str = save_path
        os.makedirs(
            os.path.dirname(self.save_path),
            exist_ok=True
        )

        self.rna_emb_dim = rna_emb_dim

        self.target_metric = target_metric
        self.number_of_trials = number_of_trials

        self.epoch_number = epoch_number
        self.dropout_probability = dropout_probability
        self.learning_rates = learning_rates
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.device = device
        self.study = optuna.create_study(
            direction='maximize',
            # storage=self.save_path,
            study_name='Optuna study'
        )
        optuna.logging.disable_default_handler()

        self.best_score = None
        self.best_model = None
        self.best_metrics = (None, None)

    def run(
        self,
    ) -> tuple[optuna.study, nn.Module,
    tuple[dict[str, list[float]]], dict[str, list[float]]]:
        self.study.optimize(
            self.train_model_,
            n_trials=self.number_of_trials,
        )
        torch.save(self.best_model.state_dict(), self.save_path)
        return self.study, self.best_model, self.best_metrics

    def train_model_(self, trial) -> tp.Any:
        dropout_p = trial.suggest_float('dropout_probability',
                                        *self.dropout_probability)
        if self.multi_protein_mode:
            prot_emb_size = 1024
        else:
            prot_emb_size = 0

        rna_dim = self.rna_emb_dim if self.rna_emb_dim is not None else 1088

        if self.model_type == 'mlp':
            model = nn.Sequential(nn.Linear(rna_dim + prot_emb_size, 512),
                                nn.ReLU(),
                                nn.Dropout(dropout_p),
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Dropout(dropout_p),
                                nn.Linear(256, 128),
                                nn.ReLU(),
                                nn.Dropout(dropout_p),
                                nn.Linear(128, 1)).to(self.device)
        else:
            model = nn.Linear(rna_dim + prot_emb_size, 1).to(self.device)

        epochs = trial.suggest_int('epochs', *self.epoch_number)
        optimizer_name = trial.suggest_categorical(
            'optimizer', self.optimizers)
        lr = trial.suggest_float('lr', *self.learning_rates,
                                 log=True)
        optimizer = getattr(torch.optim, optimizer_name)(
            model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
        #                                                 max_lr=0.01,
        #                                                 steps_per_epoch=10,
        #                                                 epochs=epochs)
        # TODO: replace with different schedulers

        model_trainer = ModelTrainer(
            multi_protein_mode=self.multi_protein_mode,
            train_path=self.train_path,
            val_path=self.val_path,
            test_path=self.test_path,
            save_path=self.save_path,
            model=model,
            loss_function=nn.BCEWithLogitsLoss(),
            optimizer=optimizer,
            lr_scheduler=scheduler,
            number_of_epochs=epochs,
            rna_input_dim=1088,
            rna_embedding_size=self.rna_emb_dim,
            device=self.device,
            metrics_to_measure=None,
            protein_embeddings=None,
            protein_weights=None
        )

        prev_target_metric = 0
        for epoch in range(epochs):
            model_trainer.run_epoch_('val')
            model_trainer.run_epoch_('train')
            subprocess.call(f'touch {self.save_path}', shell=True)
            target_metric = (
                model_trainer.val_metrics_holder.averaged_metrics)[
                self.target_metric][-1]
            if self.best_score is None or target_metric > self.best_score:
                self.best_score = target_metric
                self.best_model = model_trainer
                self.best_metrics = \
                    (model_trainer.train_metrics_holder.averaged_metrics,
                     model_trainer.val_metrics_holder.averaged_metrics)
            if epoch != 0 and abs(prev_target_metric - target_metric) < 5e-3:
                return target_metric

            prev_target_metric = target_metric
            # if epoch > epoch / 2 and target_metric < 0.4:
            #     raise optuna.TrialPruned()
        return target_metric
