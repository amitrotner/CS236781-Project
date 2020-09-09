import abc
import os
import sys
import tqdm
import torch
import numpy as np

from torch.utils.data import DataLoader
from typing import Callable, Any
from cs236781.train_results import BatchResult, EpochResult, FitResult
import copy
import utils
from clustering import calculate_predictions


class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, loss_fn, optimizer, scheduler=None, device=None, autoencoder_training=None,
                 second_loss_fn=None, update_interval=None, target_dist=None, gamma=None):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.loss_fn2 = second_loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.autoencoder_training = autoencoder_training
        self.update_interval = update_interval
        self.target_dist = target_dist
        self.gamma = gamma

        if self.device:
            model.to(self.device)

    def fit(self, dl_train: DataLoader, dl_test: DataLoader,
            num_epochs, checkpoints: str = None,
            early_stopping: int = None,
            print_every=1, **kw) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :return: A FitResult object containing train and test losses per epoch.
        """
        dl = dl_train

        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc = [], [], [], []

        best_acc = None
        epochs_without_improvement = 0
        current_learning_rate = self.scheduler.get_lr()

        for epoch in range(num_epochs):
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f'--- EPOCH {epoch + 1}/{num_epochs} ---', verbose)
            if current_learning_rate != self.scheduler.get_lr():
                current_learning_rate = self.scheduler.get_lr()
                self._print(f' learning rate has been changed to {current_learning_rate}', verbose)

            actual_num_epochs += 1
            train = self.train_epoch(dl_train, epoch, verbose=verbose, **kw)
            train_loss += [torch.mean(torch.stack(train.losses)).item()]

            if self.autoencoder_training:
                train_acc += [0]
            else:
                train_acc += [train.accuracy.item()]

            if self.scheduler:
                self.scheduler.step()

            if self.autoencoder_training is None:
                test = self.test_epoch(dl_test, epoch,  verbose=verbose, **kw)
                test_loss += [torch.mean(torch.stack(test.losses)).item()]
                if self.autoencoder_training:
                    test_acc += [0]
                else:
                    test_acc += [test.accuracy.item()]

                if epoch >= 1 and test_loss[-1] < test_loss[-2]:
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if early_stopping is not None and early_stopping == epochs_without_improvement:
                    break

        return FitResult(actual_num_epochs,
                         train_loss, train_acc, test_loss, test_acc)

    def train_epoch(self, dl_train: DataLoader, epoch, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, epoch, **kw)

    def test_epoch(self, dl_test: DataLoader, epoch, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, epoch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(dl: DataLoader,
                       forward_fn: Callable[[Any], BatchResult], epoch,
                       verbose=True, max_batches=None) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, 'w')

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data, batch_idx, epoch, dl)

                pbar.set_description(f'{pbar_name} ({batch_res.loss:.3f})')
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct

            avg_loss = sum(losses) / num_batches
            accuracy = 100. * num_correct / num_samples
            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss:.3f}, '
                                 f'Accuracy {accuracy:.1f})')

        return EpochResult(losses=losses, accuracy=accuracy)


class TorchTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer, scheduler=None, device=None, autoencoder_training=None,
                 second_loss_fn=None, update_interval=None, target_dist=None, gamma=None):
        super().__init__(model, loss_fn, optimizer, scheduler, device, autoencoder_training, second_loss_fn,
                         update_interval, target_dist, gamma)

    def train_batch(self, batch, batch_idx=None, epoch_num=None, dl=None) -> BatchResult:
        dimension = batch.shape[1] - 1
        # We train the autoencoder in an unsupervised fashion. Therefore, there are no labels.
        X = batch[:, list(range(dimension))]
        y = batch[:, dimension].unsqueeze(-1).long()

        if self.update_interval:
            # Update target distribution, check and print performance
            if (batch_idx - 1) % self.update_interval == 0 and not (batch_idx == 1 and epoch_num == 0):
                output_distribution, labels, preds_prev = calculate_predictions(self.model, copy.deepcopy(dl),
                                                                                self.device)
                self.target_dist = target(output_distribution)
                labels = labels.squeeze(-1)
                nmi = utils.metrics.nmi(labels, preds_prev)
                ari = utils.metrics.ari(labels, preds_prev)
                acc = utils.metrics.acc(labels, preds_prev)
                if epoch_num % 100 == 0 or epoch_num == 599:
                    print('NMI: {0:.5f}\tARI: {1:.5f}\tAcc {2:.5f}\n'.format(nmi, ari, acc))

        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        if self.loss_fn2:
            tar_dist = self.target_dist[(batch_idx * batch.shape[0]):((batch_idx + 1) * batch.shape[0]), :]
            tar_dist = torch.from_numpy(tar_dist).to(self.device)

        # Forward pass
        if self.autoencoder_training:
            X_rec, clusters, _ = self.model(X)
            loss = self.loss_fn.forward(X_rec, X)
            if self.loss_fn2:
                loss2 = self.loss_fn2.forward(torch.log(clusters), tar_dist) / batch.shape[0]
                loss += self.gamma * loss2
        else:
            output = self.model.forward(X)
            loss = self.loss_fn.forward(output, y)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Optimize params
        self.optimizer.step()

        if self.autoencoder_training:
            num_correct = 0
        # If not training the autoencoder, calculate number of correct predictions
        else:
            num_correct = torch.sum(torch.max(output, dim=1)[1] - y == 0)
        # ========================

        return BatchResult(loss, num_correct)

    def test_batch(self, batch, batch_idx=None, epoch_num=None, dl=None) -> BatchResult:
        dimension = batch.shape[1] - 1
        # We train the autoencoder in an unsupervised fashion. Therefore, there are no labels.
        if self.autoencoder_training:
            X = batch
            y = X
        else:
            y = batch[:, dimension].unsqueeze(-1).long()
            X = batch[:, list(range(dimension))]
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        with torch.no_grad():
            if self.autoencoder_training:
                enc = self.model.encoder(X)
                output = self.model.decoder(enc)
                loss1 = self.loss_fn.forward(output, y)
                loss2 = self.loss_fn2.forward(enc.squeeze(-1), target(enc.squeeze(-1)))
                loss = loss1 + loss2
                num_correct = 0
            else:
                out = self.model.forward(X)
                loss = self.loss_fn.forward(out, y)
                num_correct = torch.sum(torch.max(out, dim=1)[1] - y == 0)
            # ========================

        return BatchResult(loss, num_correct)


# Calculate target distribution
def target(out_distr):
    tar_dist = out_distr ** 2 / np.sum(out_distr, axis=0)
    tar_dist = np.transpose(np.transpose(tar_dist) / np.sum(tar_dist, axis=1))
    return tar_dist
