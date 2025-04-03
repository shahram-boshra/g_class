# training_utils.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from models import MGModel
from config_loader import Config
import logging

logger = logging.getLogger(__name__)

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.0, path='chk_learn.pt'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, valid_loss, model):
        if self.best_score is None:
            self.best_score = valid_loss
            self.save_model_state(valid_loss, model)
        elif valid_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = valid_loss
            self.save_model_state(valid_loss, model)
            self.counter = 0

    def save_model_state(self, valid_loss, model):
        if self.verbose:
            logger.info(f"Validation loss decreased ({self.best_score:.6f} --> {valid_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), self.path)

class TrainingLoop:
    def __init__(self, model, criterion, optimizer, step_lr, device, l1_lambda):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.step_lr = step_lr
        self.device = device
        self.l1_lambda = l1_lambda
        logger.debug("TrainingLoop initialized.")

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        num_graphs = 0

        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            out, l1_reg = self.model(batch)
            target = batch.y
            loss = self.criterion(out, target)
            loss += l1_reg * self.l1_lambda
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            num_graphs += batch.num_graphs
        self.step_lr.step()
        avg_loss = total_loss / num_graphs
        logger.debug(f"Training Epoch Loss: {avg_loss:.4f}")
        return avg_loss

    def validate_epoch(self, valid_loader):
        self.model.eval()
        total_loss = 0
        num_graphs = 0
        all_targets = []
        all_predictions = []
        all_probs = []

        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(self.device)
                out, _ = self.model(batch)
                target = batch.y
                loss = self.criterion(out, target)
                total_loss += loss.item() * batch.num_graphs
                num_graphs += batch.num_graphs
                all_targets.append(target.cpu().numpy())
                all_predictions.append(torch.argmax(out, dim=1).cpu().numpy())
                all_probs.append(torch.softmax(out, dim=1).cpu().numpy())

        avg_loss = total_loss / num_graphs
        all_targets = np.concatenate(all_targets, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)

        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        auc = roc_auc_score(all_targets, all_probs[:, 1], average='weighted')

        logger.debug(f"Validation Epoch Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        return avg_loss, accuracy, precision, recall, f1, auc

    def test_epoch(self, test_loader, return_predictions=False):
        self.model.eval()
        total_loss = 0
        num_graphs = 0
        all_targets = []
        all_predictions = []
        all_probs = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                out, _ = self.model(batch)
                target = batch.y
                loss = self.criterion(out, target)
                total_loss += loss.item() * batch.num_graphs
                num_graphs += batch.num_graphs
                all_targets.append(target.cpu().numpy())
                all_predictions.append(torch.argmax(out, dim=1).cpu().numpy())
                all_probs.append(torch.softmax(out, dim=1).cpu().numpy())

        avg_loss = total_loss / num_graphs
        all_targets = np.concatenate(all_targets, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)

        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        auc = roc_auc_score(all_targets, all_probs[:, 1], average='weighted')
        logger.info(f"Test Epoch Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        if return_predictions:
            return avg_loss, accuracy, precision, recall, f1, auc, all_targets, all_predictions
        return avg_loss, accuracy, precision, recall, f1, auc

class Trainer:
    def __init__(self, model, criterion, optimizer, step_lr, red_lr, early_stopping, config, device):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.step_lr = step_lr
        self.red_lr = red_lr
        self.early_stopping = early_stopping
        self.config = config
        self.device = device
        self.training_loop = TrainingLoop(self.model, self.criterion, self.optimizer, self.step_lr, self.device, self.config.model.l1_regularization_lambda)
        logger.debug("Trainer initialized.")

    def train_and_validate(self, train_loader, valid_loader):
        train_losses, valid_losses, accuracies, precisions, recalls, f1s, aucs = [], [], [], [], [], [], []
        for epoch in range(self.config.model.early_stopping_patience * 2):
            train_loss = self.training_loop.train_epoch(train_loader)
            valid_loss, accuracy, precision, recall, f1, auc = self.training_loop.validate_epoch(valid_loader)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            aucs.append(auc)

            self.red_lr.step(valid_loss)
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                logger.info("Early stopping triggered.")
                break
        return train_losses, valid_losses, accuracies, precisions, recalls, f1s, aucs

    def test_epoch(self, test_loader, return_predictions=False):
        return self.training_loop.test_epoch(test_loader, return_predictions)

class Plot:
    @staticmethod
    def plot_losses(train_losses, valid_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(valid_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_classification_metrics_vs_epoch(accuracies, precisions, recalls, f1s, aucs):
        epochs = range(1, len(accuracies) + 1)

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 3, 1)
        plt.plot(epochs, accuracies, label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Epoch')
        plt.legend()

        plt.subplot(2, 3, 2)
        plt.plot(epochs, precisions, label='Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.title('Precision vs. Epoch')
        plt.legend()

        plt.subplot(2, 3, 3)
        plt.plot(epochs, recalls, label='Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.title('Recall vs. Epoch')
        plt.legend()

        plt.subplot(2, 3, 4)
        plt.plot(epochs, f1s, label='F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs. Epoch')
        plt.legend()

        plt.subplot(2, 3, 5)
        plt.plot(epochs, aucs, label='AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('AUC vs. Epoch')
        plt.legend()

        plt.tight_layout()
        plt.show()
