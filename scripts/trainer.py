import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from evaluation import evaluation


class Trainer:
    def __init__(self, model, train_loader, val_loader, args, patience=50):
        self.args = args
        self.epochs = args.epochs
        self.lr = args.lr
        self.model = model

        self.loss_fn = F.cross_entropy
        self.optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        self.train_loader, self.val_loader = train_loader, val_loader
        # self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100, 200, 300, 400], gamma=0.1)
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=5,
                                                                 min_lr=1e-7)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.best_loss = np.inf
        self.patience = patience
        self.tries = 0

    def step(self, x, y):
        self.optimizer.zero_grad(set_to_none=True)  # resets the gradients to 0
        preds = self.model(x)
        loss = self.loss_fn(preds, y)
        loss.backward()
        self.optimizer.step()
        return loss

    def pre_evaluation(self):
        print('\nModel pre-evaluation...')
        results = evaluation(self.model, self.val_loader, self.loss_fn)
        return results['average_loss']

    def print_summary(self):
        print(f'\nTraining hyperparameters'
              f'\nDevice: {self.device}'
              f'\nLearning rate: {self.args.lr}'
              f'\nDropout: {self.args.dropout}'
              f'\nEpochs: {self.epochs}'
              f'\nPatience: {self.patience}'
              f'\nBatch size: {self.args.batch_size}'
              f'\n')
        pass

    def save_model(self, val_loss):
        if not os.path.exists(os.path.join('trained_models', self.args.model)):
            os.mkdir(os.path.join('trained_models', self.args.model))
        if val_loss < self.best_loss:
            self.tries = 0
            self.best_loss = val_loss
            torch.save(self.model.state_dict(), os.path.join('trained_models',
                                                             self.args.model,
                                                             f'{self.args.model}_{self.args.date}.pth'))
        else:
            self.tries += 1

    def run(self):

        self.print_summary()

        self.model.to(self.device)

        self.best_loss = self.pre_evaluation()
        print(f'Pre-evaluation loss: {self.best_loss}')

        running_loss = 0.0

        for epoch in range(self.epochs):
            self.model.train()
            print(f'\nCurrent Learning Rate: {self.optimizer.param_groups[0]["lr"]}')
            for i, (x, y) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                x, y = x.to(self.device), y.to(self.device)
                loss = self.step(x, y)
                running_loss += loss.detach().item()

            val_results = evaluation(self.model, self.val_loader, self.loss_fn)
            train_loss = running_loss / len(self.train_loader)
            running_loss = 0.0
            print(
                f'Epoch {epoch + 1}'
                f'\nAccuracy: {val_results["accuracy"]}'
                f'\n\nTraining loss: {train_loss}'
                f'\nValidation loss: {val_results["average_loss"]}\n')

            if self.args.lr_scheduler:
                # self.lr_scheduler.step()
                self.lr_scheduler.step(val_results["average_loss"])

            self.save_model(val_results["average_loss"])

            if self.tries > self.patience:
                break
