import logging, time, torch, os, wandb
from torch import nn
from torch.optim import Adam
from src.utils.config import instanciate_module
from src.utils.learning.metrics import spike_rate_accuracy
from tqdm import tqdm

class BaseTrainer(object):

    def __init__(self, model: nn.Module, parameters: dict, device: str):
        self.model = model
        self.parameters = parameters
        self.device = device

        # OPTIMIZER
        self.optimizer = Adam(
            self.model.parameters(),
            lr=parameters['lr'],
            weight_decay=parameters['weight_decay']
        )

        # LOSS FUNCTION
        self.criterion = instanciate_module(parameters['loss']['module_name'],
                                            parameters['loss']['class_name'],
                                            parameters['loss']['parameters']).to(self.device)
        
    def _run_epoch(self, loader, phase="train", epoch=None, num_epochs=None):
        is_train = phase == "train"
        self.model.train(mode=is_train)
        total_loss, total_acc = 0.0, 0.0
        desc = f"Epoch [{epoch}/{num_epochs}] - {phase.capitalize()}" if epoch is not None else f"Running {phase} phase"

        with tqdm(loader, leave=True, desc=desc) as pbar:
            for i, (data, targets) in enumerate(loader, start=1):
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                acc = spike_rate_accuracy(outputs, targets)
                total_acc += acc
                total_loss += loss.item()

                avg_loss = total_loss / i
                avg_acc = total_acc / i
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Acc': f'{avg_acc:.4f}',
                    **{f"{k}": f"{v:.4f}" for k, v in self.model.avg_spike_stats.items()}
                })
                pbar.update(1)

        avg_loss = total_loss / len(loader)
        avg_acc = total_acc / len(loader)
        return avg_loss, {'Accuracy': avg_acc}


    def train(self, train_loader, epoch=None, num_epochs=None):
        return self._run_epoch(train_loader, phase="train", epoch=epoch, num_epochs=num_epochs)

    def test(self, val_loader, epoch=None, num_epochs=None):
        return self._run_epoch(val_loader, phase="validation", epoch=epoch, num_epochs=num_epochs)

    def fit(self, train_dl, test_dl, log_dir: str):
        start_time = time.time()
        num_epochs = self.parameters['num_epochs']
        best_loss = None
        for epoch in range(num_epochs):
            train_loss, train_metrics = self.train(train_dl, epoch, num_epochs)
            test_loss, test_metrics = self.test(test_dl, epoch, num_epochs)

            if self.parameters['track']:
                log_data = {
                    f"Train/{self.parameters['loss']['class_name']}": train_loss,
                    f"Test/{self.parameters['loss']['class_name']}": test_loss,
                    "_step_": epoch
                }
                if train_metrics:
                    for metric_name, value in train_metrics.items():
                        log_data[f"Train/{metric_name}"] = value
                if test_metrics:
                    for metric_name, value in test_metrics.items():
                        log_data[f"Test/{metric_name}"] = value

                wandb.log(log_data)
          
            if best_loss is None or test_loss < best_loss:
                best_loss = test_loss
                model_object = {
                    'weights': self.model.state_dict(),
                    'min_loss': best_loss,
                    'last_epoch': epoch
                }
                torch.save(model_object, os.path.join(log_dir, 'best_model.pth'))

        end_time = time.time()
        logging.info(f"\nTraining completed in {end_time - start_time:.2f} seconds.")

        if self.parameters['track']:
            wandb.finish()
