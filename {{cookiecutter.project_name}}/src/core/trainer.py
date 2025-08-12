import logging, time, torch, os, importlib
import lava.lib.dl.slayer as slayer
from torch import nn
from torch.optim import Adam
from src.utils.config import instanciate_module


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
                                            parameters['loss']['parameters'])

        self.classifier = getattr(importlib.import_module(parameters['classifier']['module_name']), parameters['classifier']['class_name'])
        

    def fit(self, train_dl, test_dl, log_dir: str):
        start_time = time.time()
        num_epochs = self.parameters['num_epochs']
        stats = slayer.utils.LearningStats()
        assistant = slayer.utils.Assistant(
            self.model, self.criterion, self.optimizer, stats,
            classifier=self.classifier.predict
        )
        for epoch in range(num_epochs):
            
            logging.info(f"Epoch {epoch + 1} / {num_epochs}")
        
            for i, (input, label) in enumerate(train_dl):
                assistant.train(input, label)
                stats.print(epoch, iter=i, dataloader=train_dl)

            for i, (input, label) in enumerate(test_dl):
                assistant.test(input, label)
                stats.print(epoch, iter=i, dataloader=test_dl)
            
            if stats.testing.best_accuracy:
                model_object = {
                    'weights': self.model.state_dict(),
                    'max_acc': stats.testing.best_accuracy,
                }
                torch.save(model_object, os.path.join(log_dir, 'best_model.pth'))

            stats.update()
            stats.save(log_dir + '/')
            stats.plot(path=log_dir + '/')
        
        end_time = time.time()
        logging.info(
            f"Training completed in {end_time - start_time:.2f} seconds.")