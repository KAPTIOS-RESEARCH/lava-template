import logging
import os
from abc import ABC, abstractmethod
from uuid import uuid4
from src.utils.config import set_seed, instanciate_module, get_available_device
from src.core.trainer import BaseTrainer
from torch import nn
from src.models.base import BaseLavaModel

class AbstractExperiment(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abstractmethod
    def load_dataloader(self):
        raise NotImplementedError

    @abstractmethod
    def load_trainer(self):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError


class BaseExperiment(AbstractExperiment):

    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        experiment_id = str(uuid4())[:4]
        experiment_name = f"{config['name']}_{experiment_id}"

        # LOGGER INIT

        self.log_dir = os.path.join('./logs', experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)

        set_seed(config['seed'])

        logging.info(
            'Initialization of the experiment - {}'.format(experiment_name))
        self.device = get_available_device()
        logging.info(f'Experiments running on device : {self.device}')
        # CORE INIT
        self.model = self.load_model(config['model'])
        self.dataloader = self.load_dataloader(config['dataloader'])
        self.trainer = self.load_trainer(config['trainer'])

    def load_model(self, model_config) -> BaseLavaModel:
        md_name = model_config['module_name']
        cls_name = model_config['class_name']
        params = model_config['parameters']
        model: BaseLavaModel = instanciate_module(md_name, cls_name, params)
        model.to(self.device)
        model.print_model_size()
        return model

    def load_dataloader(self, dataloader_config):
        md_name = dataloader_config['module_name']
        cls_name = dataloader_config['class_name']
        params = dataloader_config['parameters']
        return instanciate_module(md_name, cls_name, params)

    def load_trainer(self, trainer_config) -> BaseTrainer:
        md_name = trainer_config['module_name']
        cls_name = trainer_config['class_name']
        params = trainer_config['parameters']
        return instanciate_module(
            md_name,
            cls_name,
            {
                "device": self.device,
                "model": self.model,
                "parameters": params,
            }
        )

    def run(self):
        train_dl = self.dataloader.train()
        val_dl = self.dataloader.val()
        self.trainer.fit(train_dl, val_dl, self.log_dir)
