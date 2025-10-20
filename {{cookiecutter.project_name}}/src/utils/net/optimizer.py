import optuna, logging, yaml
from abc import ABC, abstractmethod
from functools import partial
from src.utils.config import get_available_device

class BaseOptimizer(ABC):
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.device = get_available_device()
        
    @abstractmethod
    def objective(self, trial: optuna.Trial, *args, **kwargs):
        raise NotImplementedError

    def log_results(self, study: optuna.Study):
        best_trial = study.best_trial
        logging.info("Best Trial:")
        for k, v in best_trial.user_attrs.items():
            logging.info(f"  - {k}: {v:.4f}")
        logging.info("Best parameters:")
        for k, v in best_trial.params.items():
            logging.info(f"  - {k}: {v:.4f}")

    def save_results(self, study: optuna.Study, save_path: str):
        best_trial = study.best_trial
        result_dict = {
            "params": {k: round(v, 3) for k, v in best_trial.params.items()},
            "user_attrs": {k: round(v, 3) if isinstance(v, float) else v
                           for k, v in best_trial.user_attrs.items()}
        }
        with open(save_path, "w") as f:
            yaml.safe_dump(result_dict, f)

    def run(self, objective_fn=None, trials: int = 50, save_path: str = "best_params.yaml", *args, **kwargs):
        study = optuna.create_study(direction='minimize')
        objective = objective_fn or partial(self.objective, *args, **kwargs)
        study.optimize(objective, n_trials=trials, show_progress_bar=True)
        self.log_results(study)
        self.save_results(study, save_path)
        return study
