from collections import OrderedDict
from catalyst.dl import ConfigExperiment
from src.dataset import ContainerDataset
from src.augmentations import test_transforms, train_transforms
import pandas as pd


class Experiment(ConfigExperiment):
    def get_datasets(self,
                     stage: str,
                     path_to_dataframe: str = "./data/processed/train/data.csv",
                     fold: int = 0,
                     **kwargs):

        model_params = self._config['model_params']

        print('-' * 30)
        print('encoder_name:', model_params)
        print('-' * 30)

        datasets = OrderedDict()
        df = pd.read_csv(path_to_dataframe)

        if stage != 'infer':
            datasets["train"] = ContainerDataset(df=df, mode='train', fold=fold, transform=train_transforms)
            datasets["valid"] = ContainerDataset(df=df, mode='val', fold=fold, transform=test_transforms)

        print(datasets)

        return datasets
