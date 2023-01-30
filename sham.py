import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import utils

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchvision import transforms

from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from nam.config import defaults
from nam.data import NAMDataset
from nam.models import NAM
from nam.models import get_num_units
from nam.trainer import LitNAM
from nam.utils import plot_mean_feature_importance
from nam.utils import plot_nams

import warnings

warnings.filterwarnings("ignore")


class SHAM:
    def __init__(
            self,
            datapath: str,
            abs_datapath: str,
            shap_datapath: str,
            abs_shap_datapath: str,
            dataset_name: str,
            mode: str,
            regression=True,
            hyper_tuning=True,
            config=None,
            ) -> None:

        self.Litmodel = None
        self.Trainer = None

        if config is not None:
            self.config = config
        else:
            self.config = defaults()

        self.datapath = datapath
        self.abs_datapath = abs_datapath
        self.shap_datapath = shap_datapath
        self.abs_shap_datapath = abs_shap_datapath
        self.dataset_name = dataset_name
        self.mode = mode
        self.regression = regression
        self.hyper_tuning = hyper_tuning

    def load_datasets(self,
                      return_dataloaders=True,
                      return_nam_dataset=True):

        if self.hyper_tuning:
            self.df = pd.read_csv(self.abs_datapath)
            self.shap_df = pd.read_csv(self.abs_shap_datapath)
        else:
            self.df = pd.read_csv(self.datapath)
            self.shap_df = pd.read_csv(self.shap_datapath)

        self.shap_df = self.shap_df.add_suffix("_shap")

        self.df = self.df.drop(["Unnamed: 0", "ID", "Date"], axis=1)
        self.shap_df = self.shap_df.drop(["Unnamed: 0_shap", "ID_shap", "Date_shap"], axis=1)
        self.merged_df = pd.merge(self.df, self.shap_df, how="inner", left_index=True, right_index=True)

        feat_cols = self.df.columns[:-1]
        target_cols = self.df.columns[-1]
        shap_cols = self.shap_df.columns[:-1]
        shap_target_cols = self.shap_df.columns[-1]

        baseline_mean = self.df.mean()
        baseline_std = self.df.std()

        shap_mean = self.shap_df.mean()
        shap_std = self.shap_df.std()

        weighted_mean = self.merged_df.mean()
        weighted_std = self.merged_df.std()

        if self.mode == 'Baseline':

            dataset = NAMDataset(
                config=self.config,
                data_path=self.df,
                features_columns=feat_cols,
                targets_column=target_cols,
                transform=transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(baseline_mean,
                                                                   baseline_std)])
            )

            train, val = dataset.train_dataloaders()
            test = dataset.test_dataloaders()

        elif self.mode == 'SHAP':

            dataset = NAMDataset(
                config=self.config,
                data_path=self.shap_df,
                features_columns=shap_cols,
                targets_column=shap_target_cols,
                transform=transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(shap_mean,
                                                                   shap_std)])
            )
            train, val = dataset.train_dataloaders()
            test = dataset.test_dataloaders()

        elif self.mode == 'Weighted':

            dataset = NAMDataset(
                config=self.config,
                data_path=self.merged_df,
                features_columns=feat_cols,
                targets_column=target_cols,
                weights_column=shap_cols,
                transform=transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(weighted_mean,
                                                                   weighted_std)])
            )

            train, val = dataset.train_dataloaders()
            test = dataset.test_dataloaders()
        else:
            raise Exception('Mode not Supported')

        if return_nam_dataset:
            return dataset

        if return_dataloaders:
            return train, val, test

    def train_nam(self, cfg):

        self.config.update(**cfg)

        if self.regression:
            metrics = {"loss": "val_loss"}
        else:
            metrics = {'accuracy': 'val_accuracy'}

        callbacks = TuneReportCallback(metrics, on="validation_end")

        if self.regression:
            es = EarlyStopping(monitor='val_loss',
                               min_delta=1e-4,
                               patience=5,
                               verbose=False,
                               mode='min')
        else:
            es = EarlyStopping(monitor='val_accuracy',
                               min_delta=1e-4,
                               patience=5,
                               verbose=False,
                               mode='max')

        dataset = self.load_datasets(return_dataloaders=False,
                                     return_nam_dataset=True)

        train, val, _ = self.load_datasets(return_dataloaders=True,
                                           return_nam_dataset=False)

        model = NAM(
            config=self.config,
            name="NAM_SIM",
            num_inputs=len(dataset[0][0]),
            num_units=get_num_units(self.config, dataset.features),
        )

        litmodel = LitNAM(self.config, model)

        trainer = pl.Trainer(gpus=0, callbacks=[es, callbacks], max_epochs=10)

        trainer.fit(litmodel, train, val)

    def run(self):

        if self.hyper_tuning:
            trainable = tune.with_parameters(self.train_nam)

            print('Searching...')
            s = time.time()

            analysis = tune.run(trainable,
                                resources_per_trial={'gpu': 0, 'cpu': 8},
                                num_samples=3,
                                metric="loss",
                                mode="min",
                                config={
                                    "lr": tune.loguniform(1e-4, 1e-1),
                                    "dropout": tune.loguniform(0.01, 1.0),
                                    "batch_size": tune.choice([128, 512, 1024]),
                                    "hidden_sizes": tune.choice([[], [32], [64, 32]])
                                })
            e = time.time()
            print(f"Finished searching with {e - s}")

            lr = list(analysis.best_config.values())[0]
            dropout = list(analysis.best_config.values())[1]
            batch_size = list(analysis.best_config.values())[2]
            hidden_size = list(analysis.best_config.values())[3]

            print(f"Best config {analysis.best_config}")

            self.config.lr = lr
            self.config.dropout = dropout
            self.config.batch_size = batch_size
            self.config.hidden_size = hidden_size

        dataset = self.load_datasets(return_dataloaders=False,
                                     return_nam_dataset=True)

        train, val, _ = self.load_datasets(return_dataloaders=True,
                                              return_nam_dataset=False)

        Model = NAM(config=self.config,
                    name="NAM_Example",
                    num_inputs=len(dataset[0][0]),
                    num_units=get_num_units(self.config, dataset.features))

        self.Litmodel = LitNAM(self.config, Model)

        logger = TensorBoardLogger(self.config.logdir, name=f'{Model.name}')

        self.Trainer = pl.Trainer(max_epochs=self.config.num_epochs)

        print('Training...')
        start = time.time()

        self.Trainer.fit(self.Litmodel,
                         train_dataloader=train,
                         val_dataloaders=val)

        end = time.time()
        print('-----------------------------------')
        print(f"Finish training with {end - start}s")

    def test_models(self):

        _, _, test = self.load_datasets(return_dataloaders=True,
                                        return_nam_dataset=False)
        self.Trainer.test(self.Litmodel, test)

    def plot_predictions(self, error_estimators=True):

        dataset = self.load_datasets(return_dataloaders=False,
                                     return_nam_dataset=True)

        _, _, test = self.load_datasets(return_dataloaders=True,
                                        return_nam_dataset=False)

        prediction = self.Trainer.predict(self.Litmodel,
                                          dataloaders=test)
        
        true = np.array(self.df[dataset.targets_column])

        pred = np.array(prediction[-1]).reshape(-1, 1)
        idx = pred.shape[0]
        true = np.array(true[-idx:]).reshape(-1, 1)
        train = np.array(self.df[dataset.targets_column][idx:])

        utils.plot_results(range(true.shape[0]), true, pred)

        if error_estimators:
            utils.error_estimator(true, pred, train)
