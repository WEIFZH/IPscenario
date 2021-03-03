import os
import numpy as np
import pandas as pd
import torch
import random

from sklearn.preprocessing import QuantileTransformer

class Dataset:

    def __init__(self, dataset, random_state, data_path='./data', normalize=False,
                 quantile_transform=False, output_distribution='normal', quantile_noise=0, **kwargs):
        """
        Dataset is a dataclass that contains all training and evaluation data required for an experiment
        :param dataset: a pre-defined dataset name (see DATSETS) or a custom dataset
            Your dataset should be at (or will be downloaded into) {data_path}/{dataset}
        :param random_state: global random seed for an experiment
        :param data_path: a shared data folder path where the dataset is stored (or will be downloaded into)
        :param normalize: standardize features by removing the mean and scaling to unit variance
        :param quantile_transform: transforms the features to follow a normal distribution.
        :param output_distribution: if quantile_transform == True, data is projected onto this distribution
            See the same param of sklearn QuantileTransformer
        :param quantile_noise: if specified, fits QuantileTransformer on data with added gaussian noise
            with std = :quantile_noise: * data.std ; this will cause discrete values to be more separable
            Please not that this transformation does NOT apply gaussian noise to the resulting data,
            the noise is only applied for QuantileTransformer
        :param kwargs: depending on the dataset, you may select train size, test size or other params
            If dataset is not in DATASETS, provide six keys: X_train, y_train, X_valid, y_valid, X_test and y_test
        """
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        random.seed(random_state)

        data_dict = fetch_scenario(dataset)


        self.data_path = data_path
        self.dataset = dataset

        self.X_train = data_dict['X_train']
        self.y_train = data_dict['y_train']
        self.X_valid = data_dict['X_valid']
        self.y_valid = data_dict['y_valid']
        self.X_test = data_dict['X_test']
        self.y_test = data_dict['y_test']

        if all(query in data_dict.keys() for query in ('query_train', 'query_valid', 'query_test')):
            self.query_train = data_dict['query_train']
            self.query_valid = data_dict['query_valid']
            self.query_test = data_dict['query_test']

        if normalize:
            mean = np.mean(self.X_train, axis=0)
            std = np.std(self.X_train, axis=0)
            self.X_train = (self.X_train - mean) / std
            self.X_valid = (self.X_valid - mean) / std
            self.X_test = (self.X_test - mean) / std

        if quantile_transform:
            quantile_train = np.copy(self.X_train)
            if quantile_noise:
                stds = np.std(quantile_train, axis=0, keepdims=True)
                noise_std = quantile_noise / np.maximum(stds, quantile_noise)
                quantile_train += noise_std * np.random.randn(*quantile_train.shape)

            qt = QuantileTransformer(random_state=random_state, output_distribution=output_distribution).fit(quantile_train)
            self.X_train = qt.transform(self.X_train)
            self.X_valid = qt.transform(self.X_valid)
            self.X_test = qt.transform(self.X_test)




def fetch_scenario(dataset, train_size=0.6, valid_size=0.2, test_size=0.2):
    df = pd.read_csv("../data/" + dataset + "_cate2id.csv")
    df = df.sample(frac=1)
    df.fillna(value=0, inplace=True) # 填充0
    block_nums = df.shape[0]
    train_size = int(block_nums*train_size)
    valid_size = int(block_nums*valid_size)
    test_size = int(block_nums*test_size)

    X = df.loc[:, df.columns != 'scene']
    y = pd.Categorical(df.scene).codes

    X_train = X[:train_size].values
    X_valid = X[train_size:train_size+valid_size].values
    X_test = X[train_size+valid_size:].values

    y_train = y[:train_size]
    y_valid = y[train_size:train_size+valid_size]
    y_test = y[train_size+valid_size:]

    return dict(
        X_train=X_train, y_train=y_train,
        X_valid=X_valid, y_valid=y_valid,
        X_test=X_test, y_test=y_test,
    )

