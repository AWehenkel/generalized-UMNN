import torch
from torch.utils.data import Dataset
import pandas as pd


class AdultDataset(Dataset):
    def __init__(self, path):
        self.df, self.y = self._prepare_data(path)

    def _prepare_data(self, path):
        cols = [
            "age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "gender",
            "capital_gain", "capital_loss", "hours_per_week", "native_country",
            "income_bracket"
        ]
        x_columns = [
            "age", "workclass", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "gender",
            "capital_gain", "capital_loss", "hours_per_week", "native_country"
        ]
        df = pd.read_csv(path, names=cols, usecols=x_columns)
        df = df.iloc[1:, :]
        df["age"] = df["age"].astype(int)

        def one_hot_encode(df, col):
            s = df[col]
            encoded = pd.get_dummies(s)
            df.drop(columns=[col], inplace=True)
            df[encoded.columns] = encoded

        one_hot_encode(df, "workclass")
        one_hot_encode(df, "marital_status")
        one_hot_encode(df, "occupation")
        one_hot_encode(df, "relationship")
        one_hot_encode(df, "race")
        one_hot_encode(df, "gender")
        one_hot_encode(df, "native_country")
        one_hot_encode(df, "education")
        print(df.head())
        df = (df - df.mean())/df.std()
        col = df.columns
        mononotonic_cols = ["capital_gain", "hours_per_week", "education_num", " Male"]
        col = mononotonic_cols + list(set(col) - set(mononotonic_cols))
        y = pd.get_dummies(pd.read_csv(path, names=cols, usecols=['income_bracket'])).iloc[:, 1]
        return df[col], y

    def __len__(self):
        return 20#self.df.shape[0]

    def __getitem__(self, idx):
        return self.df.iloc[idx, :].to_numpy(), self.y[idx]



