from torch.utils.data import Dataset
import pandas as pd


class AdultDataset(Dataset):
    def __init__(self, path, test=False, normalization=True):
        self.normalization = normalization
        self.df, self.y = self._prepare_data(path, test)
        self.test = test

    def _prepare_data(self, path, test):
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

        col = ['capital_gain', 'hours_per_week', 'education_num', ' Male', ' Separated', ' Tech-support',
               ' Without-pay', ' Widowed', ' Priv-house-serv', ' Puerto-Rico', ' Ecuador', ' Prof-specialty',
               ' France', ' Farming-fishing', ' Not-in-family', ' Other', ' Ireland', ' Black', ' Nicaragua',
               ' Philippines', ' Some-college', ' Hong', ' Prof-school', ' 12th', ' Germany', ' Adm-clerical',
               ' Assoc-acdm', ' Exec-managerial', ' Never-worked', ' Private', ' 10th', ' Doctorate',
               'capital_loss', ' Transport-moving', ' Poland', ' Husband', ' Yugoslavia', ' HS-grad',
               ' Female', ' Haiti', ' Peru', ' Canada', ' White', ' India', ' South', ' Iran', ' Greece',
               ' Sales', ' Honduras', ' Hungary', ' China', ' Machine-op-inspct', ' Own-child', ' 1st-4th',
               ' Divorced', ' El-Salvador', ' Protective-serv', ' Preschool', ' Vietnam', ' Holand-Netherlands',
               ' Assoc-voc', ' 5th-6th', ' Italy', ' Japan', ' Wife', ' Craft-repair', ' Self-emp-inc',
               ' Outlying-US(Guam-USVI-etc)', ' 7th-8th', ' United-States', ' Unmarried', 'age',
               ' Married-AF-spouse', ' Taiwan', ' Trinadad&Tobago', ' Never-married', ' Jamaica',
               ' Other-service', ' Masters', ' Cambodia', ' Married-spouse-absent', ' Dominican-Republic',
               ' ?', ' Asian-Pac-Islander', ' Cuba', ' Portugal', ' England', ' State-gov', ' Armed-Forces',
               ' Married-civ-spouse', ' Amer-Indian-Eskimo', ' Guatemala', ' 11th', ' Columbia',
               ' Other-relative', ' Federal-gov', ' Local-gov', ' 9th', ' Self-emp-not-inc', ' Scotland',
               ' Laos', ' Bachelors', ' Thailand', ' Handlers-cleaners', ' Mexico']
        if test:
            df[' Holand-Netherlands'] = 0

        y = pd.get_dummies(pd.read_csv(path, names=cols, usecols=['income_bracket'])).iloc[1:, 1]
        #print(y.shape)
        #y = df[' Male']
        #print(y.shape)
        df = df[col]
        if self.normalization:
            self.mu = df.mean(0)
            self.std = df.std(0)
            df = (df - df.mean(0))/df.std(0)
        return df, y

    def normalize(self, mu, std):
        self.df = (self.df - mu)/std

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return self.df.iloc[idx, :].to_numpy(), self.y.iloc[idx]



