
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_tabular(df, target_col):
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    num_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(exclude=np.number).columns

    transformer = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    X = transformer.fit_transform(X)
    X = KNNImputer().fit_transform(X)

    return X, y, transformer
