"""The feature transformer: median-impute + scale numerics, one-hot the
categoricals. Wrapped in a ColumnTransformer so it's fit only on training data
and reused everywhere (no leakage through scaling)."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocess import CATEGORICAL


def build_preprocessor(numeric):
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, numeric),
                              ("cat", cat, CATEGORICAL)])
