from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def _clip(X):
    """Limite les valeurs standardisees a [-3, 3]."""
    return X.clip(-3, 3)


def build_numeric_preprocess():
    """Imputation mediane, standardisation et clipping."""
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clip", FunctionTransformer(_clip)),
    ])
