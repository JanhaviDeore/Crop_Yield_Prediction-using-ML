import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df, target_col='hg/ha_yield'):
    y = df[target_col]
    X = df.drop(columns=[target_col])


    cat = X.select_dtypes(include=['object']).columns.tolist()
    num = X.select_dtypes(exclude=['object']).columns.tolist()


    preprocessor = ColumnTransformer([('num', StandardScaler(), num),('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat)])


    X_transformed = preprocessor.fit_transform(X)
    return X, y, preprocessor, cat, num