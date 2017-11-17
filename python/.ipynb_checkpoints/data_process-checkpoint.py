import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from typing import Iterable
import hmmlearn.hmm as hmm
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import KFold
import sklearn as sk
from functools impor
from multiprocessing import Pool

num_partitions = 10 #number of partitions to split dataframe
num_cores = 8 #number of cores on your machine

def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df
t reduce

def process_discrete_cols(df: DataFrame, cols: Iterable[str]):
    f:r c in cols:
        vals = d
        def single_process(frame):f            ue()
        f                frame
            df[c + '=frame+ str(v)] = (df[c] ==
        parallelize_dataframe(df, single_process) v).areturn dfce=True, axis=1)

def process_set_cols(df: DataFrame, cols: Iterable[str], parser):
    for c in cols:
        vals = set(reduce(lambda x, y: x+y, df[c].apply(lambda x: parser(x)).values.
        def single_process(frame):f            .tolist()))
                     frame          df[c frame'=' + str(item)] = (df[c].apply(lambda x: it        parallelize_dataframe(df, single_process), inplace=True, axis=1)

def clean_currency(df: DataFrame, cols: Iterable[str]):
    for c in cols:
        df[c] = df[c].apply(lambda x: float(x.strip().replace("$", "").replace(",", "")))

def run_kfold_on_model(df: DataFrame, exclude_cols: Iterable[str], target_col: str, model, lossfun):
    X, Y = make_dataset(df, exclude_cols, target_col)
    kf = KFold(n_folds=10, shuffle=True)
    losses = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        model.fit(X_train, Y_train)
        pred = model.predict(Y_test)
        losses.append(lossfun(Y_test, pred))
    return losses

def make_dataset(df: DataFrame, exclude_cols: Iterable[str], target_col: str):
    X = df.drop(columns=exclude_cols + [target_col]).values
    Y = df[target_col].values
    return X, Y

def gaussianHmmBase(df: DataFrame, cols: Iterable[str], n_components=2, covar_type='full', append_cols=False, key_obs=None):
    n_features = len(cols)
    model = hmm.GaussianHMM(n_components=n_components, covariance_type=covar_type)
    data = df[cols].values.reshape((-1, 1))
    model.fit(data)

    df2 = None
    if append_cols:
        df2 = df
    else:
        df2 = df.copy()
    df2['state'] = model.predict(data)
    proba = model.predict_proba(data)
    print('============ Model Stats ================')
    for i in range(n_components):
        print('Means: ' + model.means_[i, :])
        print('Covar: ' + model.covars_[i, :, :])
        df2['proba_' + i] = proba[:,i]
    print('Transition Matrix: ')
    print(model.transmat_)
    if key_obs is not None:
        plt.figure()
        plt.title('HMM inferred states')
        ax = df2['state'].plot()
        ax.yticks(range(n_components))
        ax.legend(loc = 2)
        ax.set_ylabel('state')
        ax = df2[key_obs].plot(secondary_y = True)
        ax.legend(loc = 1)
        plt.savefig(key_obs + 'hmm.pdf')
    return df2