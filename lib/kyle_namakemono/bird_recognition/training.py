from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import xgboost as xgb
import pickle
from . import metrics
from . import datasets
from . import feature_extraction
from catboost import CatBoostClassifier
from catboost import Pool
from imblearn.over_sampling import RandomOverSampler
import lightgbm as lgb
import os
import random

import numpy as np
import tensorflow as tf
import torch


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    

def train(
    candidate_df:pd.DataFrame,
    df:pd.DataFrame,
    candidate_df_soundscapes:pd.DataFrame,
    df_soundscapes:pd.DataFrame,
    num_kfolds:int,
    num_candidates:int,
    weight_rate:float=1.0,
    xgb_params:dict=None,
    verbose:bool=False,
    mode=None,
    sampling_strategy:float=1.0,
    random_state:int=777,
):
    seed_everything(random_state)
    
    if xgb_params is None:
        xgb_params = {
            "objective": "binary:logistic",
            "tree_method": 'gpu_hist'
        }
    feature_names = feature_extraction.get_feature_names()
    if verbose:
        print("features", feature_names)
        
        
    # short audio の  k fold
    groups = candidate_df["audio_id"]
    kf = StratifiedGroupKFold(n_splits=num_kfolds) # lgbm_rankを使う場合は、groupごとにくっつけたデータを使う必要があるのでシャッフルしない
    for kfold_index, (_, valid_index) in enumerate(kf.split(candidate_df[feature_names].values, candidate_df["target"].values, groups)):
        candidate_df.loc[valid_index, "fold"] = kfold_index
                        
    X = candidate_df[feature_names].values
    y = candidate_df["target"].values
    oofa = np.zeros(len(candidate_df_soundscapes), dtype=np.float32)
    
    for kfold_index in range(num_kfolds):
        print(f"fold {kfold_index}")
        train_index = candidate_df[candidate_df["fold"] != kfold_index].index
        valid_index = candidate_df[candidate_df["fold"] == kfold_index].index
        X_train, y_train = X[train_index], y[train_index]
        #X_valid, y_valid = X[valid_index], y[valid_index]
        X_valid, y_valid = candidate_df_soundscapes[feature_names].values, candidate_df_soundscapes["target"].values
        '''
        #----------------------------------------------------------------------
        if mode=='lgbm' or mode=='cat':
            if sampling_strategy is not None:
                # 正例を10％まであげる
                ros = RandomOverSampler(
                    sampling_strategy=sampling_strategy,
                    random_state=random_state
                )
                # 学習用データに反映
                X_train, y_train = ros.fit_resample(X_train, y_train)
                print("Resampled. positive ratio: %.4f" % np.mean(y_train))
        '''
        if mode=='lgbm':
            dtrain = lgb.Dataset(X_train, label=y_train)
            dvalid = lgb.Dataset(X_valid, label=y_valid)
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'device':'gpu',
            }
            model = lgb.train(
                params,
                dtrain,
                valid_sets=dvalid,
                num_boost_round=200,
                early_stopping_rounds=20,
                verbose_eval=20,
            )
            oofa += model.predict(X_valid.astype(np.float32))/num_kfolds
            pickle.dump(model, open(f"lgbm_{kfold_index}.pkl", "wb"))

        elif mode=='cat':
            train_pool = Pool(X_train, label=y_train)
            valid_pool = Pool(X_valid, label=y_valid)
            model = CatBoostClassifier(
                loss_function='Logloss',
                task_type='GPU',
                random_seed=random_state,    
            )
            model.fit(train_pool, eval_set=[valid_pool], use_best_model=True, verbose=100)
            oofa+= model.predict_proba(valid_pool)[:,1]/num_kfolds
            pickle.dump(model, open(f"cat_{kfold_index}.pkl", "wb"))

        elif mode=='xgb':
            # 正例の重みを weight_rate, 負例を1にする
            sample_weight = np.ones(y_train.shape)
            sample_weight[y_train==1] = weight_rate
            sample_weight_val = np.ones(y_valid.shape)
            sample_weight_val[y_valid==1] = weight_rate
            sample_weight_eval_set = [sample_weight, sample_weight_val]
            clf = xgb.XGBClassifier(**xgb_params)
            clf.fit(
                X_train, y_train,
                eval_set = [
                    (X_train, y_train),
                    (X_valid, y_valid)
                ],
                eval_metric             = "logloss",
                verbose                 = 20, # None,
                early_stopping_rounds   = 100,
                sample_weight           = sample_weight,
                sample_weight_eval_set  = sample_weight_eval_set,
            )
            pickle.dump(clf, open(f"xgb_{kfold_index}.pkl", "wb"))
            oofa+= clf.predict_proba(X_valid)[:,1]/num_kfolds
        #----------------------------------------------------------------------

        if mode=='lgbm_rank':
            lgbm_params = {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'ndcg_eval_at': [3, 5],
                'boosting_type': 'gbdt',
            }
            lgtrain = lgb.Dataset(X_train, y_train,  group=[num_candidates for i in range(len(y_train)//num_candidates)])
            lgvalid = lgb.Dataset(X_valid, y_valid,  group=[num_candidates for i in range(len(y_valid)//num_candidates)])
            lgbrank = lgb.train(
                lgbm_params,
                lgtrain,
                num_boost_round=10,
                valid_sets=[lgtrain, lgvalid],
                valid_names=['train','valid'],
                early_stopping_rounds=2,
                verbose_eval=1
            )
            oofa += lgbrank.predict(X_valid, num_iteration=lgbrank.best_iteration)/num_kfolds
            pickle.dump(lgbrank, open(f"lgbm_rank_{kfold_index}.pkl", "wb"))
            
    # 0-1 に標準化
    if mode=='lgbm_rank':
        oofa = 1/(1 + np.exp(-oofa))
        
    def f(th):
        _df = candidate_df_soundscapes[(oofa > th)]
        if len(_df) == 0:
            return 0
        _gdf = _df.groupby(
            ["audio_id", "seconds"],
            as_index=False
        )["label"].apply(lambda _: " ".join(_))
        df2 = pd.merge(
            df_soundscapes[["audio_id", "seconds", "birds"]],
            _gdf,
            how="left",
            on=["audio_id", "seconds"]
        )
        df2.loc[df2["label"].isnull(), "label"] = "nocall"
        return df2.apply(
            lambda _: metrics.get_metrics(_["birds"], _["label"])["f1"],
            axis=1
        ).mean()


    print("-"*30)
    print(f"#sound_scapes (len:{len(candidate_df_soundscapes)}) でのスコア")
    lb, ub = 0, 1
    for k in range(30):
        th1 = (2*lb + ub) / 3
        th2 = (lb + 2*ub) / 3
        if f(th1) < f(th2):
            lb = th1
        else:
            ub = th2
    th = (lb + ub) / 2
    print("best th: %.4f" % th)
    print("best F1: %.4f" % f(th))
    if verbose:
        y_soundscapes =  candidate_df_soundscapes["target"].values
        oof = (oofa > th).astype(int)
        print("[details] Call or No call classirication")
        print("binary F1: %.4f" % f1_score(y_soundscapes, oof))
        print("gt positive ratio: %.4f" % np.mean(y_soundscapes))
        print("oof positive ratio: %.4f" % np.mean(oof))
        print("Accuracy: %.4f" % accuracy_score(y_soundscapes, oof))
        print("Recall: %.4f" % recall_score(y_soundscapes, oof))
        print("Precision: %.4f" % precision_score(y_soundscapes, oof))
    print("-"*30)
    print()
        