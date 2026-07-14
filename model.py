#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JinPang Yu
# @Software: PyCharm

import pandas as pd
import numpy as np
import time
import statistics
import warnings
import os
import csv
import matplotlib.font_manager as fm
from matplotlib import rcParams
from collections import Counter
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK       # 贝叶斯优化库02
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score
from sklearn.metrics import classification_report, make_scorer, confusion_matrix, mutual_info_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV, ParameterSampler, RandomizedSearchCV
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.datasets import make_classification
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier, DMatrix
import shap
from catboost import CatBoostClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier

'''加载数据并进行预处理'''
df = pd.read_csv(f'1995-2019-dataV1-11-N0F1-tw2m.csv')  # 数据集
df = df.dropna(subset=['类型'])
kuoxian = 'prof_type_tw'
df[kuoxian] = df[kuoxian].replace({1: 1, 2: 2, 3: 3, 5: 3, 4: 4})
df.loc[df[kuoxian].isin([1, 2]), [col for col in
                                  ['H0tw', 'H1tw', 'H2tw', 'refzhdtw', 'warmhdtw', 'SFTtwmin', 'WTtwmax'] if
                                  col in df.columns]] = -999.0
categorical_features = [kuoxian]
numerical_features = [col for col in df.columns if col not in categorical_features and col != '类型']
df[categorical_features] = df[categorical_features].astype("category")
# # df = df.dropna()
df = df.replace(-999.0, np.nan)
df['类型'] = df['类型'].replace({1: 0, 2: 0, 3: 0, 5: 0, 4: 1})
# 按年份划分训练集和验证集
df['datetime'] = pd.to_datetime(df['datetime'])
train_data = df[(df['datetime'].dt.year >= 1995) & (df['datetime'].dt.year <= 2014)]        # 20年训练
valid_data = df[(df['datetime'].dt.year >= 2015) & (df['datetime'].dt.year <= 2019)]
X_train = train_data.drop(columns=['类型', 'datetime', 'station', '经度', '纬度', '海拔高度'])
print(X_train.shape)
y_train = train_data['类型']
X_train_columns = [col for col in X_train.columns]
print(X_train_columns)
X_test = valid_data.drop(columns=['类型', 'datetime', 'station', '经度', '纬度', '海拔高度'])
y_test = valid_data['类型']
print("训练集:", Counter(y_train))
print("测试集:", Counter(y_test))

# TS评分函数
def calculate_ts_scores(results):
    categories = results["Actual"].unique()
    ts_scores = {category: {"NA": 0, "NC": 0, "NB": 0} for category in categories}

    for _, row in results.iterrows():
        actual = row["Actual"]
        predicted = row["Predicted"]
        for category in categories:
            if actual == category and predicted == category:
                ts_scores[category]["NA"] += 1
            elif actual == category and predicted != category:
                ts_scores[category]["NC"] += 1
            elif actual != category and predicted == category:
                ts_scores[category]["NB"] += 1

    score_data = []
    for category, scores in ts_scores.items():
        na_count = scores["NA"]
        nc_count = scores["NC"]
        nb_count = scores["NB"]
        ts_score = na_count / (na_count + nc_count + nb_count) if (na_count + nc_count + nb_count) else 0
        hit_rate = na_count / (na_count + nb_count) if (na_count + nb_count) else 0  # 命中率HIT
        pod_rate = na_count / (na_count + nc_count) if (na_count + nc_count) else 0  # 命中率POD
        far_rate = nb_count / (na_count + nb_count) if (na_count + nb_count) else 0  # 空报率
        mar_rate = nc_count / (na_count + nc_count) if (na_count + nc_count) else 0  # 漏报率
        score_data.append({
            "类别": category,
            "NA": na_count,
            "NB": nb_count,
            "NC": nc_count,
            "TS评分": round(ts_score, 4),
            "命中率 (HIT)": round(hit_rate, 4),
            "命中率 (POD)": round(pod_rate, 4),
            "空报率 (FAR)": round(far_rate, 4),
            "漏报率 (MAR)": round(mar_rate, 4),
        })
    return pd.DataFrame(score_data)

def custom_ts_scorer(y_true, y_pred):
    results = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
    ts_scores_df = calculate_ts_scores(results)
    # 提取类别1的TS评分
    category_1_ts_score = ts_scores_df[ts_scores_df['类别'] == 1]['TS评分'].values[0]
    return category_1_ts_score

f1_class_1_scorer = make_scorer(f1_score, pos_label=1)
TS_class_1_scorer = make_scorer(custom_ts_scorer)

# suanfa = 'lgb'       # rf/xgb/lgb/cab
# if suanfa == 'lgb':
#     parameter_space = {
#         'n_estimators': hp.uniform('n_estimators', 20, 1000),
#         'max_depth': hp.quniform('max_depth', 3, 10, 1),
#         'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
#         'subsample': hp.uniform('subsample', 0.5, 1.0),
#         'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
#         'min_child_samples': hp.quniform('min_child_samples', 10, 100, 1),  # LightGBM 的 min_child_samples
#         'lambda': hp.uniform('lambda', 0.01, 2),
#         'alpha': hp.uniform('alpha', 0, 1),
#         'scale_pos_weight': hp.uniform('scale_pos_weight', 1, 20),  # 不平衡问题
#     }               # LGB参数
#
# # 定义保存调优结果的全局列表
# optimization_history = []
#
# macro_f1_scorer = make_scorer(f1_score, average='macro')
# # 定义目标函数
# def objective(params):
#     '''定义LGB模型'''
#     params['n_estimators'] = int(params['n_estimators'])
#     params['max_depth'] = int(params['max_depth'])
#     params['min_child_samples'] = int(params['min_child_samples'])
#
#     sf = LGBMClassifier(
#         n_estimators=params['n_estimators'],
#         max_depth=params['max_depth'],
#         learning_rate=params['learning_rate'],
#         subsample=params['subsample'],
#         colsample_bytree=params['colsample_bytree'],
#         min_child_samples=params['min_child_samples'],
#         lambda_l2=params['lambda'],
#         lambda_l1=params['alpha'],
#         scale_pos_weight=params['scale_pos_weight'],
#         verbose=-1,
#         random_state=1, n_jobs=-1
#     )
# #
#     # K-Fold交叉验证
#
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
#     f1_scores = []
#     TS_scores = []
#     target_class = 1
#
#     for train_idx, val_idx in skf.split(X_train, y_train):
#         X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
#         y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
#         sf.fit(X_fold_train, y_fold_train)
#
#         y_fold_pred = sf.predict(X_fold_val)
#
#         # fold_f1 = f1_score(y_fold_val, y_fold_pred, labels=[target_class], average=None)  # 针对类别 计算 F1
#         fold_TS = custom_ts_scorer(y_fold_val, y_fold_pred)  # 调用 TS评分函数
#         # f1_scores.append(fold_f1)
#         TS_scores.append(fold_TS)
#
#     # mean_f1 = np.mean(f1_scores)
#     mean_TS = np.mean(TS_scores)
#     sf.fit(X_train, y_train)
#     y_pred = sf.predict(X_test)
#     TEST_TS = custom_ts_scorer(y_test, y_pred)
#     optimization_history.append({**params, "score_TS": mean_TS, "test_TS": TEST_TS})        # ,"score_TS": mean_TS
#
#     return {'loss': -mean_TS, 'status': STATUS_OK}  # Hyperopt最小化目标函数，因此取负值
#
# # 运行贝叶斯优化
# start_time = time.time()            # 记录开始时间
# trials = Trials()  # 用于保存每轮实验结果
# best_params = fmin(
#     fn=objective,
#     space=parameter_space,
#     algo=tpe.suggest,
#     max_evals=100,        # 跑100次
#     trials=trials
# )
#
# print("Best hyperparameters:", best_params)       # 显示最优参数
#
# # 保存每轮调优结果为CSV
# with open(f"{suanfa}_baye_optimization_results.csv", "w", newline="") as f:
#     writer = csv.DictWriter(f, fieldnames=list(optimization_history[0].keys()))
#     writer.writeheader()
#     writer.writerows(optimization_history)
#
# # 保存最优参数为TXT
# with open(f"{suanfa}_baye_best_params.txt", "w") as f:
#     f.write(str(best_params))
# #
# 使用最佳参数创建最终模型
# if suanfa == 'lgb':
#     sf1 = LGBMClassifier(
#         n_estimators=int(best_params['n_estimators']),
#         max_depth=int(best_params['max_depth']),
#         learning_rate=best_params['learning_rate'],
#         subsample=best_params['subsample'],
#         colsample_bytree=best_params['colsample_bytree'],
#         min_child_samples=int(best_params['min_child_samples']),
#         lambda_l2=best_params['lambda'],
#         lambda_l1=best_params['alpha'],
#         scale_pos_weight=best_params['scale_pos_weight'],  # 处理不平衡数据
#         random_state=1, n_jobs=-1
#     )           # LGB

sf1 = LGBMClassifier(n_estimators=646, max_depth=6, learning_rate=0.06270938758947427,
                                  subsample=0.753070242333618, colsample_bytree=0.6491384899484783,
                                  min_child_samples=63, lambda_l2=0.810613353418225,
                                  lambda_l1=0.8757338920254367, scale_pos_weight=2.0396350961089103, random_state=1)

model = sf1.fit(X_train, y_train)
# end_time = time.time()              # 记录结束时间
# optimization_time = end_time - start_time           # 计算模型寻优建立时间
# print(f"Optimization Time: {optimization_time:.2f} seconds")
# with open(f'{suanfa}_1995-2014_bayer.pkl', 'wb') as f:
#     pickle.dump(model, f)     # 保存模型

y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)
score_df_test = calculate_ts_scores(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))
score_df_train = calculate_ts_scores(pd.DataFrame({'Actual': y_train, 'Predicted': y_pred_train}))
print('训练集TS\n', score_df_train)
print('测试集TS\n', score_df_test)
