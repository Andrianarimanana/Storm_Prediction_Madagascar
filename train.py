#!/usr/bin/env python
# coding: utf-8

# #### Project Storm prediction in North of Madagascar
# The porpuse of this project is a machine learning focused on forcasting thunderstorms in northern Madagascar, particularly around Nosy Be. The project aims to provide accurate short-term predictions (0–6 hours) to mitigate risks, protect lives, and support emergency responses in this vulnerable region.

# #### Data importation
import sys

import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import pyplot as plt

from IPython import get_ipython

# Initialiser IPython si nécessaire
ipython = get_ipython()
if ipython is None:
    from IPython.terminal.embed import InteractiveShellEmbed
    ipython = InteractiveShellEmbed()

train_df = pd.read_csv('./Data/train.csv')
test_df = pd.read_csv('./Data/test.csv')


# #### Data exploration

train_df.head(3)
#train_df.describe()
print("Data exploration")

# #### Data Preparation and Features importance
# 
def create_time_features(df):
    # Convert time columns to datetime
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
    # Extract time features
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute']/24)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute']/24)
    return df

def create_spatial_features(df):
    # Avoid division by zero and handle size=0
    df['intensity_density'] = df['intensity'] / (df['size'].replace(1, np.nan))
    df['intensity_density'] = df['intensity_density'].fillna(0)
    df['storm_proximity'] = 1 / (df['distance'] + 1)
    return df

def create_storm_features(df):
    # Nosy Be Specific Cyclone Season (November to April)
    df['is_peak_cyclone_season'] = df['month'].apply(lambda x: 1 if x in [1, 2, 3] else 0)
    df['is_cyclone_season'] = df['month'].apply(lambda x: 1 if x in [11, 12, 1, 2, 3, 4] else 0)

    # Assign weights to months based on historical cyclone data
    cyclone_weights = {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.4, 11: 0.6, 12: 0.7}
    df['cyclone_season_weight'] = df['month'].map(cyclone_weights).fillna(0)

    # Define day as 6 AM to 6 PM
    df['is_daytime'] = df['hour'].apply(lambda x: 1 if 6 <= x < 18 else 0)

    df['cyclone_daytime_interaction'] = df['is_cyclone_season'] * df['is_daytime']
    df['peak_cyclone_daytime_interaction'] = df['is_peak_cyclone_season'] * df['is_daytime']
    
    return df

def add_lag_features(df, lag_features, intervals):
    df = df.sort_values('datetime').reset_index(drop=True)
    for feat in lag_features:
        for lag_min, lag_steps in intervals.items():
            lag_col = f"{feat}_{lag_min}"
            df[lag_col] = df[feat].shift(lag_steps)
            df[lag_col] = df[lag_col].fillna(0)
    return df

def add_size_features(df):
    df['size_change_30'] = df['size'] - df['size_30']

    return df

def latlon_to_xy(df, lat_ref = -13.3 , lon_ref = 48.3 ):
    R = 6371.0  # Earth radius in kilometers
    rad = np.pi/180.0
    
    delta_lat = (df['lat'] - lat_ref) * rad
    delta_lon = (df['lon'] - lon_ref) * rad
    
    df['distance_y'] = delta_lat * R
    df['distance_x'] = delta_lon * R * np.cos(lat_ref * rad)

    df['radial_distance'] = np.sqrt(df['distance_x']**2 + df['distance_y']**2)
    df['bearing'] = np.arctan2(df['distance_y'], df['distance_x'])
    df['intensity_distance_interaction'] = df['radial_distance'] * df['intensity']
    return df

print("------------- Apply feature engineering --------")
# Apply feature engineering
train_df = create_time_features(train_df)
test_df = create_time_features(test_df)

train_df = create_spatial_features(train_df)
test_df = create_spatial_features(test_df)

train_df = create_storm_features(train_df)
test_df = create_storm_features(test_df)

train_df = latlon_to_xy(train_df)
test_df = latlon_to_xy(test_df)

# Define lag features and intervals
lag_features = ['intensity', 'size', 'distance', 'intensity_density', 'minute_sin', 'minute_cos', 'bearing']
lag_intervals = {30: 2, 60: 4}  # 30min -> 2 steps, 60min -> 4 steps

train_df = add_lag_features(train_df, lag_features, lag_intervals)
test_df = add_lag_features(test_df, lag_features, lag_intervals)

train_df = add_size_features(train_df)
test_df = add_size_features(test_df)

#train_df.head(3)

train_df.columns

#train_df.describe()

print("------------- Prepare Training data --------")
# #### Prepare Training data

# Prepare Training data
feature_cols = [
    'hour_sin', 'hour_cos', 
    'distance_x', 'distance_y', 'intensity', 'size', 'distance',
    'is_peak_cyclone_season', 
    'cyclone_season_weight', 'peak_cyclone_daytime_interaction', 'size_change_30', 'bearing'
]

# Add lag columns to feature_cols
for feat in lag_features:
    for lag_min in lag_intervals.keys():
        feature_cols.append(f"{feat}_{lag_min}")

X = train_df[feature_cols]
y_1h = train_df['Storm_NosyBe_1h']
y_3h = train_df['Storm_NosyBe_3h']

print(" --- Split training and validation sets ----")
# #### Split training and validation sets

x_full_train, x_test,y1h_full_train, y1h_test= train_test_split(X, y_1h, test_size=0.2, random_state=11)
x_train, x_val, y1h_train, y1h_val= train_test_split(x_full_train,y1h_full_train, test_size=0.25, random_state=11)

_, _, y3h_full_train, y3h_test = train_test_split(X, y_3h, test_size=0.2, random_state=11)
_, _, y3h_train, y3h_val= train_test_split(x_full_train,y3h_full_train, test_size=0.25, random_state=11)

# #### trainning the Modele
df_train = x_train.reset_index(drop=True)
df_val = x_val.reset_index(drop=True)
df_test = x_test.reset_index(drop=True)
df_full_train = x_full_train.reset_index(drop=True)

print("trainning the Modele -- model making")
# ##### Decision Tree

train_dicts = df_train.fillna(0).to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

dt_model_1h = DecisionTreeClassifier()
dt_model_1h.fit(X_train, y1h_train)

val_dicts = df_val.fillna(0).to_dict(orient='records')
X_val = dv.transform(val_dicts)

y_pred = dt_model_1h.predict_proba(X_val)[:, 1]
roc_auc_score(y1h_val, y_pred)

y_pred = dt_model_1h.predict_proba(X_train)[:, 1]
roc_auc_score(y1h_train, y_pred)

dt_model_1h = DecisionTreeClassifier(max_depth=2)
dt_model_1h.fit(X_train, y1h_train)

y_pred = dt_model_1h.predict_proba(X_train)[:, 1]
auc = roc_auc_score(y1h_train, y_pred)
print('train:', auc)

y_pred = dt_model_1h.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y1h_val, y_pred)
print('val:', auc)

# ##### Decision tree Tunning
# selecting max_depth
# selecting min_samples_leaf

print(" Decision tree Tunning")
depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, None]

for depth in depths: 
    dt_model_1h = DecisionTreeClassifier(max_depth=depth)
    dt_model_1h.fit(X_train, y1h_train)
    
    y_pred = dt_model_1h.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y1h_val, y_pred)
    
    print('%4s -> %.3f' % (depth, auc))


scores = []

for depth in [5, 6, 7]:
    for s in [1, 5, 10, 15, 20, 500, 100, 200]:
        dt_model_1h = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=s)
        dt_model_1h.fit(X_train, y1h_train)

        y_pred = dt_model_1h.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y1h_val, y_pred)
        
        scores.append((depth, s, auc))

columns = ['max_depth', 'min_samples_leaf', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)

df_scores_pivot = df_scores.pivot(index='min_samples_leaf', columns=['max_depth'], values=['auc'])
df_scores_pivot.round(3)

sns.heatmap(df_scores_pivot, annot=True, fmt=".3f")
plt.show()

dt_model_1h = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5)
dt_model_1h.fit(X_train, y1h_train)


print(export_text(dt_model_1h, feature_names=list(dv.get_feature_names_out())))


# ##### Trainning random forest
print("Trainning random forest")

# let's test Random forest 
scores = []

for n in range(10, 201, 10):
    rf_model_1h = RandomForestClassifier(n_estimators=n, random_state=1)
    rf_model_1h.fit(X_train, y1h_train)

    y_pred = rf_model_1h.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y1h_val, y_pred)
    
    scores.append((n, auc))

df_scores = pd.DataFrame(scores, columns=['n_estimators', 'auc'])

plt.plot(df_scores.n_estimators, df_scores.auc)
plt.show()


scores = []

for d in [5, 10, 15]:
    for n in range(10, 201, 10):
        rf_model_1h = RandomForestClassifier(n_estimators=n,
                                    max_depth=d,
                                    random_state=1)
        rf_model_1h.fit(X_train, y1h_train)

        y_pred = rf_model_1h.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y1h_val, y_pred)

        scores.append((d, n, auc))


columns = ['max_depth', 'n_estimators', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)


for d in [5, 10, 15]:
    df_subset = df_scores[df_scores.max_depth == d]
    
    plt.plot(df_subset.n_estimators, df_subset.auc,
             label='max_depth=%d' % d)

plt.legend()
plt.show()


# Let's use max depth = 15
print("Let's use max depth = 15 ")
max_depth = 10


scores = []

for s in [1, 3, 5, 10, 50]:
    for n in range(10, 201, 10):
        rf_model_1h = RandomForestClassifier(n_estimators=n,
                                    max_depth=max_depth,
                                    min_samples_leaf=s,
                                    random_state=1)
        rf_model_1h.fit(X_train, y1h_train)

        y_pred = rf_model_1h.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y1h_val, y_pred)

        scores.append((s, n, auc))


columns = ['min_samples_leaf', 'n_estimators', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)


colors = ['black', 'blue', 'orange', 'red', 'grey']
values = [1, 3, 5, 10, 50]

for s, col in zip(values, colors):
    df_subset = df_scores[df_scores.min_samples_leaf == s]
    
    plt.plot(df_subset.n_estimators, df_subset.auc,
             color=col,
             label='min_samples_leaf=%d' % s)
    

plt.legend()
plt.show()


# best min_samples_leaf 

min_samples_leaf = 5


rf = RandomForestClassifier(n_estimators=200,
                            max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf,
                            random_state=1)
rf.fit(X_train, y1h_train)
# ##### trainning with Boosting and XGBoost

#!pip install xgboost

features = list(dv.get_feature_names_out())
dtrain = xgb.DMatrix(X_train, label=y1h_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y1h_val, feature_names=features)


# ##### XGBoost Tunning 
xgb_params = {
    'eta': 0.3, 
    'max_depth': 6,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=10)

y_pred = model.predict(dval)


roc_auc_score(y1h_val, y_pred)

watchlist = [(dtrain, 'train'), (dval, 'val')]



get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.3, \n    'max_depth': 6,\n    'min_child_weight': 1,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n}\n\nxgb_model_1h = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\n")



s = output.stdout


print(s[:200])

def parse_xgb_output(output):
    results = []

    for line in output.stdout.strip().split('\n'):
        it_line, train_line, val_line = line.split('\t')

        it = int(it_line.strip('[]'))
        train = float(train_line.split(':')[1])
        val = float(val_line.split(':')[1])

        results.append((it, train, val))
    
    columns = ['num_iter', 'train_auc', 'val_auc']
    df_results = pd.DataFrame(results, columns=columns)
    return df_results


df_score = parse_xgb_output(output)

plt.plot(df_score.num_iter, df_score.train_auc, label='train')
plt.plot(df_score.num_iter, df_score.val_auc, label='val')
plt.legend()
plt.show()
plt.plot(df_score.num_iter, df_score.val_auc, label='val')
plt.legend()
plt.show()


# Tunning
# 

# eta
# max_depth
# min_child_weight

scores = {}

get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.3, \n    'max_depth': 6,\n    'min_child_weight': 1,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n}\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\n")

scores = {}


key = 'eta=%s' % (xgb_params['eta'])
scores[key] = parse_xgb_output(output)
key

scores = {}
get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.3, \n    'max_depth': 10,\n    'min_child_weight': 1,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\n")


key = 'max_depth=%s' % (xgb_params['max_depth'])
scores[key] = parse_xgb_output(output)
key

del scores['max_depth=10']


for max_depth, df_score in scores.items():
    plt.plot(df_score.num_iter, df_score.val_auc, label=max_depth)


plt.ylim(0.9, 0.94)
plt.xlim(0.0, 200)
plt.legend()
plt.show()


scores = {}


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.3, \n    'max_depth': 10,\n    'min_child_weight': 30,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\n")

key = 'min_child_weight=%s' % (xgb_params['min_child_weight'])
scores[key] = parse_xgb_output(output)
key

for min_child_weight, df_score in scores.items():
    plt.plot(df_score.num_iter, df_score.val_auc, label=min_child_weight)

plt.ylim(0.82, 0.94)
plt.legend()
plt.show()


xgb_params = {
    'eta': 0.3, 
    'max_depth': 10,
    'min_child_weight': 30,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=40)

print(" eta: 0.3 max_depth: 10 min_child_weigh : 30 objective: binary:logistic eval_metric : auc nthread : 8 seed : 1 verbosity: 1")
# ##### Selecting the final model
print("Selecting the final model")
# Choosing between xgboost, random forest and decision tree
# Training the final model
# Saving the model

# Decision Tree

print("Decision Tree")
dt_model_1h = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5)
dt_model_1h.fit(X_train, y1h_train)

y_pred = dt_model_1h.predict_proba(X_val)[:, 1]
roc_auc_score(y1h_val, y_pred)
print(roc_auc_score(y1h_val, y_pred))
# Random Forest
print("Random Forest  score : ")
rf_model_1h = RandomForestClassifier(n_estimators=200,
                            max_depth=10,
                            min_samples_leaf=5,
                            random_state=1)
rf_model_1h.fit(X_train, y1h_train)

y_pred = rf.predict_proba(X_val)[:, 1]

print(roc_auc_score(y1h_val, y_pred))

# Model XGboost
print("Model XGboost score : ")
xgb_params = {
    'eta': 0.3, 
    'max_depth': 10,
    'min_child_weight': 30,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=40)

y_pred = model.predict(dval)
print(roc_auc_score(y1h_val, y_pred))


# We will Train  the project with XGBOOST Modele that win
print("----------------END-------We will Train  the project with XGBOOST Modele that win-------------------")
dicts_full_train = df_full_train.fillna(0).to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)



dicts_test = df_test.fillna(0).to_dict(orient='records')
X_test = dv.transform(dicts_test)


dfulltrain = xgb.DMatrix(X_full_train, label=y1h_full_train,
                    feature_names=features)

dtest = xgb.DMatrix(x_test, label=y1h_test, feature_names=features)


xgb_params = {
    'eta': 0.3, 
    'max_depth': 10,
    'min_child_weight': 30,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dfulltrain, num_boost_round=40)



y_pred = model.predict(dtest)


roc_auc_score(y1h_test, y_pred)


# ##### Save the model
print("Save the model")

# Saving the model with pickle
with open('model_xboost.bin', 'wb') as file:
    pickle.dump((dv,model), file)

print("model at : model_xboost.bin ")
