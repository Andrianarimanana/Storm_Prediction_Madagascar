#!/usr/bin/env python
# coding: utf-8

import pickle

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer

import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_text

# parameters



# n_splits = 5
output_file = f'model_xboost.bin'
#output_file = f'model_C={C}.bin'


# data preparation
train_df = pd.read_csv('./Data/train.csv')
test_df = pd.read_csv('./Data/test.csv')

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

# training 
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
#print(f'doing validation with C={C}')
#spilit 

x_full_train, x_test,y1h_full_train, y1h_test= train_test_split(X, y_1h, test_size=0.2, random_state=11)
x_train, x_val, y1h_train, y1h_val= train_test_split(x_full_train,y1h_full_train, test_size=0.25, random_state=11)

_, _, y3h_full_train, y3h_test = train_test_split(X, y_3h, test_size=0.2, random_state=11)
_, _, y3h_train, y3h_val= train_test_split(x_full_train,y3h_full_train, test_size=0.25, random_state=11)


df_train = x_train.reset_index(drop=True)
df_val = x_val.reset_index(drop=True)
df_test = x_test.reset_index(drop=True)

df_full_train = x_full_train.reset_index(drop=True)

# training the final model
dicts_full_train = df_full_train.fillna(0).to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)

dicts_test = df_test.fillna(0).to_dict(orient='records')
X_test = dv.transform(dicts_test)

dfulltrain = xgb.DMatrix(X_full_train, label=y1h_full_train,
                    feature_names=dv.get_feature_names_out())

dtest = xgb.DMatrix(x_test, label=y1h_test, feature_names=dv.get_feature_names_out)
print('training the final model')

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
# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')