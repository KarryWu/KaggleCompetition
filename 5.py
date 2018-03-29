#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 16:57:47 2018

@author: wukairui
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 17:10:50 2018

@author: wukairui
"""

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import gc
import lightgbm as lgb


dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

print('loading train data...')
train_df_origin = pd.read_csv("train.csv", skiprows=range(1,104903891), nrows=80000000, 
                       dtype=dtypes, usecols=['ip','app','device','os', 'channel', 
                                              'click_time', 'is_attributed'])

print('loading test data...')
test_df_origin = pd.read_csv("test.csv", dtype=dtypes, usecols=['ip','app','device',
                                                              'os', 'channel', 
                                                              'click_time', 'click_id'])


len_train = len(train_df_origin)
train_df=train_df_origin.append(test_df_origin)

del test_df_origin
gc.collect()

print('Extracting new features...')
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
train_df['click_time'] = pd.to_datetime(train_df.click_time)
gc.collect()

def count_click(variables,train_df):
    name = 'count_on_'
    for i in range(len(variables)):
        name = name + variables[i]
    gp = train_df.groupby(variables).size().reset_index(name = name)
    train_df = train_df.merge(gp, on=variables, how='left')
    return train_df


train_df = count_click(['ip','day','hour'],train_df)
gc.collect()

train_df = count_click(['ip','app','hour'],train_df)
gc.collect()

train_df = count_click(['ip','channel','hour'],train_df)
gc.collect()

train_df = count_click(['ip','channel','day','hour'],train_df)
gc.collect()

train_df = count_click(['ip','app'],train_df)
gc.collect()

train_df = count_click(['ip','channel'],train_df)
gc.collect()

train_df = count_click(['ip','day'],train_df)
gc.collect()

train_df = count_click(['ip','hour'],train_df)
gc.collect()

train_df = count_click(['ip','os','device'],train_df)
gc.collect()

train_df = count_click(['channel','day','hour'],train_df)
gc.collect()

train_df = count_click(['day','hour'],train_df)
gc.collect()

train_df = count_click(['ip'],train_df)
gc.collect()

#train_df = count_click(['day'],train_df)
#gc.collect()

#train_df = count_click(['hour'],train_df)
#gc.collect()

train_df = count_click(['channel'],train_df)
gc.collect()

train_df = count_click(['app'],train_df)
gc.collect()

##train_df = count_click(['device'],train_df)
##gc.collect()

##train_df = count_click(['os'],train_df)
##gc.collect()
###
###


GROUP_BY_NEXT_CLICKS = [
    {'groupby': ['ip']},
    {'groupby': ['ip', 'app']},
    {'groupby': ['ip', 'channel']},
    {'groupby': ['ip', 'os']},
]

# Calculate the time to next click for each group
for spec in GROUP_BY_NEXT_CLICKS:
    
    # Name of new feature
    new_feature = '{}_nextClick'.format('_'.join(spec['groupby']))    
    
    # Unique list of features to select
    all_features = spec['groupby'] + ['click_time']
    
    # Run calculation
    print(f">> Grouping by {spec['groupby']}, and saving time to next click in: {new_feature}")
    d = train_df[all_features].groupby(spec['groupby']).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
    print(len(d))
    train_df[new_feature] = d
    
gc.collect()

HISTORY_CLICKS = {
    'identical_clicks': ['ip', 'app', 'device', 'os', 'channel'],
    'app_clicks': ['ip', 'app']
}

# Go through different group-by combinations
for fname, fset in HISTORY_CLICKS.items():
    
    # Clicks in the past
    train_df['prev_'+fname] = train_df. \
        groupby(fset). \
        cumcount(). \
        rename('prev_'+fname)
        
    # Clicks in the future
    train_df['future_'+fname] = train_df.iloc[::-1]. \
        groupby(fset). \
        cumcount(). \
        rename('future_'+fname).iloc[::-1]

gc.collect()
    
test_df1 = train_df[len_train:]
val_df1 = train_df[(len_train-8000000):len_train]
train_df1 = train_df[:(len_train-8000000)]

print("train size: ", len(train_df1))
print("valid size: ", len(val_df1))
print("test size : ", len(test_df1))

target = 'is_attributed'
categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']

train_X = train_df1.drop(['ip','click_time','is_attributed'],axis = 1)
val_X = val_df1.drop(['ip','click_time','is_attributed'],axis = 1)
test_X = test_df1.drop(['ip','click_time','is_attributed'],axis = 1)

train_Y = train_df1['is_attributed']
val_Y = val_df1['is_attributed']

sub = pd.DataFrame()
sub['click_id'] = test_df1['click_id'].astype('int')

gc.collect()


print("Preparing the datasets for training...")

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 255,  
    'max_depth': 9,  
    'min_child_samples': 100,  
    'max_bin': 100,  
    'subsample': 0.7,  
    'subsample_freq': 1,  
    'colsample_bytree': 0.7,  
    'min_child_weight': 0,  
    'subsample_for_bin': 200000,  
    'min_split_gain': 0,  
    'reg_alpha': 0,  
    'reg_lambda': 0,  
   # 'nthread': 8,
    'verbose': 0,
   # 'is_unbalance': True
    'scale_pos_weight':99 
    }
    
dtrain = lgb.Dataset(train_X, label=train_Y,
                      categorical_feature=categorical
                      )
dvalid = lgb.Dataset(val_X, label=val_Y,
                      categorical_feature=categorical, reference=dtrain
                      )
                      

print("Training the model...")

lgb_model = lgb.train(params, dtrain,valid_sets=dvalid,num_boost_round=1000,
                 early_stopping_rounds=30)

gc.collect()

print('Save model...')
# save model to file
lgb_model.save_model('model6.txt')

print("Preparing data for submission...")

submit = pd.read_csv('sample_submission.csv', dtype='int', usecols=['click_id'])

print("Predicting the submission data...")

submit['is_attributed'] = lgb_model.predict(test_X, num_iteration=lgb_model.best_iteration)

print("Writing the submission data into a csv file...")

submit.to_csv("submission6.csv",index=False)

print("All done...")