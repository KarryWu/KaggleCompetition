#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 19:00:47 2018

@author: wukairui
"""

import pandas as pd
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import gc
import lightgbm as lgb
import time


dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

total_rows = 184903890
read_rows = 80000000


print('loading train data...')
train_df_origin = pd.read_csv("train.csv", skiprows=range(1,total_rows-read_rows), nrows=read_rows, 
                       dtype=dtypes, usecols=['ip','app','device','os', 'channel', 
                                              'click_time', 'is_attributed'])

print('loading test data...')
test_df_origin = pd.read_csv("test.csv", dtype=dtypes, usecols=['ip','app','device',
                                                              'os', 'channel', 
                                                              'click_time', 'click_id'])

def merge_median(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].median()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_mean(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].mean()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df


def count_click(variables,train_df):
    name = 'count_on_'
    for i in range(len(variables)):
        name = name + variables[i]
    feature = train_df.groupby(variables).size().reset_index(name = name)
    train_df = train_df.merge(feature, on=variables, how='left')
    del feature
    gc.collect()
    return train_df


def cumcount_click(variables,train_df):
    name = 'cumcount_on_'
    for i in range(len(variables)):
        name = name + variables[i]
    train_df[name] = train_df.groupby(variables).cumcount()
    gc.collect()
    return train_df


def create_features(train_df):

    print('Simple preparation >>>>>>')
    start_time = time.time()

    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    train_df['wday'] = pd.to_datetime(train_df.click_time).dt.dayofweek.astype('uint8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    train_df['click_time'] = pd.to_datetime(train_df.click_time)
    gc.collect()

    print("Simple preparation finished  in {}".format(time.time()-start_time))


    print("Creating groupby features >>>>>>")
    start_time = time.time()

    
    train_df = cumcount_click(['ip','wday','hour'],train_df)
    print('Done>>>>')
    train_df = cumcount_click(['ip','wday','hour','os'],train_df)
    print('Done>>>>')
    train_df = cumcount_click(['ip','day','hour','app'],train_df)
    print('Done>>>>')
    train_df = cumcount_click(['ip','day','hour','app','os'],train_df)
    print('Done>>>>')
    train_df = cumcount_click(['app','wday','hour'],train_df)
    print('Done>>>>')
    

    train_df = count_click(['ip','day','hour'],train_df)
    print('Done>>>>')
    train_df = count_click(['ip','app','hour'],train_df)
    print('Done>>>>')
    train_df = count_click(['ip','channel','hour'],train_df)
    print('Done>>>>')
    train_df = count_click(['ip','app','channel','os'],train_df)
    print('Done>>>>')
    train_df = count_click(['ip','app','channel','device'],train_df)
    print('Done>>>>')
    train_df = count_click(['ip','app','channel','device','os'],train_df)
    print('Done>>>>')
    train_df = count_click(['ip','app','channel','device','os','day'],train_df)
    print('Done>>>>')
    train_df = count_click(['ip','app','channel','device','os','day','hour'],train_df)
    print('Done>>>>')
    train_df = count_click(['ip','app','channel','device','os','hour'],train_df)
    print('Done>>>>')
    train_df = count_click(['ip','channel','os','device'],train_df)
    print('Done>>>>')
    train_df = count_click(['ip','channel','day','hour'],train_df)
    print('Done>>>>')
    train_df = count_click(['ip','app'],train_df)
    print('Done>>>>')
    train_df = count_click(['ip','channel'],train_df)
    print('Done>>>>')
    train_df = count_click(['ip','day'],train_df)
    print('Done>>>>')
    train_df = count_click(['ip','hour'],train_df)
    print('Done>>>>')
    train_df = count_click(['ip','os','device'],train_df)
    print('Done>>>>')
    train_df = count_click(['channel','day','hour'],train_df)
    print('Done>>>>')
    train_df = count_click(['day','hour'],train_df)
    print('Done>>>>')

    print("Finished creating groupby features in {}".format(time.time()-start_time))

    GROUP_BY_NEXT_CLICKS = [
        {'groupby': ['ip', 'channel','app']},
        {'groupby': ['ip','channel','day','hour']},
        {'groupby': ['ip','app','channel','device','os','hour']},
        {'groupby': ['ip','app','channel','device','os','day','hour']},
        {'groupby': ['ip','app','channel','device','os','day','hour']},
        {'groupby': ['ip','app','channel','device','os']}
    ]

    print("Creating next/previous features >>>>>>")
    start_time = time.time()



    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_CLICKS:
        
        # Name of new feature
        new_feature = '{}_nextClick'.format('_'.join(spec['groupby']))    
        
        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']
        
        # Run calculation
        print(f">> Grouping by {spec['groupby']}, and saving time to next click in: {new_feature}")
        d = train_df[all_features].groupby(spec['groupby']).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
        train_df[new_feature] = d
        del d
        #gc.collect()
        
    gc.collect()

    GROUP_BY_PREV_CLICKS = [
        {'groupby': ['ip', 'app']},
        {'groupby': ['ip', 'channel','app','os']},
        {'groupby': ['ip', 'channel']},
        {'groupby': ['ip', 'os']},
        {'groupby': ['ip', 'channel','app']},
        {'groupby': ['ip','channel','day','hour']},
        {'groupby': ['ip','app','channel','device','os','hour']},
        {'groupby': ['ip','app','channel','device','os','day','hour']},
        {'groupby': ['ip','app','channel','device','os','day','hour']},
        {'groupby': ['ip','app','channel','device','os']}
    ]


    previouscolsname = []
    for spec in GROUP_BY_PREV_CLICKS:
        
        # Name of new feature
        new_feature = '{}_previousClick'.format('_'.join(spec['groupby']))  

        previouscolsname.append(new_feature)
        
        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']
        
        # Run calculation
        print(f">> Grouping by {spec['groupby']}, and saving time to next click in: {new_feature}")
        d = train_df[all_features].groupby(spec['groupby']).click_time.transform(lambda x: x.diff().shift(1)).dt.seconds
        train_df[new_feature] = d
        del d
        gc.collect()
        
    gc.collect()

    print("Finished creating next/previous features in {}".format(time.time()-start_time))

    '''

    FOR_STATS = [
        {'groupby': ['ip','channel','day','hour']},
    ]

    for column in previouscolsname:
        for spec in FOR_STATS:
            train_df = cumcount_click(['ip','wday','hour'],train_df)
            vich = train_df.groupby(spec['groupby'])[column].mean()
            train_df[column+'mean'] = vich
            del vich
            gc.collect()
            vich = train_df.groupby(spec['groupby'])[column].median()
            train_df[column+'median'] = vich
            del vich
            gc.collect()

    '''


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

    print('Features extracted>>>>>')

    return train_df


train_df_origin = create_features(train_df_origin)
test_df_origin = create_features(test_df_origin)

train_df_origin.to_csv('TrainDF_Kairui.csv', sep = '\t')
test_df_origin.to_csv('TestDF_Kairui.csv', sep = '\t')