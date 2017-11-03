from sklearn.model_selection import train_test_split
# import xgboost as xgb
import time
from datetime import datetime
from datetime import timedelta
import math
import pandas as pd
import pickle
import os
import numpy as np


dir = 'data/'
COMMENT_FILE = dir+"JData_Comment.csv"
PRODUCT_FILE = dir+"JData_Product.csv"
USER_FILE = dir+"JData_User.csv"
ACTION_201602_FILE = dir+"JData_Action_201602.csv"
ACTION_201603_FILE = dir+"JData_Action_201603.csv"
ACTION_201604_FILE = dir+"JData_Action_201604.csv"

def convert_age(age_str):
    if age_str == u'-1':
        return 0
    elif age_str == u'15岁以下':
        return 1
    elif age_str == u'16-25岁':
        return 2
    elif age_str == u'26-35岁':
        return 3
    elif age_str == u'36-45岁':
        return 4
    elif age_str == u'46-55岁':
        return 5
    elif age_str == u'56岁以上':
        return 6
    else:
        return -1

#产生(sku, attr) (user_id, attr)的pairs
def gen_triples():
    user_age = 'user_age'
    user_sex = 'user_sex'
    user_lv = 'user_lv'
    product_a1 = 'product_a1'
    product_a2 = 'product_a2'
    product_a3 = 'product_a3'
    product_cate = 'product_cate'
    product_brand = 'product_brand'

    triples = []
    user = pd.read_csv(USER_FILE)
    user['age'] = user['age'].map(convert_age)
    for index, row in user.iterrows():
        user_id = 'U_{}'.format( row['user_id'] )
        triples.append( (user_id, user_age, 'age_{}'.format(row['age']) ))
        triples.append( (user_id, user_sex, 'sex_{}'.format(row['sex']) ))
        triples.append( (user_id, user_lv, 'lv_{}'.format(row['user_lv_cd']) ))
    user = None
    print('user done.')

    product = pd.read_csv(PRODUCT_FILE)
    for index, row in product.iterrows():
        sku_id = 'S_{}'.format( row['sku_id'] )
        triples.append( (sku_id,product_a1, 'a1_{}'.format( row['a1'])) )
        triples.append((sku_id, product_a2, 'a2_{}'.format(row['a2'])) )
        triples.append((sku_id, product_a3, 'a3_{}'.format(row['a3'])) )
        triples.append((sku_id, product_cate, 'cate_{}'.format(row['cate'])))
        triples.append( (sku_id, product_brand, 'brand_{}'.format(row['brand'])))
    product = None
    print('product done.')

    seps = ('1','2','4','5','6')
    for sep in seps:
        rel = 'actions_{}'.format(sep)
        # fname = 'up_matrix_{}.csv'.format(sep)
        with open(dir+'up_matrix_{}.csv'.format(sep),'r' ) as fin:
            fin.readline()
            for line in fin:
                terms = line.split(',')
                user_id = 'U_'+str( int(float(terms[1])) )
                sku_id = 'S_'+str( int(terms[2]) )
                triples.append( (user_id, rel, sku_id ) )
    print('actions done.')

    with open(dir+'triples.csv','w') as fout:
        for triple in triples:
            fout.write(','.join(triple))
            fout.write('\n')


def get_from_action_data(fname, chunk_size=10000000):
    reader = pd.read_csv(fname, header=0, iterator=True)
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)[['sku_id', "cate", "brand"]]
            chunk = chunk.drop_duplicates(['sku_id'])
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")
    # 将块拼接为pandas dataframe格式
    df_ac = pd.concat(chunks, ignore_index=True)
    df_ac = df_ac.drop_duplicates(['sku_id'])
    return df_ac

def get_actions( ):
    df_ac = []
    df_ac.append(get_from_action_data(ACTION_201602_FILE))
    df_ac.append(get_from_action_data(ACTION_201603_FILE))
    df_ac.append(get_from_action_data(ACTION_201604_FILE))
    actions = pd.concat(df_ac, ignore_index=True)
    actions.drop_duplicates(['sku_id'])
    return actions

def add_triples():
    product_cate = 'product_cate'
    product_brand = 'product_brand'
    triples = []
    actions = get_actions( )
    for index, row in actions.iterrows():
        if row['cate'] != 8:
            sku_id = 'S_{}'.format(row['sku_id'])
            triples.append((sku_id, product_cate, 'cate_{}'.format(row['cate'])))
            triples.append((sku_id, product_brand, 'brand_{}'.format(row['brand'])))
    with open(dir+'triples_extra.csv','w') as fout:
        for triple in triples:
            fout.write(','.join(triple))
            fout.write('\n')




if __name__ == '__main__':
    gen_triples( )
    add_triples( )
    print('done.')