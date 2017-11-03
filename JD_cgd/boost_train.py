

from basic_feats import *
import time
from datetime import datetime
from datetime import timedelta
import math
import pandas as pd
import pickle
import os
import numpy as np
#使用自举的方式，不断选取新的负样本训练集合
import extra_feat
import xgboost as xgb

def predict( actions ):
    bst = pickle.load(open('cache/model_user_sku.pkl', 'rb'))
    actions = actions.replace([np.inf, -np.inf], np.nan)
    actions = actions.fillna(0.)
    users = actions[['user_id', 'sku_id']].copy()
    labels = actions['label'].copy()
    del actions['label']
    del actions['user_id']
    del actions['sku_id']
    del actions['cate']
    data = xgb.DMatrix(actions.values)
    y_hat = bst.predict(data)
    return y_hat


def load_boost_df(train_start_date, train_end_date, test_start_date, test_end_date,
                       sample_neg = True, for_test = False,only_cate8 = False ):
    # train_start_date = '2016-02-10'
    # train_end_date = '2016-04-01'
    # dump_path = './cache/boost_%s_%s.pkl' % (train_start_date, train_end_date )
    # if for_test:
    #     dump_path = './cache/test_set_%s_%s.pkl' % (train_start_date, train_end_date)
    # if os.path.exists(dump_path):
    #     actions = pickle.load(open(dump_path,'rb'))
    # else:
    start_days = "2016-02-01"
    user = load_user_feat( )
    # print('user feature done.')
    product = load_product_feat( )
    # print('product feature done.')
    user_acc = get_accumulate_user_feat( start_days, train_end_date )
    # print('user acc feature done.')
    product_acc = get_accumulate_product_feat(start_days, train_end_date )
    # print('product acc feature done.')
    comment_acc = load_comment_feat(train_start_date, train_end_date)
    # print('comment acc feature done.')
    # u_vec = load_user_vector( )
    # p_vec = load_product_vector( )
    if not for_test:
        labels = get_labels(test_start_date, test_end_date, only_cate8=only_cate8)
        # print('positive samples:')
        # print(labels.shape)
    actions = None
    i_arr = ( 7, 3, 1, 30, 2,  5, 10,  15, 21)
    if for_test:
        i_arr = (7, 3, 1, 30, 2, 5, 10,  15, 21)
    for i in i_arr:
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')
        if actions is None:
            actions = get_action_feat(start_days, train_end_date, i)
        else:
            actions = pd.merge(actions, get_action_feat(start_days, train_end_date, i ), how='left',
                               on=['user_id', 'sku_id'])
    actions = actions.fillna(0.)
    #delete already buy.
    # if delete_already_buy:
    #     actions = actions[actions['30-action_4']==0.0 ]
    # print('actions feature done.')
    if not for_test:
        # actions = get_action_feat(train_start_date, train_end_date)
        actions = pd.merge(actions, labels, how='left', on=['user_id', 'sku_id'])
    actions = actions.drop_duplicates(['user_id','sku_id'])
    actions = pd.merge( actions, user, how='left', on=['user_id'])
    actions = pd.merge(actions, user_acc, how='left', on=['user_id'])
    actions = pd.merge( actions, product, how='left', on=['sku_id'])
    actions = pd.merge(actions, product_acc, how='left', on=['sku_id'])
    #如果不对comment数据进行分段，这里的merge comment_acc居然会增加样本的数量，导致数据(user_id,sku_id)重复。
    actions = pd.merge(actions, comment_acc, how='left', on=['sku_id'])
    # actions = pd.merge(actions, u_vec, how = 'left',on = ['user_id'])
    # actions = pd.merge(actions, p_vec, how = 'left', on=['sku_id'])
    if only_cate8:
        actions = actions[actions['cate'] == 8]
    actions = extra_feat.add_vectors( actions )
    y_hat = predict( actions )
    actions['y_hat'] = y_hat

    if (not for_test) and sample_neg:
        pos_actions = actions[actions['label'] == 1]
        thres = 0.5
        #负样本预测正确
        neg_actions_1 = actions[(actions['label'] != 1)&(actions['y_hat']<thres)]
        #负样本预测错误
        neg_actions_2 = actions[(actions['label'] != 1) & (actions['y_hat'] >= thres)]
        print('pos and neg samples: {}, {}, {}'.format(pos_actions.shape, neg_actions_1.shape, neg_actions_2.shape))
        # print(neg_actions.shape)
        num_pos = pos_actions.shape[0]
        times_pos = 5
        num_neg_1 = min(num_pos * times_pos, neg_actions_1.shape[0])
        if num_neg_1 != 0:
            neg_actions_1 = neg_actions_1.sample(n=num_neg_1)
        num_neg_2 = min(num_pos * times_pos, neg_actions_2.shape[0])
        if num_neg_2 != 0:
            neg_actions_2 = neg_actions_2.sample( n = num_neg_2)
        if num_neg_1 != 0 and num_neg_2 != 0:
            actions = pd.concat([pos_actions, neg_actions_2, neg_actions_1], ignore_index=True)
        elif num_neg_1 != 0:
            actions = pd.concat([pos_actions, neg_actions_1], ignore_index=True)
        else:
            actions = pd.concat([pos_actions, neg_actions_2], ignore_index=True)
    del actions['y_hat']
    actions['start_date'] = train_start_date
    actions['end_date'] = train_end_date
    # pickle.dump(actions, open(dump_path, 'wb'))
    # 所以没有label的全部都被当成了负样本。
    actions = actions.replace([np.inf, -np.inf], np.nan)
    actions = actions.fillna(0.)
    print('sub actions shape: {}'.format(actions.shape))
    return actions

#包含特征：基本特征、用户、产品的向量
def load_boost_all(train_start_dates, train_end_dates, test_start_dates, test_end_dates ):
    actions = []
    for train_start_date, train_end_date, test_start_date, test_end_date in \
        zip(train_start_dates, train_end_dates, test_start_dates, test_end_dates):
        print('{}\t{}'.format( train_start_date, train_end_date) )
        if train_start_date == '2016-02-06' or train_start_date == '2016-02-12'\
                or train_start_date == '2016-02-13':
            print('skip {}'.format(train_start_date))
            continue
        actions.append( load_boost_df(train_start_date, train_end_date,
                                           test_start_date, test_end_date ,only_cate8=only_keep_cate8) )
    actions = pd.concat(actions, ignore_index=True)
    # actions.to_csv('./cache/boost_feats.csv', index=False, index_label=False)
    return actions
    # actions = actions.replace([np.inf, -np.inf], np.nan)
    # actions = actions.fillna(0.)
    # users = actions[['user_id', 'sku_id']].copy( )
    # labels = actions['label'].copy( )
    # del actions['label']
    # del actions['user_id']
    # del actions['sku_id']
    # return users, actions, labels


if __name__ == '__main__':
    train_start_dates, train_end_dates, test_start_dates, test_end_dates = get_dates(all_data=True, step=3)
    actions = load_boost_all( train_start_dates, train_end_dates, test_start_dates, test_end_dates )
    # actions = pd.read_csv('./cache/boost_feats.csv')
    actions.to_csv('./cache/add_user_sku_feats.csv', index=False, index_label=False)
    actions = actions[ actions['start_date']<'2016-03-08' ]
    actions.to_csv('./cache/add_user_sku_feats_valid.csv', index=False, index_label=False)
    #调用model_laarn.tune_xgb()继续学习模型。