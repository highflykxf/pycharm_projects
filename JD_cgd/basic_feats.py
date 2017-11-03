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
ACTION_201602_FILE = dir+"JData_Action_201602.csv"
ACTION_201603_FILE = dir+"JData_Action_201603.csv"
ACTION_201604_FILE = dir+"JData_Action_201604.csv"
COMMENT_FILE = dir+"JData_Comment.csv"
PRODUCT_FILE = dir+"JData_Product.csv"
USER_FILE = dir+"JData_User.csv"
USER_TABLE_FILE = dir+"user_table.csv"
PRODUCT_TABLE_FILE = dir+"item_table.csv"


delete_already_buy = False
only_keep_cate8 = False

comment_date = ["2016-02-01", "2016-02-08", "2016-02-15", "2016-02-22", "2016-02-29", "2016-03-07", "2016-03-14",
                "2016-03-21", "2016-03-28",
                "2016-04-04", "2016-04-11", "2016-04-15"]

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


def load_user_feat():
    dump_path = './cache/basic_user.pkl'
    if os.path.exists(dump_path):
        user = pickle.load(open(dump_path,'rb'))
    else:
        user = pd.read_csv(USER_FILE)
        user['age'] = user['age'].map(convert_age)
        age_df = pd.get_dummies(user["age"], prefix="age")
        sex_df = pd.get_dummies(user["sex"], prefix="sex")
        user_lv_df = pd.get_dummies(user["user_lv_cd"], prefix="user_lv_cd")
        user = pd.concat([user['user_id'], age_df, sex_df, user_lv_df], axis=1)
        pickle.dump(user, open(dump_path, 'wb'))
    return user

def load_product_feat():
    dump_path = './cache/basic_product.pkl'
    if os.path.exists(dump_path):
        product = pickle.load(open(dump_path,'rb'))
    else:
        product = pd.read_csv(PRODUCT_FILE)
        attr1_df = pd.get_dummies(product["a1"], prefix="a1")
        attr2_df = pd.get_dummies(product["a2"], prefix="a2")
        attr3_df = pd.get_dummies(product["a3"], prefix="a3")
        product = pd.concat([product[['sku_id', 'cate', 'brand']], attr1_df, attr2_df, attr3_df], axis=1)
        pickle.dump(product, open(dump_path, 'wb'))
    return product


def load_comment_feat(start_date, end_date):
    dump_path = './cache/comments_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        comments = pickle.load(open(dump_path,'rb'))
    else:
        comments = pd.read_csv(COMMENT_FILE)
        comment_date_end = end_date
        #TODO 这样筛选可能会遗漏一些comment
        comment_date_begin = comment_date[0]
        #这个操作可能可以解决有多个comment统计数据的情况。
        #如果不去重，那么在后面的merge left join过程中会产生多余的数据。
        for date in reversed(comment_date):
            if date < comment_date_end:
                comment_date_begin = date
                break
        comments = comments[(comments.dt >= comment_date_begin) & (comments.dt < comment_date_end)]
        # 由于评论数<4，所以直接用one-hot表示。。。
        # comments.drop_duplicates(['sku_id'])
        df = pd.get_dummies(comments['comment_num'], prefix='comment_num')
        comments = pd.concat([comments, df], axis=1)  # type: pd.DataFrame
        # del comments['dt']
        # del comments['comment_num']
        comments = comments[
            ['sku_id', 'has_bad_comment', 'bad_comment_rate', 'comment_num_1', 'comment_num_2', 'comment_num_3',
             'comment_num_4']]
        pickle.dump(comments, open(dump_path, 'wb'))
    return comments


# def get_accumulate_user_feat():
#     import data
#     actions = pd.read_csv(USER_TABLE_FILE)
#     # df = pd.get_dummies(actions['type'], prefix='action')
#     # actions = pd.concat([actions['user_id'], df], axis=1)
#     # actions = actions.groupby(['user_id'], as_index=False).sum()
#     # actions['user_action_1_ratio'] = actions['action_4'] / actions['action_1']
#     # actions['user_action_2_ratio'] = actions['action_4'] / actions['action_2']
#     # actions['user_action_3_ratio'] = actions['action_4'] / actions['action_3']
#     # actions['user_action_5_ratio'] = actions['action_4'] / actions['action_5']
#     # actions['user_action_6_ratio'] = actions['action_4'] / actions['action_6']
#     # feature = ['user_id', 'user_action_1_ratio', 'user_action_2_ratio', 'user_action_3_ratio',
#     #            'user_action_5_ratio', 'user_action_6_ratio']
#     #这个只是购买量占点击量的比率，并不是真正的转换率
#     feature = ['user_id', 'buy_addcart_ratio','buy_browse_ratio','buy_click_ratio','buy_favor_ratio']
#     actions = actions[feature]
#     return actions
#
#
# def get_accumulate_product_feat( ):
#     feature = ['sku_id', 'buy_addcart_ratio','buy_browse_ratio','buy_click_ratio','buy_favor_ratio']
#     actions = pd.read_csv(PRODUCT_TABLE_FILE)
#     actions = actions[feature]
#     return actions

def get_accumulate_user_feat(start_date, end_date):
    feature = ['user_id', 'user_action_1_ratio', 'user_action_2_ratio', 'user_action_3_ratio',
               'user_action_5_ratio', 'user_action_6_ratio']
    dump_path = './cache/user_feat_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path,'rb'))
    else:
        actions = get_actions(start_date, end_date)
        # actions = actions[['user_id','type']]
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions['user_id'], df], axis=1)
        actions = actions.groupby(['user_id'], as_index=False).sum()
        actions['user_action_1_ratio'] = actions['action_4'] / actions['action_1']
        actions['user_action_2_ratio'] = actions['action_4'] / actions['action_2']
        actions['user_action_3_ratio'] = actions['action_4'] / actions['action_3']
        actions['user_action_5_ratio'] = actions['action_4'] / actions['action_5']
        actions['user_action_6_ratio'] = actions['action_4'] / actions['action_6']
        actions = actions[feature]
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions


def get_accumulate_product_feat(start_date, end_date):
    feature = ['sku_id', 'product_action_1_ratio', 'product_action_2_ratio', 'product_action_3_ratio',
               'product_action_5_ratio', 'product_action_6_ratio']
    dump_path = './cache/product_feat_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path,'rb'))
    else:
        actions = get_actions(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions['sku_id'], df], axis=1)
        actions = actions.groupby(['sku_id'], as_index=False).sum()
        actions['product_action_1_ratio'] = actions['action_4'] / actions['action_1']
        actions['product_action_2_ratio'] = actions['action_4'] / actions['action_2']
        actions['product_action_3_ratio'] = actions['action_4'] / actions['action_3']
        actions['product_action_5_ratio'] = actions['action_4'] / actions['action_5']
        actions['product_action_6_ratio'] = actions['action_4'] / actions['action_6']
        actions = actions[feature]
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions


def load_product_vector():
    p_vector = pd.read_csv( dir + 'tsku_1.mat')
    return p_vector

def load_user_vector():
    u_vector = pd.read_csv(dir+'tuser_1.mat')
    return u_vector

def get_from_action_data(fname, start_date, end_date, chunk_size=1000000):
    reader = pd.read_csv(fname, header=0, iterator=True)
    chunks = []
    loop = True
    while loop:
        try:
            # 只读取user_id和type两个字段
            chunk = reader.get_chunk(chunk_size)[['user_id','sku_id', "type", "time"]]
            chunk = chunk[(chunk['time']>=start_date )& (chunk['time']<end_date) ]
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")
    # 将块拼接为pandas dataframe格式
    df_ac = pd.concat(chunks, ignore_index=True)
    # df_ac = df_ac.drop_duplicates(['user_id','sku_id'])
    # df_ac.sort_values( )
    return df_ac

def get_actions(start_date, end_date):
    # dump_path = './cache/all_action_%s_%s.pkl' % (start_date, end_date)
    # if os.path.exists(dump_path):
    #     actions = pickle.load(open(dump_path,'rb'))
    # else:
    df_ac = []
    df_ac.append(get_from_action_data(ACTION_201602_FILE, start_date, end_date))
    df_ac.append(get_from_action_data(ACTION_201603_FILE, start_date, end_date))
    df_ac.append(get_from_action_data(ACTION_201604_FILE, start_date, end_date))
    actions = pd.concat(df_ac, ignore_index=True)
    # df_ac = df_ac[(df_ac['time'] >= start_date) & (df_ac['time'] < end_date)]
    # 用户在不同action表中统计量求和
    # df_ac = df_ac.groupby(['user_id','sku_id'], as_index=False).sum()
    # df_ac = df_ac['user_id','sku_id','type']
    # df_ac = df_ac.drop_duplicates(['user_id','sku_id','type'])
    # pickle.dump(actions, open(dump_path, 'wb'))
    return actions

# def get_action_feat(start_date, end_date):
#     dump_path = './cache/action_accumulate_%s_%s.pkl' % (start_date, end_date)
#     if os.path.exists(dump_path):
#         actions = pickle.load(open(dump_path))
#     else:
#         actions = get_actions(start_date, end_date)
#         actions = actions[['user_id', 'sku_id', 'type']]
#         #这是一个十分稀疏的特征，action类型按照不同时间段进行划分。
#         df = pd.get_dummies(actions['type'], prefix='%s-%s-action' % (start_date, end_date))
#         actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
#         actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
#         del actions['type']
#         pickle.dump(actions, open(dump_path, 'wb'))
#     return actions


def get_from_action_data_type(fname, chunk_size=1000000, only_cate8 = False):
    reader = pd.read_csv(fname, header=0, iterator=True)
    chunks = []
    loop = True
    while loop:
        try:
            # 只读取user_id和type两个字段
            chunk = reader.get_chunk(chunk_size)[['user_id','sku_id', "time",'type','cate']]
            if only_cate8:
                chunk = chunk[ (chunk['type'] == 4) & (chunk['cate']==8)]
            else:
                chunk = chunk[(chunk['type'] == 4) ]
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")
    # 将块拼接为pandas dataframe格式
    df_ac = pd.concat(chunks, ignore_index=True)
    # df_ac = df_ac.drop_duplicates(['user_id','sku_id'])
    return df_ac

def get_actions_type(start_date, end_date, only_cate8 = False):
    df_ac = []
    df_ac.append(get_from_action_data_type(fname=ACTION_201602_FILE, only_cate8=only_cate8))
    df_ac.append(get_from_action_data_type(fname=ACTION_201603_FILE, only_cate8=only_cate8))
    df_ac.append(get_from_action_data_type(fname=ACTION_201604_FILE, only_cate8=only_cate8))
    df_ac = pd.concat(df_ac, ignore_index=True)
    df_ac = df_ac[(df_ac['time'] >= start_date) & (df_ac['time'] < end_date)]
    df_ac = df_ac[['user_id','sku_id']]
    df_ac = df_ac.drop_duplicates(['user_id','sku_id'])
    return df_ac

def get_action_feat(start_date, end_date, back_days):
    dump_path = './cache/action_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path,'rb'))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[['user_id', 'sku_id', 'type']]
        #这是一个十分稀疏的特征，action类型按照不同时间段进行划分。
        df = pd.get_dummies(actions['type'], prefix='{}-action'.format(back_days))
        actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum( )
        del actions['type']
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions


def get_labels(start_date, end_date, only_cate8 = False):
    dump_path = './cache/labels_{}_{}_cate8{}.pkl'.format(start_date, end_date, only_cate8)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path,'rb'))
    else:
        actions = get_actions_type(start_date, end_date, only_cate8=only_cate8)
        # actions = actions[actions['type'] == 4]
        # actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        actions['label'] = 1
        actions = actions[['user_id', 'sku_id', 'label']]
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions

def load_train_test_df(train_start_date, train_end_date, test_start_date, test_end_date,
                       sample_neg = True, for_test = False, only_cate8 = False ):
    # train_start_date = '2016-02-10'
    # train_end_date = '2016-04-01'
    dump_path = './cache/train_set_%s_%s.pkl' % (train_start_date, train_end_date )
    if for_test:
        dump_path = './cache/test_set_%s_%s.pkl' % (train_start_date, train_end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path,'rb'))
        # pickle.dump(actions, open(dump_path, 'wb'))
    else:
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
        actions = None
        i_arr = (   5, 3, 1, 7, 30, 2, 10,  15, 21)
        if for_test:
            i_arr = ( 5, 3, 1, 7, 30, 2, 10,  15, 21)
        for i in i_arr:
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            if actions is None:
                actions = get_action_feat(start_days, train_end_date, i)
            else:
                actions = pd.merge(actions, get_action_feat(start_days, train_end_date, i ), how='left',
                                   on=['user_id', 'sku_id'])
        # print('actions feature done.')
        idx = ((actions['5-action_2'] > 0)) & \
              (actions['5-action_3'] <= 0) & \
              (actions['5-action_4'] <= 0)
        actions = actions[idx]
        if not for_test:
            # actions = get_action_feat(train_start_date, train_end_date)
            actions = pd.merge(actions, labels, how='left', on=['user_id', 'sku_id'])
        actions = pd.merge(actions, user, how='left', on=['user_id'])
        actions = pd.merge(actions, user_acc, how='left', on=['user_id'])
        actions = pd.merge(actions, product, how='left', on=['sku_id'])
        actions = pd.merge(actions, product_acc, how='left', on=['sku_id'])
        #如果不对comment数据进行分段，这里的merge comment_acc居然会增加样本的数量，导致数据(user_id,sku_id)重复。
        actions = pd.merge(actions, comment_acc, how='left', on=['sku_id'])
        # actions = pd.merge(actions, u_vec, how = 'left',on = ['user_id'])
        # actions = pd.merge(actions, p_vec, how = 'left', on=['sku_id'])
        if only_cate8:
            actions = actions[actions['cate'] == 8]
        if (not for_test) and sample_neg:
            pos_actions = actions[actions['label'] == 1]
            neg_actions = actions[actions['label'] != 1]
            print('pos and neg samples: {}'.format(pos_actions.shape))
            # print(neg_actions.shape)
            num_pos = pos_actions.shape[0]
            num_neg = min(num_pos * 5, neg_actions.shape[0])
            neg_actions = neg_actions.sample(n=num_neg)
            actions = pd.concat([neg_actions, pos_actions], ignore_index=True)
        actions['start_date'] = train_start_date
        actions['end_date'] = train_end_date
        #正在调试，暂时不缓存
        # pickle.dump(actions, open(dump_path, 'wb'))
    print('sub actions shape: {}'.format(actions.shape))
    return actions

#test数据labels都是0，只要把labels都去除就可以了。
def load_train_test(train_start_date, train_end_date, test_start_date, test_end_date,
                    sample_neg = True, for_test = False,only_cate8=False ):
    actions = load_train_test_df(train_start_date, train_end_date,test_start_date,
                                     test_end_date, sample_neg, for_test, only_cate8=only_cate8)
    # 所以没有label的全部都被当成了负样本。
    actions = actions.replace([np.inf, -np.inf], np.nan)
    actions = actions.fillna(0.)
    users = actions[['user_id', 'sku_id']].copy( )
    labels = None
    if not for_test:
        labels = actions['label'].copy()
        del actions['label']
    del actions['user_id']
    del actions['sku_id']

    return users, actions, labels




def load_train_test_all(train_start_dates, train_end_dates, test_start_dates, test_end_dates ):
    # dump_path = './cache/train_set_all.pkl'
    # if os.path.exists(dump_path):
    #     actions = pickle.load(open(dump_path,'rb'))
    # else:
    actions = []
    for train_start_date, train_end_date, test_start_date, test_end_date in \
        zip(train_start_dates, train_end_dates, test_start_dates, test_end_dates):
        print('{}\t{}'.format( train_start_date, train_end_date) )
        actions.append( load_train_test_df(train_start_date, train_end_date,
                                           test_start_date, test_end_date ,sample_neg=False, only_cate8=only_keep_cate8) )

    actions = pd.concat(actions, ignore_index=True)
    actions.to_csv('./data/basic_feats.csv', index=False, index_label=False)
    # pickle.dump( actions, open(dump_path,'wb') )
    # actions = actions.replace([np.inf, -np.inf], np.nan)
    # actions = actions.fillna(0.)
    # users = actions[['user_id', 'sku_id']].copy( )
    # labels = actions['label'].copy( )
    # del actions['label']
    # del actions['user_id']
    # del actions['sku_id']
    # del actions['start_date']
    # del actions['end_date']
    # return users, actions, labels

def get_dates( all_data =True, step = 1):
    train_start_dates = []
    train_end_dates = []
    test_start_dates = []
    test_end_dates = []
    # start = '2016-02-01'
    start = '2016-03-01'
    #用于本地调试
    end = '2016-03-06'
    if all_data:
        #用于训练模型提交结果
        end = '2016-03-12'
    start_days = datetime.strptime(start, '%Y-%m-%d')
    end_days = datetime.strptime(end, '%Y-%m-%d')
    while start_days <= end_days:
        sd = start_days.strftime('%Y-%m-%d')
        ed = (start_days + timedelta(days=30)).strftime('%Y-%m-%d')
        tsd = (start_days + timedelta(days=30)).strftime('%Y-%m-%d')
        ted = (start_days + timedelta(days=30 + 5)).strftime('%Y-%m-%d')
        train_start_dates.append(sd)
        train_end_dates.append(ed)
        test_start_dates.append(tsd)
        test_end_dates.append(ted)
        start_days += timedelta(days=step)
    return train_start_dates, train_end_dates, test_start_dates, test_end_dates

def gen_all():
    # 生成用于训练、测试的数据
    train_start_dates, train_end_dates, test_start_dates, test_end_dates = get_dates(all_data=True, step=1)
    load_train_test_all(train_start_dates, train_end_dates, test_start_dates, test_end_dates)
    # training_data = None
    # user_index=None
    # label = None
    # dump_path = './cache/train_set_all.pkl'
    # if os.path.exists(dump_path):
    #     actions = pickle.load(open(dump_path, 'rb'))
    #     actions.to_csv('./basic_feats.csv', index=False, index_label=False)
    #     actions = None
    print('train dataset done.')
    # 用于线下测试
    sub_start_date = '2016-03-12'
    sub_end_date = '2016-04-11'  # 不包含
    sub_test_start_date = '2016-04-11'
    sub_test_end_date = '2016-04-16'  # 不包含
    actions = load_train_test_df(sub_start_date, sub_end_date, None,
                                 None, sample_neg=False, for_test=True, only_cate8=True)
    actions.to_csv('./data/test_{}_{}.csv'.format(sub_start_date, sub_end_date), index=False, index_label=False)
    actions = None
    # 用于提交结果
    sub_start_date = '2016-03-17'
    sub_end_date = '2016-04-16'  # 不包含
    # 2016-04-16-2016-04-21
    actions = load_train_test_df(sub_start_date, sub_end_date, None,
                                 None, sample_neg=False, for_test=True, only_cate8=True)
    actions.to_csv('./data/test_{}_{}.csv'.format(sub_start_date, sub_end_date), index=False, index_label=False)
    actions = None
    print('done.')

if __name__ == '__main__':
    gen_all( )




