from sklearn.model_selection import train_test_split
# import xgboost as xgb
import time
import sys
from datetime import datetime
from datetime import timedelta
import math
import pandas as pd
import pickle
import os
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import preprocessing
import basic_feats
import extra_feat


def report(pred, label):
    actions = label
    result = pred
    # 所有用户商品对
    all_user_item_pair = actions['user_id'].map(str) + '-' + actions['sku_id'].map(str)
    all_user_item_pair = np.array(all_user_item_pair)
    user_item_pairs = set( )
    for pair in all_user_item_pair:
        user_item_pairs.add( pair )
        # print(pair)
    # 所有购买用户
    all_user_set = actions['user_id'].unique()
    # 所有品类中预测购买的用户
    all_user_test_set = result['user_id'].unique()
    all_user_test_item_pair = result['user_id'].map(str) + '-' + result['sku_id'].map(str)
    all_user_test_item_pair = np.array(all_user_test_item_pair)

    # 计算所有用户购买评价指标
    pos, neg = 0,0
    for user_id in all_user_test_set:
        if user_id in all_user_set:
            pos += 1
        else:
            neg += 1
    all_user_acc = 1.0 * pos / ( pos + neg+1e-12)
    all_user_recall = 1.0 * pos / len(all_user_set)
    print('预测数量:{}, 实际数量：{}'.format( pred.shape[0], label.shape[0] ))
    print('所有用户中预测购买用户的准确率为 ' + str(all_user_acc))
    print( '所有用户中预测购买用户的召回率' + str(all_user_recall) )

    pos, neg = 0, 0
    for user_item_pair in all_user_test_item_pair:
        if user_item_pair in user_item_pairs:
            pos += 1
        else:
            neg += 1
    all_item_acc = 1.0 * pos / ( pos + neg+1e-12)
    all_item_recall = 1.0 * pos / len(user_item_pairs)
    print('所有用户中预测购买商品的准确率为 ' + str(all_item_acc))
    print( '所有用户中预测购买商品的召回率' + str(all_item_recall))
    F11 = 6.0 * all_user_recall * all_user_acc / (5.0 * all_user_recall + all_user_acc+1e-12)
    F12 = 5.0 * all_item_acc * all_item_recall / (2.0 * all_item_recall + 3 * all_item_acc+1e-12)
    score = 0.4 * F11 + 0.6 * F12
    print( 'F11=' + str(F11))
    print( 'F12=' + str(F12) )
    print( 'score=' + str(score))
    return score


def get_dates( all_data =False):
    train_start_dates = []
    train_end_dates = []
    test_start_dates = []
    test_end_dates = []
    start = '2016-02-01'
    end = '2016-03-01'
    if all_data:
        end = '2016-03-05'
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
        start_days += timedelta(days=1)
    return train_start_dates, train_end_dates, test_start_dates, test_end_dates


class Model(object):
    def __init__(self):
        pass

    def train(self, X_train, y_train, X_test, y_test, weight_train, weight_test):
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=weight_train)
        dtest = xgb.DMatrix(X_test, label=y_test, weight=weight_test)
        # eta: 0.03 ~ 0.5, 与learning_rate作用相同
        param = { 'learning_rate': 0.01,'n_estimators': 1000, 'max_depth': 5,
                 'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8,
                 'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
        num_round = 500
        param['nthread'] = 4
        # param['eval_metric'] = 'auc'
        plst = []
        plst.extend(param.items())
        plst.append(('eval_metric', 'logloss'))
        # plst += [('eval_metric', 'logloss')]
        evallist = [(dtest, 'eval'), (dtrain, 'train')]

        print('start fit.')
        self.bst = xgb.train(plst, dtrain, num_round, evallist, verbose_eval=30)

    def predict(self, sub_trainning_data ):
        sub_trainning_data = xgb.DMatrix(sub_trainning_data)
        y = self.bst.predict(sub_trainning_data)
        return y

    def save(self, fname ):
        pickle.dump(self.bst, open(fname, 'wb'))


class ModelRF(object):
    def __init__(self):
        pass

    def train(self, X_train, y_train, X_test, y_test ):
        print('start fit.')
        self.bst = RandomForestClassifier(n_estimators=int(500), n_jobs=4, verbose=True,
                                          min_samples_leaf=50, max_depth=5)
        self.bst.fit(X_train, y_train)

    def predict(self, sub_trainning_data ):
        y = self.bst.predict(sub_trainning_data)
        return y

    def save(self, fname ):
        pickle.dump(self.bst, open(fname, 'wb'))



def rule_predict( ):
    sub_start_date = '2016-03-12'
    sub_end_date = '2016-04-11'
    sub_test_start_date = '2016-04-11'
    sub_test_end_date = '2016-04-16'
    # user_index, training_data, label, weights = extra_feat.get_basic_feats(valid=True)
    sub_user_index, sub_trainning_data, _ = extra_feat.load_basic_train_test(sub_start_date, sub_end_date,
                                                                                 valid=True)
    y_true = basic_feats.get_labels(sub_test_start_date, sub_test_end_date, only_cate8=True)
    half_num = int(y_true.shape[0] / 2)
    y_true = y_true.sample(n=half_num)
    y_true['user_id'] = y_true['user_id'].astype(int)
    y_true = y_true.drop_duplicates(['user_id', 'sku_id'])

    #加入购物车，且没有删除购物车，没有购买过
    # idx = ( (sub_trainning_data['30-action_2'] > 0) | (sub_trainning_data['30-action_5'] > 0 ) |
    #         (sub_trainning_data['30-action_1'] > 30 ))& \
    #       (sub_trainning_data['30-action_3'] <= 0) & \
    #       (sub_trainning_data['30-action_4'] <= 0)
    idx = ((sub_trainning_data['3-action_2'] > 0) | (sub_trainning_data['3-action_5'] > 0) ) & \
          (sub_trainning_data['3-action_3'] <= 0) &\
          (sub_trainning_data['3-action_4'] <= 0)
    sub_trainning_data['label'] = 0.0
    sub_trainning_data.loc[idx,'label'] = 1
    # sub_trainning_data = sub_trainning_data.fillna(0.0)
    y = sub_trainning_data['label'].copy()
    pred = sub_user_index.copy()
    pred['label'] = y
    pred = pred.drop_duplicates(['user_id','sku_id'])
    pred = pred[pred['label']>0]
    idx = pred.groupby(['user_id'])['label'].transform(max) == pred['label']
    pred = pred[idx]
    pred = pred.sort_values(['label'], ascending=False)
    pred = pred[['user_id', 'sku_id']]
    # 只使用了first, 没有考虑哪一个数值更大
    # pred = pred.groupby('user_id').first().reset_index( )
    pred['user_id'] = pred['user_id'].astype(int)
    pred['sku_id'] = pred['sku_id'].astype(int)
    report(pred, y_true)
    print('****************')


def performance( model, scaler ):
    sub_start_date = '2016-03-12'
    sub_end_date = '2016-04-11'
    sub_test_start_date = '2016-04-11'
    sub_test_end_date = '2016-04-16'
    sub_user_index, sub_trainning_data, _ = extra_feat.load_extra_train_test(sub_start_date, sub_end_date, valid=True)
    sub_trainning_data = scaler.transform(sub_trainning_data)
    y = model.predict(sub_trainning_data)  # .values
    y_true = basic_feats.get_labels(sub_test_start_date, sub_test_end_date, only_cate8=True)
    y_true['user_id'] = y_true['user_id'].astype(int)
    y_true = y_true.drop_duplicates(['user_id', 'sku_id'])
    half_num = int(y_true.shape[0] / 2)
    y_true = y_true.sample(n=half_num)
    # for thres in (1-1e-6, 1-1e-3,1-0.01,1-0.05, 1-0.1):#0.0,
    # for thres in ( 0.03,0.5,0.1,0.3,0.5,0.7,0.9):
    best_score = 0.0
    best_top_num = 1000
    for top_num in ( 1000, 1200, 1500, 2000, 3000):
        pred = sub_user_index.copy( )
        pred['label'] = y
        pred = pred.drop_duplicates(['user_id', 'sku_id'])
        # pred = pred[pred['label'] >= thres]
        #TODO 这里有错误。选取概率最大的那个sku，但是没有效果，不管取什么thres，都是一样的score
        idx = pred.groupby(['user_id'])['label'].transform(max) == pred['label']
        pred = pred[idx]
        pred = pred.sort_values(['label'], ascending=False)
        pred = pred.head(top_num)
        print(pred.head(10))
        pred = pred[['user_id', 'sku_id']]
        # 只使用了first, 没有考虑哪一个数值更大
        # pred = pred.groupby('user_id').first().reset_index( )
        pred['user_id'] = pred['user_id'].astype(int)
        pred['sku_id'] = pred['sku_id'].astype(int)
        score = report(pred, y_true)
        if score > best_score:
            best_score = score
            best_top_num = top_num
        # xgb.plot_importance(bst)
        print('****************')
    print('best top num: {}, best score: {}'.format(best_top_num, best_score) )
    return best_top_num

def get_weight(label):
    if label == 1.0:
        return 10.0
    return 0.01


def submission():
    sub_start_date = '2016-03-17'
    sub_end_date = '2016-04-16'  # 不包含
    sub_test_start_date = '2016-04-11'
    sub_test_end_date = '2016-04-16'
    user_index, training_data, label, weights = extra_feat.get_extra_feats()
    print('data shape: ')
    print(training_data.shape)
    print('start fit.')
    sub_user_index, sub_trainning_data, _= extra_feat.load_extra_train_test(sub_start_date, sub_end_date )
    scaler = preprocessing.StandardScaler().fit(training_data)
    training_data = scaler.transform(training_data)
    sub_trainning_data = scaler.transform(sub_trainning_data)
    # weight = label['label'].map(get_weight)
    X_train, X_test, y_train, y_test,weight_train,weight_test = \
        train_test_split(training_data, label.values,weights.values, test_size=0.2,
                                                        random_state=0)
    model = Model()
    print('data shape: ')
    print(training_data.shape)
    model.train( X_train, y_train, X_test, y_test, weight_train, weight_test)
    best_top_num = performance(model, scaler )
    y = model.predict( sub_trainning_data )#.values
    sub_user_index['label'] = y
    # pred = sub_user_index[sub_user_index['label'] >= 0.5]
    # pred = pred[['user_id', 'sku_id']]
    # #只使用了first, 没有考虑哪一个数值更大
    # pred = pred.groupby('user_id').first().reset_index()
    # pred['user_id'] = pred['user_id'].astype(int)
    pred = sub_user_index.copy()
    pred['label'] = y
    pred = pred.drop_duplicates(['user_id', 'sku_id'])
    idx = pred.groupby(['user_id'])['label'].transform(max) == pred['label']
    pred = pred[idx]
    pred = pred.sort_values(['label'], ascending=False)
    pred = pred.head(best_top_num)
    # print(pred.head(10))
    pred = pred[['user_id', 'sku_id']]
    # 只使用了first, 没有考虑哪一个数值更大
    # pred = pred.groupby('user_id').first().reset_index( )
    pred['user_id'] = pred['user_id'].astype(int)
    pred['sku_id'] = pred['sku_id'].astype(int)
    print( 'submission number: {}'.format(pred.shape[0]) )
    pred.to_csv('./submission.csv', index=False, index_label=False)



def tune_xgb( use_basic = False ):
    sub_start_date = '2016-03-12'
    sub_end_date = '2016-04-11'
    sub_test_start_date = '2016-04-11'
    sub_test_end_date = '2016-04-16'
    if use_basic:
        user_index, training_data, label, weights = extra_feat.get_basic_feats(valid=True)
        sub_user_index, sub_trainning_data, _ = extra_feat.load_basic_train_test(sub_start_date, sub_end_date,
                                                                                 valid=True)
    else:
        user_index, training_data, label, weights = extra_feat.get_extra_feats(valid = True)
        sub_user_index, sub_trainning_data, _ = extra_feat.load_extra_train_test(sub_start_date, sub_end_date, valid=True)
    scaler = preprocessing.StandardScaler().fit( training_data )
    training_data = scaler.transform( training_data )
    sub_trainning_data = scaler.transform( sub_trainning_data )
    # X_train, X_test, y_train, y_test = train_test_split(training_data, label.values, test_size=0.2,
    #                                                     random_state=0)
    print(label.head(10))
    # weight = label['label'].map(get_weight)
    X_train, X_test, y_train, y_test, weight_train, weight_test = \
        train_test_split(training_data, label.values, weights.values, test_size=0.2,
                         random_state=0)
    print('data shape: ')
    print(training_data.shape)
    model = Model()
    model.train( X_train, y_train, X_test, y_test, weight_train, weight_test)
    model.save( 'cache/model_user_sku.pkl' )
    pickle.dump( scaler, open('cache/model_scaler.pkl','wb') )
    y = model.predict(sub_trainning_data)#.values
    y_true = basic_feats.get_labels(sub_test_start_date, sub_test_end_date,only_cate8=True)
    half_num = int(y_true.shape[0]/2)
    y_true = y_true.sample(n=half_num)
    y_true['user_id'] = y_true['user_id'].astype(int)
    y_true = y_true.drop_duplicates(['user_id', 'sku_id'])
    # for thres in (1-1e-6, 1-1e-3,1-0.01,1-0.05, 1-0.1):#0.0,
    # for thres in ( 0.03,0.5,0.1,0.3,0.5,0.7,0.9):
    # thres = 0.5
    # print('thres : {}'.format(thres) )
    for top_num in (800, 1000, 1200, 1500, 2000):
        pred = sub_user_index.copy( )
        pred['label'] = y
        pred = pred.drop_duplicates(['user_id', 'sku_id'])
        # pred = pred[pred['label'] >= thres]
        #TODO 这里有错误。选取概率最大的那个sku，但是没有效果，不管取什么thres，都是一样的score
        idx = pred.groupby(['user_id'])['label'].transform(max) == pred['label']
        pred = pred[idx]
        pred = pred.sort_values(['label'], ascending=False)
        pred = pred.head(top_num)
        # print(pred.head(10))
        pred = pred[['user_id', 'sku_id']]
        # 只使用了first, 没有考虑哪一个数值更大
        # pred = pred.groupby('user_id').first().reset_index( )
        pred['user_id'] = pred['user_id'].astype(int)
        pred['sku_id'] = pred['sku_id'].astype(int)
        report(pred, y_true)
        # xgb.plot_importance(bst)
        print('****************')


if __name__ == '__main__':
    # tune( )
    if (len(sys.argv)) > 1:
        if sys.argv[1] == 'sub':
            print('submission')
            submission()
        elif sys.argv[1] == 'basic':
            print('tuning, use basic feats.')
            tune_xgb(use_basic=True)
        elif sys.argv[1] == 'rule':
            print('tuning, rule model.')
            rule_predict( )
        else:
            print('unknown arg. {}'.format(sys.argv[1]))
    else:
        print('tuning')
        tune_xgb( use_basic=False )
    # submission( )