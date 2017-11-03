import os
import pickle
import pandas as pd
import numpy as np
import basic_feats
dir = 'data/'


def get_weight(label):
    if label == 1.0:
        return 10.0
    return 0.01

def load_predata(dump_path):
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        print('error {} not found.'.format(dump_path))
        actions = None
    return actions

def load_vector( fname ):
    p_vector = pd.read_csv( fname )
    return p_vector


def convert_sims():
    seps = ('1', '2', '4', '5', '6')
    sku_fns = [( 'tsku_{}.mat'.format(sep)) for sep in seps]
    # sku_fns.append( dir+'trans_sku.mat')
    user_fns = [( 'tuser_{}.mat'.format(sep)) for sep in seps]
    fname = './data/basic_feats.csv'
    actions = pd.read_csv(fname)
    actions = actions[['user_id','sku_id']]

    test_one = pd.read_csv('./data/new_test_2016-03-12_2016-04-11.csv')
    test_one = test_one[['user_id','sku_id']]
    test_two = pd.read_csv('./data/new_test_2016-03-17_2016-04-16.csv')
    test_two = test_two[['user_id','sku_id']]

    for sku_fn, user_fn in zip(sku_fns, user_fns):
        sku_vec = load_vector(dir+sku_fn)
        sku_ids = sku_vec['sku_id'].copy()
        del sku_vec['sku_id']
        sku_id_index = dict()
        for index, sku_id in enumerate(sku_ids):
            sku_id_index[sku_id] = index
        sku_vec = sku_vec.values

        user_vec = load_vector(dir+user_fn)
        user_ids = user_vec['user_id'].copy()
        del user_vec['user_id']
        user_id_index = dict()
        for index, user_id in enumerate(user_ids):
            user_id_index[ user_id ] = index
        user_vec = user_vec.values
        # mat = np.zeros( (user_vec.shape[0], sku_vec.shape[0]) )
        # for i in range(user_vec.shape[0]):
        #     for j in range( sku_vec.shape[0]):
        #         mat[i,j] = np.dot(user_vec[i], sku_vec[j])
        with open(dir+'sim_us_{}.csv'.format(sku_fn[0:-4]), 'w') as fout:
            fout.write('user_id,sku_id,sim_{}\n'.format(sku_fn[0:-4]))
            for df in (actions, test_one, test_two):
                for index, row in df.iterrows():
                    user_id = row['user_id']
                    sku_id = row['sku_id']
                    if (sku_id in sku_id_index) and (user_id in user_id_index):
                        sim = np.dot( user_vec[user_id_index[user_id]], sku_vec[sku_id_index[sku_id]])
                        fout.write('{:g},{:g},{}\n'.format(user_id, sku_id, sim))
        print('{} done.'.format(sku_fn[0:-4]))

def add_vectors( actions ):
    seps = ('1','2','4','5','6')
    sku_fns = [('tsku_{}.mat'.format( sep ) ) for sep in seps]
    # sku_fns.append( dir+'trans_sku.mat')
    user_fns = [('tuser_{}.mat'.format( sep )) for sep in seps]
    # user_fns.append( dir+'trans_user.mat')
    for sku_fn, user_fn in zip(sku_fns, user_fns):
        # fname = dir + 'tsku_{}.mat'.format( sep )
        sku_vec = load_vector( dir+sku_fn )
        names = ['sku_id']
        for i in range(sku_vec.shape[1] - 1):
            names.append('sku_vec_{}_{}'.format(sku_fn[0:-4], i))
        sku_vec.columns = names
        # fname = dir+ 'tuser_{}.mat'.format( sep )
        #将sep加入特征名字中
        user_vec = load_vector(dir+user_fn)
        names = ['user_id']
        for i in range( user_vec.shape[1]-1):
            names.append( 'user_vec_{}_{}'.format(user_fn[0:-4], i))
        user_vec.columns = names
        actions = pd.merge(actions, user_vec, how='left', on=['user_id'])
        actions = pd.merge(actions, sku_vec, how='left', on=['sku_id'])
    return actions

def add_vectors_sim( actions ):
    seps = ('1','2','4','5','6')
    sim_fns = [('sim_us_tsku_{}.csv'.format( sep ) ) for sep in seps]
    for sim_fn in sim_fns:
        sim_vec = load_vector( dir+sim_fn )
        sim_vec = sim_vec.drop_duplicates( ['user_id','sku_id'] )
        actions = pd.merge(actions, sim_vec, how='left', on=['user_id','sku_id'])
    return actions

def get_basic_feats( valid = False):
    fname = './data/basic_feats.csv'
    actions = pd.read_csv(fname)
    if valid:
        actions = actions[actions['start_date']<'2016-03-06']
    actions = actions.replace([np.inf, -np.inf], np.nan)
    actions = actions.fillna(0.)
    users = actions[['user_id', 'sku_id']].copy()
    actions['weight'] = actions['label'].map(get_weight)
    weights = actions['weight'].copy()
    del actions['weight']
    labels = actions['label'].copy()
    del actions['label']
    del actions['user_id']
    del actions['sku_id']
    del actions['start_date']
    del actions['end_date']
    del actions['cate']
    return users, actions, labels, weights

def load_basic_train_test(sub_start_date, sub_end_date, valid=False ):
    fname = './data/test_{}_{}.csv'.format(sub_start_date, sub_end_date)
    actions = pd.read_csv( fname )
    actions = actions.replace([np.inf, -np.inf], np.nan)
    actions = actions.fillna(0.)
    users = actions[['user_id', 'sku_id']].copy()
    del actions['user_id']
    del actions['sku_id']
    del actions['start_date']
    del actions['end_date']
    del actions['cate']
    return users, actions, None

def get_extra_feats( valid = False):
    # fname = './wang/new_basic_feats_wang.csv'
    # actions = pd.read_csv(fname)
    # if valid:
    #     actions = actions[actions['start_date'] < '2016-03-08']
    fname = './cache/train_extra_feats.csv'
    if valid:
        fname = './cache/train_extra_feats_valid.csv'
    actions = pd.read_csv(fname)
    actions = actions.replace([np.inf, -np.inf], np.nan)
    actions = actions.fillna(0.)
    users = actions[['user_id', 'sku_id']].copy()
    actions['weight'] = actions['label'].map( get_weight )
    weights = actions['weight'].copy()
    del actions['weight']
    labels = actions['label'].copy()
    del actions['label']
    del actions['user_id']
    del actions['sku_id']
    del actions['start_date']
    del actions['end_date']
    del actions['cate']
    return users, actions, labels, weights


def load_extra_train_test(sub_start_date, sub_end_date, valid=False ):
    # fname = './wang/new_test_2016-03-12_2016-04-11_wang.csv'
    # if valid:
    #     fname = './wang/new_test_2016-03-17_2016-04-16_wang.csv'

    fname = './cache/test_{}_{}_extra.csv'.format(sub_start_date, sub_end_date)
    if valid:
        fname = './cache/test_{}_{}_extra.csv'.format(sub_start_date, sub_end_date)
    actions = pd.read_csv( fname )
    actions = actions.replace([np.inf, -np.inf], np.nan)
    actions = actions.fillna(0.)
    users = actions[['user_id', 'sku_id']].copy()
    del actions['user_id']
    del actions['sku_id']
    del actions['start_date']
    del actions['end_date']
    del actions['cate']
    return users, actions, None


def add_upvec_to_basic():
    actions = pd.read_csv('./basic_feats.csv')
    # actions = actions[actions['start_date']>'2016-02-20']
    actions = add_vectors(actions)
    actions.to_csv('./cache/train_extra_feats.csv', index=False, index_label=False)
    actions = actions[actions['start_date'] < '2016-03-06']
    actions.to_csv('./cache/train_extra_feats_valid.csv', index=False, index_label=False)
    actions = None
    print('train set done.')

    # 处理测试、验证数据


    sub_start_date = '2016-03-12'
    sub_end_date = '2016-04-11'  # 不包含
    sub_test_start_date = '2016-04-11'
    sub_test_end_date = '2016-04-16'  # 不包含
    actions = pd.read_csv('./test_{}_{}.csv'.format(sub_start_date, sub_end_date))
    actions = add_vectors(actions)
    actions.to_csv('./cache/test_{}_{}_extra.csv'.format(sub_start_date, sub_end_date), index=False, index_label=False)
    sub_start_date = '2016-03-17'
    sub_end_date = '2016-04-16'  # 不包含
    sub_test_start_date = '2016-04-11'
    sub_test_end_date = '2016-04-16'
    actions = pd.read_csv('./test_{}_{}.csv'.format(sub_start_date, sub_end_date))
    actions = add_vectors(actions)
    actions.to_csv('./cache/test_{}_{}_extra.csv'.format(sub_start_date, sub_end_date), index=False, index_label=False)


def add_upvec_to_wang():
    actions = pd.read_csv('./data/new_basic_feats.csv')
    #过滤过年的时间没有什么大的影响。
    # actions = actions[actions['start_date']>'2016-02-20']
    actions = add_vectors(actions)
    actions = add_vectors_sim(actions)
    actions.to_csv('./cache/train_extra_feats.csv', index=False, index_label=False)
    actions = actions[actions['start_date'] < '2016-03-06']
    actions.to_csv('./cache/train_extra_feats_valid.csv', index=False, index_label=False)
    actions = None
    print('train set done.')
    # 处理测试、验证数据
    sub_start_date = '2016-03-12'
    sub_end_date = '2016-04-11'  # 不包含
    sub_test_start_date = '2016-04-11'
    sub_test_end_date = '2016-04-16'  # 不包含
    actions = pd.read_csv('./data/new_test_2016-03-12_2016-04-11.csv')
    actions = add_vectors(actions)
    actions = add_vectors_sim(actions)
    actions.to_csv('./cache/test_{}_{}_extra.csv'.format(sub_start_date, sub_end_date), index=False, index_label=False)
    sub_start_date = '2016-03-17'
    sub_end_date = '2016-04-16'  # 不包含
    sub_test_start_date = '2016-04-11'
    sub_test_end_date = '2016-04-16'
    actions = pd.read_csv('./data/new_test_2016-03-17_2016-04-16.csv')
    actions = add_vectors(actions)
    actions = add_vectors_sim(actions)
    actions.to_csv('./cache/test_{}_{}_extra.csv'.format(sub_start_date, sub_end_date), index=False, index_label=False)


if __name__ == '__main__':
    # dump_paths = ('./cache/train_set_all.pkl','./cache/train_set_valid.pkl')
    # basic_paths = ('./basic_feats.csv','basic_feats_valid.csv')
    # save_paths = ('./feats_all.csv','feats_valid.csv')
    # for dump_path,save_path in (dump_paths, save_paths):
    #     actions = load_predata(  dump_path )
    #     actions.to_csv('./basic_feats.csv', index=False, index_label=False)
    #处理训练数据
    # add_upvec_to_basic( )
    #利用向量表示计算用户、产品相似度，每次改变候选集的时候需要重新计算
    convert_sims( )
    add_upvec_to_wang( )
    print('done.')