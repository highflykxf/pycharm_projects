import data
import pandas as pd
import numpy as np
from collections import Counter

dir = 'data/'

ACTION_201602_FILE = dir+"JData_Action_201602.csv"
ACTION_201603_FILE = dir+"JData_Action_201603.csv"
ACTION_201604_FILE = dir+"JData_Action_201604.csv"
COMMENT_FILE = dir+"JData_Comment.csv"
PRODUCT_FILE = dir+"JData_Product.csv"
USER_FILE = dir+"JData_User.csv"
USER_TABLE_FILE = dir+"user_table.csv"
PRODUCT_TABLE_FILE = dir+"item_table.csv"

# for_valid = True

def user_ids(FILE):
    df_usr = pd.read_csv(FILE, header=0)
    # df_usr = data.not_nan(df_usr)
    df_usr = df_usr[(df_usr['buy_num'] > 2)]
    df_usr = df_usr[['user_id','buy_num']]
    df_usr.to_csv(dir+'user_ids.csv')

def sku_ids(FILE):
    df_usr = pd.read_csv(FILE, header=0)
    df_usr = df_usr[(df_usr['buy_num'] > 0 )]
    df_usr = df_usr[['sku_id','buy_num']]
    df_usr.to_csv(dir+'sku_ids.csv')


def actives():
    user_ids(USER_TABLE_FILE)
    print('user ids done.')
    sku_ids(PRODUCT_TABLE_FILE)
    print('sku ids done.')

#得到验证数据集合，用于调试超参数
def gen_valid(fname = ACTION_201604_FILE, chunk_size = 100000):
    reader = pd.read_csv(fname, header=0, iterator=True)
    chunks = []
    loop = True
    while loop:
        try:
            # 只读取user_id和type两个字段
            chunk = reader.get_chunk(chunk_size)[['user_id','sku_id','time', "type"]]
            #使用五天的数据来得到验证集合
            chunk = chunk[(chunk['type']==4) & (chunk['time']<'2016-04-06 00:00:00')]
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")
    # 将块拼接为pandas dataframe格式
    df_ac = pd.concat(chunks, ignore_index=True)
    df_ac = df_ac.drop_duplicates(['user_id','sku_id'])
    df_ac.to_csv(dir+'valid.csv')
    # return df_ac

#用户够买过的商品
#购买，用户兴趣矩阵
#点击，类似的商品、可以组合购买的商品
#browse/click/addcart/addfavor/buy/delcart
def gen_matrix(fname = ACTION_201604_FILE, type = 4, chunk_size = 100000):
    reader = pd.read_csv(fname, header=0, iterator=True)
    chunks = []
    loop = True
    while loop:
        try:
            # 只读取user_id和type两个字段
            chunk = reader.get_chunk(chunk_size)[['user_id', 'sku_id', "type"]]
            chunk = chunk[(chunk['type'] == type)]
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")
    # 将块拼接为pandas dataframe格式
    df_ac = pd.concat(chunks, ignore_index=True)
    df_ac = df_ac.drop_duplicates(['user_id', 'sku_id'])
    return df_ac

def gen_matrixs( type = 4 ):
    df_ac = []
    df_ac.append( gen_matrix( fname=ACTION_201602_FILE , type=type) )
    df_ac.append( gen_matrix( fname=ACTION_201603_FILE , type=type) )
    # if not for_valid:
    #     df_ac.append( gen_matrix( fname=ACTION_201604_FILE , type=type) )
    df_ac = pd.concat(df_ac, ignore_index=True)
    df_ac = df_ac.drop_duplicates(['user_id', 'sku_id'])
    del df_ac['type']
    df_ac.to_csv(dir+'up_matrix_'+str(type)+'.csv')

#得到比较的序列
#产品排序结果
def gen_compare_pairs_all():

    click_browse_index = 1
    addcart_favor_index = 2
    #del不管了
    buy_index = 4
    with open(dir+'compare_pairs_buy.csv','w'):
        pass
    for sep in range(10):
        print('sep: {}'.format(sep))
        df_ac = []
        df_ac.append( gen_compare_pairs(ACTION_201602_FILE, sep) )
        df_ac.append( gen_compare_pairs(ACTION_201603_FILE, sep) )
        df_ac.append( gen_compare_pairs(ACTION_201604_FILE, sep) )
        df_ac = pd.concat(df_ac, ignore_index=True)
        #groupby以后会按照时间顺序排列吗？
        grouped = df_ac.groupby(['user_id'], as_index=False)
        pairs = []
        for name, group in grouped:
            #TODO 假设了每次只购买一个商品，实际上可能同时购买多个商品
            #key: action type, value: id set
            series = dict( )
            buy_set = set( )
            for index, row in group.iterrows():
                type = row['type']
                here_id = row['sku_id']
                if addcart_favor_index in series:
                    addcart_set = series[addcart_favor_index]
                else:
                    addcart_set = set( )
                if type == 4:#buy
                    buy_set.add( here_id )
                    #产生对比
                    if click_browse_index in series:
                        for cb_id in series[click_browse_index]:
                            if cb_id not in buy_set and ( cb_id not in addcart_set):
                                pairs.append( ( cb_id, here_id ) )
                    if addcart_favor_index in series:
                        for af_id in series[addcart_favor_index]:
                            if af_id not in buy_set:
                                pairs.append( (af_id, here_id ) )
                    # if (addcart_favor_index in series) and (click_browse_index in series ):
                    #     for cb_id in series[click_browse_index]:
                    #         for af_id in series[addcart_favor_index]:
                    #             if (cb_id != af_id) and (cb_id not in buy_set) and (af_id not in buy_set):
                    #                 pairs.append( (cb_id, af_id) )

                    #一次购买序列结束
                    series.clear( )
                elif type == 1 or type == 6:
                    here_id = row['sku_id']
                    if click_browse_index not in series:
                        series[click_browse_index] = set()
                    series[click_browse_index].add( here_id )
                elif type == 2 or type == 5:
                    here_id = row['sku_id']
                    if addcart_favor_index not in series:
                        series[addcart_favor_index] = set()
                    series[addcart_favor_index].add( here_id )

            # if (addcart_favor_index in series) and (click_browse_index in series):
            #     for cb_id in series[click_browse_index]:
            #         for af_id in series[addcart_favor_index]:
            #             if cb_id != af_id and (cb_id not in buy_set) and (af_id not in buy_set):
            #                 pairs.append((cb_id, af_id))
            series.clear()
        # df_ac = df_ac.sort_values(by=['user_id','time'])
        with open(dir+'compare_pairs_buy.csv', 'a',encoding='utf-8') as fout:
            print('save pairs: {}'.format(len(pairs) ) )
            for pair in pairs:
                fout.write('{}\t{}\n'.format( pair[0], pair[1] ) )

#click/browse < addcart/addfavor < buy
def gen_compare_pairs(fname, sep = 0, chunk_size = 100000):
    reader = pd.read_csv(fname, header=0, iterator=True)
    chunks = []
    loop = True
    while loop:
        try:
            # 只读取user_id和type两个字段
            chunk = reader.get_chunk(chunk_size)[['user_id', 'sku_id', 'time', "type"]]
            chunk = chunk[(chunk['user_id']%10 == sep)]
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")
    # 将块拼接为pandas dataframe格式
    df_ac = pd.concat(chunks, ignore_index=True)
    # df_ac = df_ac.drop_duplicates(['user_id', 'sku_id'])
    return df_ac


def gen_compare_pairs_all2():
    click_browse_index = 1
    addcart_favor_index = 2
    for sep in range(10):
        print('sep: {}'.format(sep))
        df_ac = []
        df_ac.append( gen_compare_pairs(ACTION_201602_FILE, sep) )
        df_ac.append( gen_compare_pairs(ACTION_201603_FILE, sep) )
        df_ac.append( gen_compare_pairs(ACTION_201604_FILE, sep) )
        df_ac = pd.concat(df_ac, ignore_index=True)
        df_ac = df_ac.sort_values(by=['user_id','time'])
        series = dict()
        pairs = []
        last_user_id = None
        for index, row in df_ac.iterrows():
            user_id = row['user_id']
            if last_user_id is not None and user_id != last_user_id:
                if (addcart_favor_index in series) and (click_browse_index in series):
                    for cb_id in series[click_browse_index]:
                        for af_id in series[addcart_favor_index]:
                            if cb_id != af_id:
                                pairs.append((cb_id, af_id))
                series.clear( )
            last_user_id = user_id
            type = row['type']
            here_id = row['sku_id']
            if type == 4:#buy
                #产生对比
                if click_browse_index in series:
                    for cb_id in series[click_browse_index]:
                        if cb_id != here_id:
                            pairs.append( ( cb_id, here_id ) )
                if addcart_favor_index in series:
                    for af_id in series[addcart_favor_index]:
                        if af_id != here_id:
                            pairs.append( (af_id, here_id ) )
                if (addcart_favor_index in series) and (click_browse_index in series ):
                    for cb_id in series[click_browse_index]:
                        for af_id in series[addcart_favor_index]:
                            if (cb_id != af_id) and (cb_id !=here_id) and (af_id !=here_id):
                                pairs.append( (cb_id, af_id) )

                #一次购买序列结束
                series.clear( )
            elif type == 1 or type == 6:
                here_id = row['sku_id']
                if click_browse_index not in series:
                    series[click_browse_index] = set()
                series[click_browse_index].add( here_id )
            elif type == 2 or type == 5:
                here_id = row['sku_id']
                if addcart_favor_index not in series:
                    series[addcart_favor_index] = set()
                series[addcart_favor_index].add( here_id )


        # df_ac = df_ac.sort_values(by=['user_id','time'])
        with open(dir+'compare_pairs.csv', 'wa',encoding='utf-8') as fout:
            print('save pairs: {}'.format(len(pairs) ) )
            for pair in pairs:
                fout.write('{}\t{}\n'.format( pair[0], pair[1] ) )
        # df_ac.to_csv(dir+'compares_{}.csv'.format(sep))


def gen_click_cart_all():
    for sep in range(10):
        print('sep: {}'.format(sep))
        df_ac = []
        df_ac.append( gen_compare_pairs(ACTION_201602_FILE, sep) )
        df_ac.append( gen_compare_pairs(ACTION_201603_FILE, sep) )
        df_ac.append( gen_compare_pairs(ACTION_201604_FILE, sep) )
        df_ac = pd.concat(df_ac, ignore_index=True)
        df_ac = df_ac.sort_values(by=['user_id','time'])
        df_ac.to_csv(dir+'compares_{}.csv'.format(sep))

#之前点击、查看、加入购物车、收藏，却没有够买的商品。
#用于预测
#TODO 如何确定一次选择、比较、购买的过程。相同cate?时间窗？
import datetime
def gen_click_cart():
    click_browse_index = 1
    addcart_favor_index = 2
    for sep in range(10):
        print('sep: {}'.format(sep))
        #key: (user_id, sku_id) value:[user_id, sku_id, click_browse_num,
        # addcart_favor_num, del_cart(0,1), firsttime, lasttime, delta_time, buy]
        #delta_time: 从最后一次动作到购买的时间
        action_dict = dict()
        buy_set = set()

        df_ac = []
        df_ac.append( gen_compare_pairs(ACTION_201602_FILE, sep) )
        df_ac.append( gen_compare_pairs(ACTION_201603_FILE, sep) )
        df_ac.append( gen_compare_pairs(ACTION_201604_FILE, sep) )
        df_ac = pd.concat(df_ac, ignore_index=True)
        df_ac = df_ac.sort_values(by=['user_id','time'])
        last_user_id = None
        for index, row in df_ac.iterrows():
            user_id = row['user_id']
            if last_user_id is not None and user_id != last_user_id:
                pass
            last_user_id = user_id
            type = row['type']
            here_id = row['sku_id']
            time = datetime.datetime.strptime(row['time'], '%Y-%m-%d %H:%M:%S')
            if (user_id, here_id) not in action_dict:
                action_dict[(user_id, here_id)] = [user_id, here_id, 0, 0, 0, time, time, 0, 0]
            if type == 4:#buy
                buy_set.add( (user_id, here_id))
                action_dict[(user_id, here_id)][8] = 1
                last_time = action_dict[(user_id,here_id)][6]
                action_dict[(user_id, here_id)][7] = (time-last_time).days
            else:
                action_dict[(user_id, here_id)][6] = time

            if type == 1 or type == 6:
                action_dict[(user_id, here_id)][2] += 1
            elif type == 2 or type == 5:
                action_dict[(user_id, here_id)][3] += 1
            elif type == 3:
                action_dict[(user_id, here_id)][4] += 1
        import pickle
        with open(dir+'action_dict_{}.pkl'.format(sep),'wb') as fout1,\
            open(dir+'buy_set_{}.pkl'.format(sep),'wb') as fout2:
            pickle.dump(action_dict, fout1)
            pickle.dump(buy_set, fout2)


import atribute_pairs
def products():
    product_cate = 'product_cate'
    product_brand = 'product_brand'
    triples = []
    actions = atribute_pairs.get_actions( )
    pro_dict = dict()
    cat8_pro = set()
    for index, row in actions.iterrows():
        sku_id = row['sku_id']
        if sku_id not in pro_dict:
            pro_dict[sku_id] = row['cate']
        if row['cate'] == 8:
            cat8_pro.add( sku_id )
        if sku_id == 75018:
            print(str(sku_id))
    print('数量:{}'.format( len(pro_dict ) ) )
    print('cate 8 数量： {}'.format( len(cat8_pro )) )
    product_set = set()
    import basic_feats
    product_feat = basic_feats.load_product_feat()
    for _, row in product_feat.iterrows():
        sku_id = row['sku_id']
        product_set.add( sku_id )
    num_miss = 0
    for sku_id in cat8_pro:
        if sku_id not in product_set:
            num_miss+=1
            print(str(sku_id))
    print('没有信息的产品数量：{}'.format(num_miss))
    #     pro_set.add( row['sku_id'])
    #     if row['cate'] != 8:
    #         sku_id = 'S_{}'.format(row['sku_id'])
    #         triples.append((sku_id, product_cate, 'cate_{}'.format(row['cate'])))
    #         triples.append((sku_id, product_brand, 'brand_{}'.format(row['brand'])))
    # with open(dir+'triples_extra.csv','w') as fout:
    #     for triple in triples:
    #         fout.write(','.join(triple))
    #         fout.write('\n')

if __name__ == '__main__':
    #购买量大于阈值的用户或者产品
    # actives( )
    #4.1~4.5的label
    # gen_valid(  )
    #产生各个action对应的共现矩阵，用于学习user、product representation
    # gen_matrixs( 1 )
    # gen_matrixs(2)
    # gen_matrixs(4)
    # gen_matrixs(5)
    # gen_matrixs(6)
    #用户的行为时间序列，按照时间顺序排列
    gen_click_cart_all( )
    #产生对比序列
    #使用set记录之前，6649个pairs in data_sample; 6529
    # gen_compare_pairs_all( )

    # gen_click_cart( )


    # products( )
    print('done.')

