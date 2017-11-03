

dir_origin = 'data/'
dir = 'data/'

ACTION_201602_FILE = dir+"JData_Action_201602.csv"
ACTION_201603_FILE = dir+"JData_Action_201603.csv"
ACTION_201604_FILE = dir+"JData_Action_201604.csv"
COMMENT_FILE = dir+"JData_Comment.csv"
PRODUCT_FILE = dir+"JData_Product.csv"
USER_FILE = dir+"JData_User.csv"

valid_mode = True
RECENT_FILE=ACTION_201604_FILE
if valid_mode:
    RECENT_FILE = ACTION_201603_FILE

USER_TABLE_FILE = dir+"user_table.csv"
PRODUCT_TABLE_FILE = dir+"item_table.csv"
import pandas as pd
import numpy as np
from collections import Counter


def add_type_count(group ):
    behavior_type = group.type.astype(int)
    # 用户行为类别
    type_cnt = Counter(behavior_type)
    # 1: 浏览 2: 加购 3: 删除
    # 4: 购买 5: 收藏 6: 点击
    group['browse_num'] = type_cnt[1]
    group['addcart_num'] = type_cnt[2]
    group['delcart_num'] = type_cnt[3]
    group['buy_num'] = type_cnt[4]
    group['favor_num'] = type_cnt[5]
    group['click_num'] = type_cnt[6]

    return group[["user_id", 'browse_num', 'addcart_num',
                  'delcart_num', 'buy_num', 'favor_num',
                  'click_num']]


def add_type_count_product(group ):
    behavior_type = group.type.astype(int)
    # 用户行为类别
    type_cnt = Counter(behavior_type)
    # 1: 浏览 2: 加购 3: 删除
    # 4: 购买 5: 收藏 6: 点击
    group['browse_num'] = type_cnt[1]
    group['addcart_num'] = type_cnt[2]
    group['delcart_num'] = type_cnt[3]
    group['buy_num'] = type_cnt[4]
    group['favor_num'] = type_cnt[5]
    group['click_num'] = type_cnt[6]

    return group[["sku_id", 'browse_num', 'addcart_num',
                  'delcart_num', 'buy_num', 'favor_num',
                  'click_num']]

#　对action数据进行统计
# 根据自己的需求调节chunk_size大小
def get_from_action_data(fname, group_id = 'user_id', chunk_size=100000):
    reader = pd.read_csv(fname, header=0, iterator=True)
    chunks = []
    loop = True
    while loop:
        try:
            # 只读取user_id和type两个字段
            chunk = reader.get_chunk(chunk_size)[[group_id, "type"]]
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")
    # 将块拼接为pandas dataframe格式
    df_ac = pd.concat(chunks, ignore_index=True)
    # 按user_id分组，对每一组进行统计
    if group_id=='user_id':
        df_ac = df_ac.groupby([group_id], as_index=False).apply(add_type_count)
    else:
        df_ac = df_ac.groupby([group_id], as_index=False).apply( add_type_count_product)
    # 将重复的行丢弃
    df_ac = df_ac.drop_duplicates(group_id)

    return df_ac


def merge_action_data():
    df_ac = []
    df_ac.append(get_from_action_data(fname=ACTION_201602_FILE))
    df_ac.append(get_from_action_data(fname=ACTION_201603_FILE))
    if not valid_mode:
        df_ac.append(get_from_action_data(fname=ACTION_201604_FILE))

    df_ac = pd.concat(df_ac, ignore_index=True)
    # 用户在不同action表中统计量求和
    df_ac = df_ac.groupby(['user_id'], as_index=False).sum()
    # 　构造转化率字段
    #TODO 这里的转化率计算很有问题，买的数量除以加入购物车的数量是错误的计算方式
    #TODO 可能加入购物车的是一个物品，但是买入的是另一个商品
    df_ac['buy_addcart_ratio'] = df_ac['buy_num'] / df_ac['addcart_num']
    df_ac['buy_browse_ratio'] = df_ac['buy_num'] / df_ac['browse_num']
    df_ac['buy_click_ratio'] = df_ac['buy_num'] / df_ac['click_num']
    df_ac['buy_favor_ratio'] = df_ac['buy_num'] / df_ac['favor_num']

    # 将大于１的转化率字段置为１(100%)
    df_ac.ix[df_ac['buy_addcart_ratio'] > 1., 'buy_addcart_ratio'] = 1.
    df_ac.ix[df_ac['buy_browse_ratio'] > 1., 'buy_browse_ratio'] = 1.
    df_ac.ix[df_ac['buy_click_ratio'] > 1., 'buy_click_ratio'] = 1.
    df_ac.ix[df_ac['buy_favor_ratio'] > 1., 'buy_favor_ratio'] = 1.

    return df_ac


def get_from_jdata_user():
    df_usr = pd.read_csv(USER_FILE, header=0)
    print(df_usr.head(0))
    # df_usr = df_usr[["user_id", "age", "sex", "user_lv_cd"]]
    #文件必须使用utf-8无BOM编码，不然会出错
    df_usr = df_usr[["user_id", "age", "sex", "user_lv_cd"]]
    return df_usr



def convert_user():
    user_base = get_from_jdata_user()
    user_behavior = merge_action_data()

    # 连接成一张表，类似于SQL的左连接(left join)
    user_behavior = pd.merge(user_base, user_behavior, on=['user_id'], how='left')
    # 保存为user_table.csv
    user_behavior.to_csv(USER_TABLE_FILE, index=False, encoding ='utf-8')


def not_nan(df):
    df = df[np.isfinite(df['buy_addcart_ratio'] ) & np.isfinite(df['buy_browse_ratio'])
            & np.isfinite(df['buy_click_ratio'])
            & np.isfinite(df['buy_favor_ratio'])]
    return df

def read(FILE):
    df_usr = pd.read_csv(FILE, header=0)
    df_usr = not_nan(df_usr)
    res = df_usr[df_usr['buy_num'] == 0].tail(5)
    print(res)
    # 输出前5行数据
    res = df_usr.head(5)
    print(res)
    df_usr = df_usr[df_usr['buy_num'] != 0]

    pd.options.display.float_format = '{:,.3f}'.format

    # 输出user table的统计信息,包括
    # 总数，均值，方差，最小值，1/4分位数，1/2分位数，3/4分位数，最大值
    res = df_usr.describe()
    print(res)
    res = df_usr[(df_usr['buy_num'] < 2) & (df_usr['browse_num'] > 6000)]
    print(res.head(5))
    df_usr.to_csv('D:/tmp_file.csv', index=False, encoding='utf-8')


def merge_action_data_product():
    df_ac = []
    df_ac.append(get_from_action_data( group_id = 'sku_id', fname=ACTION_201602_FILE))
    df_ac.append(get_from_action_data( group_id = 'sku_id', fname=ACTION_201603_FILE))
    if not valid_mode:
        df_ac.append(get_from_action_data( group_id = 'sku_id', fname=ACTION_201604_FILE))

    df_ac = pd.concat(df_ac, ignore_index=True)
    # 用户在不同action表中统计量求和
    df_ac = df_ac.groupby(['sku_id'], as_index=False).sum()
    # 　构造转化率字段
    df_ac['buy_addcart_ratio'] = df_ac['buy_num'] / df_ac['addcart_num']
    df_ac['buy_browse_ratio'] = df_ac['buy_num'] / df_ac['browse_num']
    df_ac['buy_click_ratio'] = df_ac['buy_num'] / df_ac['click_num']
    df_ac['buy_favor_ratio'] = df_ac['buy_num'] / df_ac['favor_num']

    # 将大于１的转化率字段置为１(100%)
    df_ac.ix[df_ac['buy_addcart_ratio'] > 1., 'buy_addcart_ratio'] = 1.
    df_ac.ix[df_ac['buy_browse_ratio'] > 1., 'buy_browse_ratio'] = 1.
    df_ac.ix[df_ac['buy_click_ratio'] > 1., 'buy_click_ratio'] = 1.
    df_ac.ix[df_ac['buy_favor_ratio'] > 1., 'buy_favor_ratio'] = 1.

    return df_ac


def get_from_sku():
    df_product = pd.read_csv(PRODUCT_FILE, header=0)
    print(df_product.head(0))
    # df_usr = df_usr[["user_id", "age", "sex", "user_lv_cd"]]
    #文件必须使用utf-8无BOM编码，不然会出错
    df_product = df_product[["sku_id","a1","a2","a3","cate","brand"]]
    return df_product

def convert_sku():
    product_base = get_from_sku()
    product_behavior = merge_action_data_product()

    # 连接成一张表，类似于SQL的左连接(left join)
    user_behavior = pd.merge(product_base, product_behavior, on=['sku_id'], how='left')
    # 保存为user_table.csv
    user_behavior.to_csv(PRODUCT_TABLE_FILE, index=False, encoding='utf-8')

if __name__ == '__main__':
    convert_user( )
    # read(USER_TABLE_FILE)
    convert_sku( )
    # read(PRODUCT_TABLE_FILE)