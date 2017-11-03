#_*_ coding=utf-8
'''
Created on 2017年5月5日

@author: wang
'''


#性别对产品，品牌，类别的购买的交叉特征
#每种产品购买行为中，男女分别有多少购买，字典的value是数组[female_count, male_count, unknown]
pur_product_to_sex = {}
#每种品牌购买行为中，男女分别有多少购买，字典的value是数组[female_count, male_count, unknown]
pur_brand_to_sex = {}
#每种类别购买行为中，男女分别有多少购买，字典的value是数组[female_count, male_count, unknown]
pur_cat_to_sex = {}

#性别对产品，品牌，类别的浏览的交叉特征
#每种产品购买行为中，男女分别有多少浏览，字典的value是数组[female_count, male_count, unknown]
view_product_to_sex = {}
#每种品牌购买行为中，男女分别有多少浏览，字典的value是数组[female_count, male_count, unknown]
view_brand_to_sex = {}
#每种类别购买行为中，男女分别有多少浏览，字典的value是数组[female_count, male_count, unknown]
view_cat_to_sex = {}

#性别对产品，品牌，类别的加入购物车的交叉特征
#每种产品购买行为中，男女分别有多少加入购物车，字典的value是数组[female_count, male_count, unknown]
add_product_to_sex = {}
#每种品牌购买行为中，男女分别有多少加入购物车，字典的value是数组[female_count, male_count, unknown]
add_brand_to_sex = {}
#每种类别购买行为中，男女分别有多少加入购物车，字典的value是数组[female_count, male_count, unknown]
add_cat_to_sex = {}

#性别对产品，品牌，类别的删除的交叉特征
#每种产品购买行为中，男女分别有多少删除，字典的value是数组[female_count, male_count, unknown]
del_product_to_sex = {}
#每种品牌购买行为中，男女分别有多少删除，字典的value是数组[female_count, male_count, unknown]
del_brand_to_sex = {}
#每种类别购买行为中，男女分别有多少删除，字典的value是数组[female_count, male_count, unknown]
del_cat_to_sex = {}

#性别对产品，品牌，类别的关注的交叉特征
#每种产品购买行为中，男女分别有多少关注，字典的value是数组[female_count, male_count, unknown]
follow_product_to_sex = {}
#每种品牌购买行为中，男女分别有多少关注，字典的value是数组[female_count, male_count, unknown]
follow_brand_to_sex = {}
#每种类别购买行为中，男女分别有多少关注，字典的value是数组[female_count, male_count, unknown]
follow_cat_to_sex = {}

#性别对产品，品牌，类别的点击的交叉特征
#每种产品购买行为中，男女分别有多少点击，字典的value是数组[female_count, male_count, unknown]
click_product_to_sex = {}
#每种品牌购买行为中，男女分别有多少点击，字典的value是数组[female_count, male_count, unknown]
click_brand_to_sex = {}
#每种类别购买行为中，男女分别有多少点击，字典的value是数组[female_count, male_count, unknown]
click_cat_to_sex = {}

import cPickle as pkl

def save_features():
    print('pur_product_to_sex count is '+str(len(pur_product_to_sex)))
    pkl.dump(pur_product_to_sex, open('./data/pur_product_to_sex.pkl', 'wb'), protocol=2)
    print('pur_brand_to_sex count is '+str(len(pur_brand_to_sex)))
    pkl.dump(pur_brand_to_sex, open('./data/pur_brand_to_sex.pkl', 'wb'), protocol=2)
    print('view_product_to_sex count is '+str(len(view_product_to_sex)))
    pkl.dump(view_product_to_sex, open('./data/view_product_to_sex.pkl', 'wb'), protocol=2)
    print('view_brand_to_sex count is '+str(len(view_brand_to_sex)))
    pkl.dump(view_brand_to_sex, open('./data/view_brand_to_sex.pkl', 'wb'), protocol=2)
    print('add_product_to_sex count is '+str(len(add_product_to_sex)))
    pkl.dump(add_product_to_sex, open('./data/add_product_to_sex.pkl', 'wb'), protocol=2)
    print('add_brand_to_sex count is '+str(len(add_brand_to_sex)))
    pkl.dump(add_brand_to_sex, open('./data/add_brand_to_sex.pkl', 'wb'), protocol=2)
    print('del_product_to_sex count is '+str(len(del_product_to_sex)))
    pkl.dump(del_product_to_sex, open('./data/del_product_to_sex.pkl', 'wb'), protocol=2)
    print('del_brand_to_sex count is '+str(len(del_brand_to_sex)))
    pkl.dump(del_brand_to_sex, open('./data/del_brand_to_sex.pkl', 'wb'), protocol=2)
    print('follow_product_to_sex count is '+str(len(follow_product_to_sex)))
    pkl.dump(follow_product_to_sex, open('./data/follow_product_to_sex.pkl', 'wb'), protocol=2)
    print('follow_brand_to_sex count is '+str(len(follow_brand_to_sex)))
    pkl.dump(follow_brand_to_sex, open('./data/follow_brand_to_sex.pkl', 'wb'), protocol=2)
    print('click_product_to_sex count is '+str(len(click_product_to_sex)))
    pkl.dump(click_product_to_sex, open('./data/click_product_to_sex.pkl', 'wb'), protocol=2)
    print('click_brand_to_sex count is '+str(len(click_brand_to_sex)))
    pkl.dump(click_brand_to_sex, open('./data/click_brand_to_sex.pkl', 'wb'), protocol=2)
    
def load_features():
    global pur_product_to_sex
    global pur_brand_to_sex
    global view_product_to_sex
    global view_brand_to_sex
    global add_product_to_sex
    global add_brand_to_sex
    global del_product_to_sex
    global del_brand_to_sex
    global follow_product_to_sex
    global follow_brand_to_sex
    global click_product_to_sex
    global click_brand_to_sex
    pur_product_to_sex = pkl.load(open('./data/pur_product_to_sex.pkl','rb'))
    pur_brand_to_sex = pkl.load(open('./data/pur_brand_to_sex.pkl','rb'))
    view_product_to_sex = pkl.load(open('./data/view_product_to_sex.pkl','rb'))
    view_brand_to_sex = pkl.load(open('./data/view_brand_to_sex.pkl','rb'))
    add_product_to_sex = pkl.load(open('./data/add_product_to_sex.pkl','rb'))
    add_brand_to_sex = pkl.load(open('./data/add_brand_to_sex.pkl','rb'))
    del_product_to_sex = pkl.load(open('./data/del_product_to_sex.pkl','rb'))
    del_brand_to_sex = pkl.load(open('./data/del_brand_to_sex.pkl','rb'))
    follow_product_to_sex = pkl.load(open('./data/follow_product_to_sex.pkl','rb'))
    follow_brand_to_sex = pkl.load(open('./data/follow_brand_to_sex.pkl','rb'))
    click_product_to_sex = pkl.load(open('./data/click_product_to_sex.pkl','rb'))
    click_brand_to_sex = pkl.load(open('./data/click_brand_to_sex.pkl','rb'))
    print('pur_product_to_sex count is '+str(len(pur_product_to_sex)))
    print('pur_brand_to_sex count is '+str(len(pur_brand_to_sex)))
    print('view_product_to_sex count is '+str(len(view_product_to_sex)))
    print('view_brand_to_sex count is '+str(len(view_brand_to_sex)))
    print('add_product_to_sex count is '+str(len(add_product_to_sex)))
    print('add_brand_to_sex count is '+str(len(add_brand_to_sex)))
    print('del_product_to_sex count is '+str(len(del_product_to_sex)))
    print('del_brand_to_sex count is '+str(len(del_brand_to_sex)))
    print('follow_product_to_sex count is '+str(len(follow_product_to_sex)))
    print('follow_brand_to_sex count is '+str(len(follow_brand_to_sex)))
    print('click_product_to_sex count is '+str(len(click_product_to_sex)))
    print('click_brand_to_sex count is '+str(len(click_brand_to_sex)))

if __name__ == '__main__':
    click_cat_to_sex['1'] = [1,2,3]
    click_cat_to_sex['2'] = [1,2,3]
    print len(click_cat_to_sex)
    pass