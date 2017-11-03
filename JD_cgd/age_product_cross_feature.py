#_*_ coding=utf-8
'''
Created on 2017年5月5日

@author: wang
'''



#年龄对产品，品牌，类别的购买的交叉特征
#每种产品购买行为中，各个年龄段分别有多少购买，字典的value是数组[age1-0, age2-1, age3-2, age4-4, age5-5, age6-6]
pur_product_to_age = {}
#每种产品购买行为中，各个年龄段分别有多少购买，字典的value是数组[age1-0, age2-1, age3-2, age4-4, age5-5, age6-6]
pur_brand_to_age = {}
#每种产品购买行为中，各个年龄段分别有多少购买，字典的value是数组[age1-0, age2-1, age3-2, age4-4, age5-5, age6-6]
pur_cat_to_age = {}

#年龄对产品，品牌，类别的浏览的交叉特征
#每种产品购买行为中，各个年龄段分别有多少购买，字典的value是数组[age1-0, age2-1, age3-2, age4-4, age5-5, age6-6]
view_product_to_age = {}
#每种产品购买行为中，各个年龄段分别有多少购买，字典的value是数组[age1-0, age2-1, age3-2, age4-4, age5-5, age6-6]
view_brand_to_age = {}
#每种产品购买行为中，各个年龄段分别有多少购买，字典的value是数组[age1-0, age2-1, age3-2, age4-4, age5-5, age6-6]
view_cat_to_age = {}

#年龄对产品，品牌，类别的加入购物车的交叉特征
#每种产品购买行为中，各个年龄段分别有多少购买，字典的value是数组[age1-0, age2-1, age3-2, age4-4, age5-5, age6-6]
add_product_to_age = {}
#每种产品购买行为中，各个年龄段分别有多少购买，字典的value是数组[age1-0, age2-1, age3-2, age4-4, age5-5, age6-6]
add_brand_to_age = {}
#每种产品购买行为中，各个年龄段分别有多少购买，字典的value是数组[age1-0, age2-1, age3-2, age4-4, age5-5, age6-6]
add_cat_to_age = {}

#年龄对产品，品牌，类别的删除的交叉特征
#每种产品购买行为中，各个年龄段分别有多少购买，字典的value是数组[age1-0, age2-1, age3-2, age4-4, age5-5, age6-6]
del_product_to_age = {}
#每种产品购买行为中，各个年龄段分别有多少购买，字典的value是数组[age1-0, age2-1, age3-2, age4-4, age5-5, age6-6]
del_brand_to_age = {}
#每种产品购买行为中，各个年龄段分别有多少购买，字典的value是数组[age1-0, age2-1, age3-2, age4-4, age5-5, age6-6]
del_cat_to_age = {}

#年龄对产品，品牌，类别的关注的交叉特征
#每种产品购买行为中，各个年龄段分别有多少购买，字典的value是数组[age1-0, age2-1, age3-2, age4-4, age5-5, age6-6]
follow_product_to_age = {}
#每种产品购买行为中，各个年龄段分别有多少购买，字典的value是数组[age1-0, age2-1, age3-2, age4-4, age5-5, age6-6]
follow_brand_to_age = {}
#每种产品购买行为中，各个年龄段分别有多少购买，字典的value是数组[age1-0, age2-1, age3-2, age4-4, age5-5, age6-6]
follow_cat_to_age = {}

#年龄对产品，品牌，类别的点击的交叉特征
#每种产品购买行为中，各个年龄段分别有多少购买，字典的value是数组[age1-0, age2-1, age3-2, age4-4, age5-5, age6-6]
click_product_to_age = {}
#每种产品购买行为中，各个年龄段分别有多少购买，字典的value是数组[age1-0, age2-1, age3-2, age4-4, age5-5, age6-6]
click_brand_to_age = {}
#每种产品购买行为中，各个年龄段分别有多少购买，字典的value是数组[age1-0, age2-1, age3-2, age4-4, age5-5, age6-6]
click_cat_to_age = {}

import pickle as pkl

def save_features():
    print('pur_product_to_age count is '+str(len(pur_product_to_age)))
    pkl.dump(pur_product_to_age, open('./data/pur_product_to_age.pkl', 'wb'), protocol=2)
    print('pur_brand_to_age count is '+str(len(pur_brand_to_age)))
    pkl.dump(pur_brand_to_age, open('./data/pur_brand_to_age.pkl', 'wb'), protocol=2)
    print('view_product_to_age count is '+str(len(view_product_to_age)))
    pkl.dump(view_product_to_age, open('./data/view_product_to_age.pkl', 'wb'), protocol=2)
    print('view_brand_to_age count is '+str(len(view_brand_to_age)))
    pkl.dump(view_brand_to_age, open('./data/view_brand_to_age.pkl', 'wb'), protocol=2)
    print('add_product_to_age count is '+str(len(add_product_to_age)))
    pkl.dump(add_product_to_age, open('./data/add_product_to_age.pkl', 'wb'), protocol=2)
    print('add_brand_to_age count is '+str(len(add_brand_to_age)))
    pkl.dump(add_brand_to_age, open('./data/add_brand_to_age.pkl', 'wb'), protocol=2)
    print('del_product_to_age count is '+str(len(del_product_to_age)))
    pkl.dump(del_product_to_age, open('./data/del_product_to_age.pkl', 'wb'), protocol=2)
    print('del_brand_to_age count is '+str(len(del_brand_to_age)))
    pkl.dump(del_brand_to_age, open('./data/del_brand_to_age.pkl', 'wb'), protocol=2)
    print('follow_product_to_age count is '+str(len(follow_product_to_age)))
    pkl.dump(follow_product_to_age, open('./data/follow_product_to_age.pkl', 'wb'), protocol=2)
    print('follow_brand_to_age count is '+str(len(follow_brand_to_age)))
    pkl.dump(follow_brand_to_age, open('./data/follow_brand_to_age.pkl', 'wb'), protocol=2)
    print('click_product_to_age count is '+str(len(click_product_to_age)))
    pkl.dump(click_product_to_age, open('./data/click_product_to_age.pkl', 'wb'), protocol=2)
    print('click_brand_to_age count is '+str(len(click_brand_to_age)))
    pkl.dump(click_brand_to_age, open('./data/click_brand_to_age.pkl', 'wb'), protocol=2)
    
def load_features():
    global pur_product_to_age
    global pur_brand_to_age
    global view_product_to_age
    global view_brand_to_age
    global add_product_to_age
    global add_brand_to_age
    global del_product_to_age
    global del_brand_to_age
    global follow_product_to_age
    global follow_brand_to_age
    global click_product_to_age
    global click_brand_to_age
    pur_product_to_age = pkl.load(open('./data/pur_product_to_age.pkl','rb'))
    pur_brand_to_age = pkl.load(open('./data/pur_brand_to_age.pkl','rb'))
    view_product_to_age = pkl.load(open('./data/view_product_to_age.pkl','rb'))
    view_brand_to_age = pkl.load(open('./data/view_brand_to_age.pkl','rb'))
    add_product_to_age = pkl.load(open('./data/add_product_to_age.pkl','rb'))
    add_brand_to_age = pkl.load(open('./data/add_brand_to_age.pkl','rb'))
    del_product_to_age = pkl.load(open('./data/del_product_to_age.pkl','rb'))
    del_brand_to_age = pkl.load(open('./data/del_brand_to_age.pkl','rb'))
    follow_product_to_age = pkl.load(open('./data/follow_product_to_age.pkl','rb'))
    follow_brand_to_age = pkl.load(open('./data/follow_brand_to_age.pkl','rb'))
    click_product_to_age = pkl.load(open('./data/click_product_to_age.pkl','rb'))
    click_brand_to_age = pkl.load(open('./data/click_brand_to_age.pkl','rb'))
    print('pur_product_to_age count is '+str(len(pur_product_to_age)))
    print('pur_brand_to_age count is '+str(len(pur_brand_to_age)))
    print('view_product_to_age count is '+str(len(view_product_to_age)))
    print('view_brand_to_age count is '+str(len(view_brand_to_age)))
    print('add_product_to_age count is '+str(len(add_product_to_age)))
    print('add_brand_to_age count is '+str(len(add_brand_to_age)))
    print('del_product_to_age count is '+str(len(del_product_to_age)))
    print('del_brand_to_age count is '+str(len(del_brand_to_age)))
    print('follow_product_to_age count is '+str(len(follow_product_to_age)))
    print('follow_brand_to_age count is '+str(len(follow_brand_to_age)))
    print('click_product_to_age count is '+str(len(click_product_to_age)))
    print('click_brand_to_age count is '+str(len(click_brand_to_age)))


if __name__ == '__main__':
    pass