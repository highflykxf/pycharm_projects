#_*_ coding=utf-8
'''
Created on 2017年5月5日
@author: wang
'''

import numpy

import pandas as pd


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



'''
读取用户信息，年龄和性别，都转成字符串型
'''
def load_users_info():
    #保存用户信息，value为2维数组[age, sex]
    users_info = {}
    users = pd.read_csv('./data/JData_User.csv', dtype = {'user_id':numpy.str, 'sex':numpy.str, 'age':numpy.str}, encoding='utf-8')
    users_id = users['user_id']
    users_sex = users['sex']
    users_age = users['age']
    for i in range(len(users_id)):
        user_info = []
        user_info.append(str(convert_age(unicode(users_age[i]))))
        if(users_sex[i]!='1' and users_sex[i]!='0' and users_sex[i]!='2'):
            print users_id[i],users_sex[i],users_age[i]
            users_sex[i] = '2'
            users_age[i] = '-1'
        user_info.append(users_sex[i])
        users_info[users_id[i]] = user_info
    return users_info


'''
读取产品信息，产品类别和品牌都是字符串型
'''
def load_pros_info():
    #保存产品信息,value为2维数组[cate,brand]
    pros_info = {}
    pros = pd.read_csv('./data/JData_Product.csv', dtype = {'sku_id':numpy.str, 'cate':numpy.str, 'brand':numpy.str}, encoding='utf-8')
    pros_id = pros['sku_id']
    pros_cate = pros['cate']
    pros_brand = pros['brand']
    for i in range(len(pros_id)):
        pro_info = []
        pro_info.append(pros_cate[i])
        pro_info.append(pros_brand[i])
        pros_info[pros_id[i]] = pro_info
    return pros_info

# str1 = u'56岁以上'
# print convert_age(str1)

users_info = load_users_info()
pros_info = load_pros_info()
 
print 'user count is :' + str(len(users_info))
print 'product count is '+ str(len(pros_info))
 
# print users_info['200001']
# print pros_info['10']
# 
# print users_info['200001'][1]=='2'
# print pros_info['10'][0]=='8'





