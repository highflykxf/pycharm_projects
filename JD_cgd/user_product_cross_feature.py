#_*_ coding=utf-8
'''
Created on 2017年5月5日

@author: wang
'''

import sex_product_cross_feature
import age_product_cross_feature
import read_proinfo_userinfo

users_info = read_proinfo_userinfo.users_info
pros_info = read_proinfo_userinfo.pros_info

def process_sex_product_cross(action_parts):
    if(not users_info.has_key(action_parts[0])):
        return
    sex = int(users_info[action_parts[0]][1])
    action_type = action_parts[4]
    day_time = action_parts[2].split(' ')[0]
    pro_id = action_parts[1] + '-'+day_time
    brand = action_parts[6] + '-'+day_time
    if(action_type=='1'):
        view_product_to_sex = sex_product_cross_feature.view_product_to_sex
        view_brand_to_sex = sex_product_cross_feature.view_brand_to_sex
        if(not view_product_to_sex.has_key(pro_id)):
            view_product_to_sex[pro_id] = [0,0,0]
        if(not view_brand_to_sex.has_key(brand)):
            view_brand_to_sex[brand] = [0,0,0]
        view_product_to_sex[pro_id][sex] += 1
        view_brand_to_sex[brand][sex] += 1
        return
    
    if(action_type=='2'):
        add_product_to_sex = sex_product_cross_feature.add_product_to_sex
        add_brand_to_sex = sex_product_cross_feature.add_brand_to_sex
        if(not add_product_to_sex.has_key(pro_id)):
            add_product_to_sex[pro_id] = [0,0,0]
        if(not add_brand_to_sex.has_key(brand)):
            add_brand_to_sex[brand] = [0,0,0]
        add_product_to_sex[pro_id][sex] += 1
        add_brand_to_sex[brand][sex] += 1
        return
    
    if(action_type=='3'):
        del_product_to_sex = sex_product_cross_feature.del_product_to_sex
        del_brand_to_sex = sex_product_cross_feature.del_brand_to_sex
        if(not del_product_to_sex.has_key(pro_id)):
            del_product_to_sex[pro_id] = [0,0,0]
        if(not del_brand_to_sex.has_key(brand)):
            del_brand_to_sex[brand] = [0,0,0]
        del_product_to_sex[pro_id][sex] += 1
        del_brand_to_sex[brand][sex] += 1
        return
    
    if(action_type=='4'):
        pur_product_to_sex = sex_product_cross_feature.pur_product_to_sex
        pur_brand_to_sex = sex_product_cross_feature.pur_brand_to_sex
        if(not pur_product_to_sex.has_key(pro_id)):
            pur_product_to_sex[pro_id] = [0,0,0]
        if(not pur_brand_to_sex.has_key(brand)):
            pur_brand_to_sex[brand] = [0,0,0]
        pur_product_to_sex[pro_id][sex] += 1
        pur_brand_to_sex[brand][sex] += 1
        return
    
    if(action_type=='5'):
        follow_product_to_sex = sex_product_cross_feature.follow_product_to_sex
        follow_brand_to_sex = sex_product_cross_feature.follow_brand_to_sex
        if(not follow_product_to_sex.has_key(pro_id)):
            follow_product_to_sex[pro_id] = [0,0,0]
        if(not follow_brand_to_sex.has_key(brand)):
            follow_brand_to_sex[brand] = [0,0,0]
        follow_product_to_sex[pro_id][sex] += 1
        follow_brand_to_sex[brand][sex] += 1
        return
    
    if(action_type=='6'):
        click_product_to_sex = sex_product_cross_feature.click_product_to_sex
        click_brand_to_sex = sex_product_cross_feature.click_brand_to_sex
        if(not click_product_to_sex.has_key(pro_id)):
            click_product_to_sex[pro_id] = [0,0,0]
        if(not click_brand_to_sex.has_key(brand)):
            click_brand_to_sex[brand] = [0,0,0]
        click_product_to_sex[pro_id][sex] += 1
        click_brand_to_sex[brand][sex] += 1
        return
    
    pass


def process_age_product_cross(action_parts):
    if(not users_info.has_key(action_parts[0])):
        return
    age = int(users_info[action_parts[0]][0])
    action_type = action_parts[4]
    day_time = action_parts[2].split(' ')[0]
    pro_id = action_parts[1]+'-'+day_time
    brand = action_parts[6]+'-'+day_time
    if(action_type=='1'):
        view_product_to_age = age_product_cross_feature.view_product_to_age
        view_brand_to_age = age_product_cross_feature.view_brand_to_age
        if(not view_product_to_age.has_key(pro_id)):
            view_product_to_age[pro_id] = [0,0,0,0,0,0,0]
        if(not view_brand_to_age.has_key(brand)):
            view_brand_to_age[brand] = [0,0,0,0,0,0,0]
        view_product_to_age[pro_id][age] += 1
        view_brand_to_age[brand][age] += 1
        return
    if(action_type=='2'):
        add_product_to_age = age_product_cross_feature.add_product_to_age
        add_brand_to_age = age_product_cross_feature.add_brand_to_age
        if(not add_product_to_age.has_key(pro_id)):
            add_product_to_age[pro_id] = [0,0,0,0,0,0,0]
        if(not add_brand_to_age.has_key(brand)):
            add_brand_to_age[brand] = [0,0,0,0,0,0,0]
        add_product_to_age[pro_id][age] += 1
        add_brand_to_age[brand][age] += 1
        return
    if(action_type=='3'):
        del_product_to_age = age_product_cross_feature.del_product_to_age
        del_brand_to_age = age_product_cross_feature.del_brand_to_age
        if(not del_product_to_age.has_key(pro_id)):
            del_product_to_age[pro_id] = [0,0,0,0,0,0,0]
        if(not del_brand_to_age.has_key(brand)):
            del_brand_to_age[brand] = [0,0,0,0,0,0,0]
        del_product_to_age[pro_id][age] += 1
        del_brand_to_age[brand][age] += 1
        return
    if(action_type=='4'):
        pur_product_to_age = age_product_cross_feature.pur_product_to_age
        pur_brand_to_age = age_product_cross_feature.pur_brand_to_age
        if(not pur_product_to_age.has_key(pro_id)):
            pur_product_to_age[pro_id] = [0,0,0,0,0,0,0]
        if(not pur_brand_to_age.has_key(brand)):
            pur_brand_to_age[brand] = [0,0,0,0,0,0,0]
        pur_product_to_age[pro_id][age] += 1
        pur_brand_to_age[brand][age] += 1
        return
    if(action_type=='5'):
        follow_product_to_age = age_product_cross_feature.follow_product_to_age
        follow_brand_to_age = age_product_cross_feature.follow_brand_to_age
        if(not follow_product_to_age.has_key(pro_id)):
            follow_product_to_age[pro_id] = [0,0,0,0,0,0,0]
        if(not follow_brand_to_age.has_key(brand)):
            follow_brand_to_age[brand] = [0,0,0,0,0,0,0]
        follow_product_to_age[pro_id][age] += 1
        follow_brand_to_age[brand][age] += 1
        return
    if(action_type=='6'):
        click_product_to_age = age_product_cross_feature.click_product_to_age
        click_brand_to_age = age_product_cross_feature.click_brand_to_age
        if(not click_product_to_age.has_key(pro_id)):
            click_product_to_age[pro_id] = [0,0,0,0,0,0,0]
        if(not click_brand_to_age.has_key(brand)):
            click_brand_to_age[brand] = [0,0,0,0,0,0,0]
        click_product_to_age[pro_id][age] += 1
        click_brand_to_age[brand][age] += 1
        return
    pass


#产品的类别映射
pro_to_brand = {}
def process_pro_to_brand(action_parts):
    sku_id = action_parts[1]
    brand = action_parts[6]
    if(not pro_to_brand.has_key(sku_id)):
        pro_to_brand[sku_id] = brand
    
#每个用户对每个品牌的6种行为的次数统计，key为"userid-brand"，value为6维数组[,...,]表示6种行为的次数
user_to_brand_six_behavior = {}
# #每个用户对每个类别的6种行为的次数统计，key为"userid-cate"，value为6维数组[,...,]表示6种行为的次数
# user_to_cat_six_behavior = {}

def process_user_product_cross(action_parts):
    day_time = action_parts[2].split(' ')[0]
    key = action_parts[0]+'-'+action_parts[6]+'-'+day_time
    if(not user_to_brand_six_behavior.has_key(key)):
        user_to_brand_six_behavior[key] = [0,0,0,0,0,0]
    user_to_brand_six_behavior[key][int(action_parts[4])-1] += 1
    
import pickle as pkl    
def save_features():
    print('user_to_brand_six_behavior count is '+str(len(user_to_brand_six_behavior)))
    pkl.dump(user_to_brand_six_behavior, open('./data/user_to_brand_six_behavior.pkl','wb'), protocol=2)
    
    print('pro_to_brand count is '+str(len(pro_to_brand)))
    pkl.dump(pro_to_brand, open('./data/pro_to_brand.pkl','wb'), protocol=2)

def load_features():
    global user_to_brand_six_behavior
    global pro_to_brand
    user_to_brand_six_behavior = pkl.load(open('./data/user_to_brand_six_behavior.pkl','rb'))
    pro_to_brand = pkl.load(open('./data/pro_to_brand.pkl','rb'))
    print('user_to_brand_six_behavior count is '+str(len(user_to_brand_six_behavior)))
    print('pro_to_brand count is '+str(len(pro_to_brand)))

def read_pre_product(filepath):
    f1 = open(filepath, 'r')
    count = 0
    for line in f1:
        if count==0:
            count +=1
            continue
        line = line[:-1] #去掉\n
        action_parts = line.split(',')
        action_parts[0] = action_parts[0][:-2]
        count +=1
        if(count%100000==0):
            print(str(count)+' already processed*****')
#         if(action_parts[2]>'2016-03-09 24:00:00' and action_parts[2]<'2016-04-11 00:00:00'):
        process_pro_to_brand(action_parts)
        process_sex_product_cross(action_parts)
        process_age_product_cross(action_parts)
        process_user_product_cross(action_parts)
    f1.close()

if __name__ == '__main__':
    read_pre_product(r'./data/JData_Action_201602.csv')
    print('201602 already processed******')
    read_pre_product(r'./data/JData_Action_201603.csv')
    print('201603 already processed******')
    read_pre_product(r'./data/JData_Action_201604.csv')
    age_product_cross_feature.save_features()
    sex_product_cross_feature.save_features()
    save_features()
#     age_product_cross_feature.load_features()
#     
#     sex_product_cross_feature.load_features()
#     load_features()
#     pro_to_brand['75018']
    pass