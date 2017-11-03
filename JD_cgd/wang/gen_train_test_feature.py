#_*_ coding=utf-8
'''
Created on 2017年5月6日

@author: wang
'''
import age_product_cross_feature
import read_proinfo_userinfo
import sex_product_cross_feature
import user_product_cross_feature
import pandas as pd
from datetime import datetime
from datetime import timedelta


users_info = read_proinfo_userinfo.users_info
pros_info = read_proinfo_userinfo.pros_info

age_product_cross_feature.load_features() 
sex_product_cross_feature.load_features()
user_product_cross_feature.load_features()

def get_dates( all_data =False, step = 1):
    train_start_dates = []
    train_end_dates = []
    test_start_dates = []
    test_end_dates = []
    start = '2016-02-01'
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

def gen_date_list(start_date, end_date, step = 1):
    start_days = datetime.strptime(start_date, '%Y-%m-%d')
    end_days = datetime.strptime(end_date, '%Y-%m-%d')
    dates_list = []
    while start_days < end_days:
        dates_list.append(start_days.strftime('%Y-%m-%d'))
        start_days += timedelta(days=step)
    return dates_list

'''
获取用户-产品品牌的交叉特征(5个转化率)
'''
def gen_user_pro_cross_feature(dates_list, user_id_str, sku_id_str):
    features = [0.0,0.0,0.0,0.0,0.0]
    action_type_count = [0,0,0,0,0,0]
    user_to_brand_six_behavior = user_product_cross_feature.user_to_brand_six_behavior
    brand = user_product_cross_feature.pro_to_brand[sku_id_str]
    for date_day in dates_list:
        key = user_id_str+'-'+brand+'-'+date_day
        if user_to_brand_six_behavior.has_key(key):
            action_type_count = [action_type_count[i]+user_to_brand_six_behavior[key][i] for i in range(6)]
    if(action_type_count[3]==0):
        return features
    for i in [0,1,2,4,5]:
        idx = i
        if i>3:
            idx = i-1
        if(action_type_count[i]>0):
            features[idx] = min(action_type_count[3]*1.0/action_type_count[i],1.0)
        else:
            features[idx] = 1
    return features




def gen_age_pro_cross_feature(dates_list, user_id_str, sku_id_str):
    age_cross_pur_rate = [0.0,0.0] #[年龄与产品交叉, 年龄与品牌交叉]，购买率
    age_cross_pur_trans_pro = [0.0,0.0,0.0,0.0,0.0]
    age_cross_pur_trans_brand = [0.0,0.0,0.0,0.0,0.0]
    if(not users_info.has_key(user_id_str)):
        return age_cross_pur_rate+age_cross_pur_trans_pro+age_cross_pur_trans_brand
    age = int(users_info[user_id_str][0])
    brand = user_product_cross_feature.pro_to_brand[sku_id_str]
    action_type1_count_pro = [0,0,0,0,0,0,0]
    action_type2_count_pro = [0,0,0,0,0,0,0]
    action_type3_count_pro = [0,0,0,0,0,0,0]
    action_type4_count_pro = [0,0,0,0,0,0,0]
    action_type5_count_pro = [0,0,0,0,0,0,0]
    action_type6_count_pro = [0,0,0,0,0,0,0]
    
    action_type1_count_brand = [0,0,0,0,0,0,0]
    action_type2_count_brand = [0,0,0,0,0,0,0]
    action_type3_count_brand = [0,0,0,0,0,0,0]
    action_type4_count_brand = [0,0,0,0,0,0,0]
    action_type5_count_brand = [0,0,0,0,0,0,0]
    action_type6_count_brand = [0,0,0,0,0,0,0]
    
    for date_day in dates_list:
        pro_key = sku_id_str+'-'+date_day
        view_product_to_age = age_product_cross_feature.view_product_to_age
        if(view_product_to_age.has_key(pro_key)):
            action_type1_count_pro = [action_type1_count_pro[i]+view_product_to_age[pro_key][i] for i in range(7)]
        add_product_to_age = age_product_cross_feature.add_product_to_age
        if(add_product_to_age.has_key(pro_key)):
            action_type2_count_pro = [action_type2_count_pro[i]+add_product_to_age[pro_key][i] for i in range(7)]
        del_product_to_age = age_product_cross_feature.del_product_to_age
        if(del_product_to_age.has_key(pro_key)):
            action_type3_count_pro = [action_type3_count_pro[i]+del_product_to_age[pro_key][i] for i in range(7)]
        pur_product_to_age = age_product_cross_feature.pur_product_to_age
        if(pur_product_to_age.has_key(pro_key)):
            action_type4_count_pro = [action_type4_count_pro[i]+pur_product_to_age[pro_key][i] for i in range(7)]
        follow_product_to_age = age_product_cross_feature.follow_product_to_age
        if(follow_product_to_age.has_key(pro_key)):
            action_type5_count_pro = [action_type5_count_pro[i]+follow_product_to_age[pro_key][i] for i in range(7)]
        click_product_to_age = age_product_cross_feature.click_product_to_age
        if(click_product_to_age.has_key(pro_key)):
            action_type6_count_pro = [action_type6_count_pro[i]+click_product_to_age[pro_key][i] for i in range(7)]
        
        
        brand_key = brand+'-'+date_day
        view_brand_to_age = age_product_cross_feature.view_brand_to_age
        if(view_brand_to_age.has_key(brand_key)):
            action_type1_count_brand = [action_type1_count_brand[i]+view_brand_to_age[brand_key][i] for i in range(7)]
        add_brand_to_age = age_product_cross_feature.add_brand_to_age
        if(add_brand_to_age.has_key(brand_key)):
            action_type2_count_brand = [action_type2_count_brand[i]+add_brand_to_age[brand_key][i] for i in range(7)]
        del_brand_to_age = age_product_cross_feature.del_brand_to_age
        if(del_brand_to_age.has_key(brand_key)):
            action_type3_count_brand = [action_type3_count_brand[i]+del_brand_to_age[brand_key][i] for i in range(7)]
        pur_brand_to_age = age_product_cross_feature.pur_brand_to_age
        if(pur_brand_to_age.has_key(brand_key)):
            action_type4_count_brand = [action_type4_count_brand[i]+pur_brand_to_age[brand_key][i] for i in range(7)]
        follow_brand_to_age = age_product_cross_feature.follow_brand_to_age
        if(follow_brand_to_age.has_key(brand_key)):
            action_type5_count_brand = [action_type5_count_brand[i]+follow_brand_to_age[brand_key][i] for i in range(7)]
        click_brand_to_age = age_product_cross_feature.click_brand_to_age
        if(click_brand_to_age.has_key(brand_key)):
            action_type6_count_brand = [action_type6_count_brand[i]+click_brand_to_age[brand_key][i] for i in range(7)]
        
    sum1 = sum(action_type4_count_pro)
    if sum1!=0:
        age_cross_pur_rate[0] = action_type4_count_pro[age]*1.0/sum1 
    sum2 = sum(action_type4_count_brand)
    if sum2!=0:
        age_cross_pur_rate[1] = action_type4_count_brand[age]*1.0/sum2
    if(action_type4_count_pro[age]!=0):
        action_type_count_pro = []
        action_type_count_pro.append(action_type1_count_pro)
        action_type_count_pro.append(action_type2_count_pro)
        action_type_count_pro.append(action_type3_count_pro)
        action_type_count_pro.append(action_type5_count_pro)
        action_type_count_pro.append(action_type6_count_pro)
        for i in [0,1,2,3,4]:
            if(action_type_count_pro[i][age]>0):
                age_cross_pur_trans_pro[i] = min(action_type4_count_pro[age]*1.0/action_type_count_pro[i][age],1.0)
            else:
                age_cross_pur_trans_pro[i]
    if(action_type4_count_brand[age]!=0):
        action_type_count_brand = []
        action_type_count_brand.append(action_type1_count_brand)
        action_type_count_brand.append(action_type2_count_brand)
        action_type_count_brand.append(action_type3_count_brand)
        action_type_count_brand.append(action_type5_count_brand)
        action_type_count_brand.append(action_type6_count_brand)
        for i in [0,1,2,3,4]:
            if(action_type_count_brand[i][age]>0):
                age_cross_pur_trans_pro[i] = min(action_type4_count_brand[age]*1.0/action_type_count_brand[i][age],1.0)
            else:
                age_cross_pur_trans_pro[i]
    return age_cross_pur_rate+age_cross_pur_trans_pro+age_cross_pur_trans_brand

def gen_sex_pro_cross_feature(dates_list, user_id_str, sku_id_str):
    sex_cross_pur_rate = [0.0,0.0] #[性别与产品交叉, 性别与品牌交叉]，购买率
    sex_cross_pur_trans_pro = [0.0,0.0,0.0,0.0,0.0]
    sex_cross_pur_trans_brand = [0.0,0.0,0.0,0.0,0.0]
    if(not users_info.has_key(user_id_str)):
        return sex_cross_pur_rate+sex_cross_pur_trans_pro+sex_cross_pur_trans_brand
    sex = int(users_info[user_id_str][1])
    brand = user_product_cross_feature.pro_to_brand[sku_id_str]
    action_type1_count_pro = [0,0,0]
    action_type2_count_pro = [0,0,0]
    action_type3_count_pro = [0,0,0]
    action_type4_count_pro = [0,0,0]
    action_type5_count_pro = [0,0,0]
    action_type6_count_pro = [0,0,0]
    
    action_type1_count_brand = [0,0,0]
    action_type2_count_brand = [0,0,0]
    action_type3_count_brand = [0,0,0]
    action_type4_count_brand = [0,0,0]
    action_type5_count_brand = [0,0,0]
    action_type6_count_brand = [0,0,0]
    
    for date_day in dates_list:
        pro_key = sku_id_str+'-'+date_day
        view_product_to_sex = sex_product_cross_feature.view_product_to_sex
        if(view_product_to_sex.has_key(pro_key)):
            action_type1_count_pro = [action_type1_count_pro[i]+view_product_to_sex[pro_key][i] for i in range(3)]
        add_product_to_sex = sex_product_cross_feature.add_product_to_sex
        if(add_product_to_sex.has_key(pro_key)):
            action_type2_count_pro = [action_type2_count_pro[i]+add_product_to_sex[pro_key][i] for i in range(3)]
        del_product_to_sex = sex_product_cross_feature.del_product_to_sex
        if(del_product_to_sex.has_key(pro_key)):
            action_type3_count_pro = [action_type3_count_pro[i]+del_product_to_sex[pro_key][i] for i in range(3)]
        pur_product_to_sex = sex_product_cross_feature.pur_product_to_sex
        if(pur_product_to_sex.has_key(pro_key)):
            action_type4_count_pro = [action_type4_count_pro[i]+pur_product_to_sex[pro_key][i] for i in range(3)]
        follow_product_to_sex = sex_product_cross_feature.follow_product_to_sex
        if(follow_product_to_sex.has_key(pro_key)):
            action_type5_count_pro = [action_type5_count_pro[i]+follow_product_to_sex[pro_key][i] for i in range(3)]
        click_product_to_sex = sex_product_cross_feature.click_product_to_sex
        if(click_product_to_sex.has_key(pro_key)):
            action_type6_count_pro = [action_type6_count_pro[i]+click_product_to_sex[pro_key][i] for i in range(3)]
        
        
        brand_key = brand+'-'+date_day
        view_brand_to_sex = sex_product_cross_feature.view_brand_to_sex
        if(view_brand_to_sex.has_key(brand_key)):
            action_type1_count_brand = [action_type1_count_brand[i]+view_brand_to_sex[brand_key][i] for i in range(3)]
        add_brand_to_sex = sex_product_cross_feature.add_brand_to_sex
        if(add_brand_to_sex.has_key(brand_key)):
            action_type2_count_brand = [action_type2_count_brand[i]+add_brand_to_sex[brand_key][i] for i in range(3)]
        del_brand_to_sex = sex_product_cross_feature.del_brand_to_sex
        if(del_brand_to_sex.has_key(brand_key)):
            action_type3_count_brand = [action_type3_count_brand[i]+del_brand_to_sex[brand_key][i] for i in range(3)]
        pur_brand_to_sex = sex_product_cross_feature.pur_brand_to_sex
        if(pur_brand_to_sex.has_key(brand_key)):
            action_type4_count_brand = [action_type4_count_brand[i]+pur_brand_to_sex[brand_key][i] for i in range(3)]
        follow_brand_to_sex = sex_product_cross_feature.follow_brand_to_sex
        if(follow_brand_to_sex.has_key(brand_key)):
            action_type5_count_brand = [action_type5_count_brand[i]+follow_brand_to_sex[brand_key][i] for i in range(3)]
        click_brand_to_sex = sex_product_cross_feature.click_brand_to_sex
        if(click_brand_to_sex.has_key(brand_key)):
            action_type6_count_brand = [action_type6_count_brand[i]+click_brand_to_sex[brand_key][i] for i in range(3)]
        
    sum1 = sum(action_type4_count_pro)
    if sum1!=0:
        sex_cross_pur_rate[0] = action_type4_count_pro[sex]*1.0/sum1 
    sum2 = sum(action_type4_count_brand)
    if sum2!=0:
        sex_cross_pur_rate[1] = action_type4_count_brand[sex]*1.0/sum2
    if(action_type4_count_pro[sex]!=0):
        action_type_count_pro = []
        action_type_count_pro.append(action_type1_count_pro)
        action_type_count_pro.append(action_type2_count_pro)
        action_type_count_pro.append(action_type3_count_pro)
        action_type_count_pro.append(action_type5_count_pro)
        action_type_count_pro.append(action_type6_count_pro)
        for i in [0,1,2,3,4]:
            if(action_type_count_pro[i][sex]>0):
                sex_cross_pur_trans_pro[i] = min(action_type4_count_pro[sex]*1.0/action_type_count_pro[i][sex],1.0)
            else:
                sex_cross_pur_trans_pro[i]
    if(action_type4_count_brand[sex]!=0):
        action_type_count_brand = []
        action_type_count_brand.append(action_type1_count_brand)
        action_type_count_brand.append(action_type2_count_brand)
        action_type_count_brand.append(action_type3_count_brand)
        action_type_count_brand.append(action_type5_count_brand)
        action_type_count_brand.append(action_type6_count_brand)
        for i in [0,1,2,3,4]:
            if(action_type_count_brand[i][sex]>0):
                sex_cross_pur_trans_pro[i] = min(action_type4_count_brand[sex]*1.0/action_type_count_brand[i][sex],1.0)
            else:
                sex_cross_pur_trans_pro[i]  
    return sex_cross_pur_rate+sex_cross_pur_trans_pro+sex_cross_pur_trans_brand

def gen_cross_features(user_id_str, sku_id_str, start_date, end_date):
    dates_list = gen_date_list(start_date, end_date, step=1)
    user_pro_cross_feature = gen_user_pro_cross_feature(dates_list, user_id_str, sku_id_str)
    age_pro_cross_feature = gen_age_pro_cross_feature(dates_list, user_id_str, sku_id_str)
    sex_pro_cross_feature = gen_sex_pro_cross_feature(dates_list, user_id_str, sku_id_str)
    return user_pro_cross_feature+age_pro_cross_feature+sex_pro_cross_feature

def merge_features(filename):
    features_name =[]
    features_name.append('user_pro_cross_trans_1')
    features_name.append('user_pro_cross_trans_2')
    features_name.append('user_pro_cross_trans_3')
    features_name.append('user_pro_cross_trans_4')
    features_name.append('user_pro_cross_trans_5')
    features_name.append('age_cross_pur_rate_1')
    features_name.append('age_cross_pur_rate_2')
    features_name.append('age_cross_pur_trans_pro_1')
    features_name.append('age_cross_pur_trans_pro_2')
    features_name.append('age_cross_pur_trans_pro_3')
    features_name.append('age_cross_pur_trans_pro_4')
    features_name.append('age_cross_pur_trans_pro_5')
    features_name.append('age_cross_pur_trans_brand_1')
    features_name.append('age_cross_pur_trans_brand_2')
    features_name.append('age_cross_pur_trans_brand_3')
    features_name.append('age_cross_pur_trans_brand_4')
    features_name.append('age_cross_pur_trans_brand_5')
    features_name.append('sex_cross_pur_rate_1')
    features_name.append('sex_cross_pur_rate_2')
    features_name.append('sex_cross_pur_trans_pro_1')
    features_name.append('sex_cross_pur_trans_pro_2')
    features_name.append('sex_cross_pur_trans_pro_3')
    features_name.append('sex_cross_pur_trans_pro_4')
    features_name.append('sex_cross_pur_trans_pro_5')
    features_name.append('sex_cross_pur_trans_brand_1')
    features_name.append('age_cross_pur_trans_brand_2')
    features_name.append('sex_cross_pur_trans_brand_3')
    features_name.append('sex_cross_pur_trans_brand_4')
    features_name.append('sex_cross_pur_trans_brand_5')
    df = pd.read_csv( './data/'+filename )
    feats = []
    count = 0
    for index, row in df.iterrows():
        user_id = str(row['user_id'])[:-2]
        sku_id = str(row['sku_id'])
        start_date = str(row['start_date'])
        end_date = str(row['end_date'])
        feat = gen_cross_features(user_id, sku_id, start_date, end_date)
        feats.append( feat )
        count +=1
        if(count %10000==0):
            print(str(count)+' already processed')
    df2 = pd.DataFrame( feats , columns=features_name)
    #保存feature
    df2.to_csv('./data/user_cross_sku_'+filename, index=False, index_label=False)
    #保存合并的feature
    df = pd.concat([df, df2],axis=1)#
    df.to_csv('./data/new_'+filename, index=False, index_label=False)
    
merge_features('basic_feats.csv')
merge_features('test_2016-03-12_2016-04-11.csv')
merge_features('test_2016-03-17_2016-04-16.csv')
    




