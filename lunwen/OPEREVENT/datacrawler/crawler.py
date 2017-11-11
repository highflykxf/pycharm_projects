#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path
import csv
from newspaper import Article
import shutil
import numpy as np
import sys
import time
import codecs
import csv
import pandas as pd
reload(sys)
sys.setdefaultencoding('utf-8')


def list_current_data_files(path_this):
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(path_this) if isfile(join(path_this, f))]
    return onlyfiles


def import_from_csv(im_filename, index_col=False, sep=',', **kwargs):
    return pd.io.parsers.read_csv(im_filename, index_col=index_col, sep=sep, encoding='utf-8', **kwargs)


# Dictionary of codebooks
# The constructor by default uses the flattened and shortened field names as keys to the dictionaries of Notes and original names of the Fields.
# Orginal fieldnames lookup: e.g. dict_meta['Fields']['ti_id']  produces GlobalEventID
# Orginal fieldnames lookup: e.g. dict_meta['Fields']['ti_y'] produces Year
# Orginal Notes lookup: e.g. dict_meta['Fields']['ti_y'] produces "(integer) Alternative formatting of the event date, in YYYY format."
def dict_meta_constructor(df_meta, column_key=0):
    key = []
    value = []
    columns_list = df_meta.columns.tolist()
    fieldname_key = columns_list[column_key]  # 0 -->'flattened_shortened'
    for i, column_name in enumerate(columns_list):
        if i == column_key:
            pass
        else:
            key.append(column_name)
            value.append(df_meta.set_index(fieldname_key)[column_name].to_dict())
    dict_outcome = dict(zip(key, value))
    return dict_outcome


def export_to_csv(df, ex_filename, sep=',', **kwargs):
    if sep == ',':
        df.to_csv(ex_filename, sep=sep, quoting=csv.QUOTE_ALL, na_rep='{na}', encoding='utf-8', **kwargs)  # +'.csv'
    if sep == '\t':
        df.to_csv(ex_filename, sep=sep, quoting=csv.QUOTE_NONE, na_rep='{na}', encoding='utf-8',
                  **kwargs)  # +'.tsv'  , escapechar="'", quotechar=""


if __name__ == "__main__":
    # 读取csv文件，生成存放结果文件的目录
    path_local_script = os.path.dirname(os.path.realpath(__file__))
    path_local_data = os.path.abspath(os.path.join(path_local_script, os.pardir, "data"))
    print ">>Current script folder:{0}".format(path_local_script)
    print ">>Default data folder:{0}".format(path_local_data)
    if not os.path.exists(path_local_data):
        print ">>Data folder does not exist yet...please run GDELT_data_download.py first"
    else:
        print ">>Data folder exists, with csv files including"
        files_data_existing_csv = [fn for fn in list_current_data_files(path_local_data) if fn[-4:].lower() == ".csv"]
        print ">>>", files_data_existing_csv

    output_path = os.path.join(path_local_script, os.pardir, "data", "newspaper_dataset")
    if os.path.isdir(output_path):
        print "The directory exists."
    else:
        shutil.rmtree(output_path, True)
        os.makedirs(output_path)

    # ===load meta data information for the GDELT dataset===
    # meta data information is scraped from Philip A. Schrodt's "CAMEO Conflict and Mediation Event Observations Event and Actor Codebook"
    # filename: CAMEO.Manual.1.1b3.pdf
    df_meta = import_from_csv(os.path.join(path_local_data, "GDELT_meta.tsv"), sep='\t', header=0)
    dict_meta = dict_meta_constructor(df_meta, column_key=df_meta.columns.get_loc(u'flattened_shortened'))

    df_list = []
    # loading the freshly extracted files
    list_flattened_shortened = df_meta['flattened_shortened'].tolist()

    # the following two lines need debugging with data
    # list_dtype=[eval(x) for x in df_meta['dtype'].tolist()]
    # dict_dtype1=dict(zip(list_flattened_shortened,list_dtype))
    # print dict_dtype1

    dict_dtype = {u'ti_id': np.int32,
                  u'ti_d': np.int32,
                  u'ti_my': np.int32,
                  u'ti_y': np.int32,
                  u'ti_f': np.float64,
                  u'a1_knowngroup': np.dtype((str, 16)),
                  u'a2_knowngroup': np.dtype((str, 16)),
                  u'a1_type1': np.dtype((str, 16)),
                  u'a1_type2': np.dtype((str, 16)),
                  u'a1_type3': np.dtype((str, 16)),
                  u'a2_type1': np.dtype((str, 16)),
                  u'a2_type2': np.dtype((str, 16)),
                  u'a2_type3': np.dtype((str, 16)),
                  u'a1_religion1': np.dtype((str, 255)),
                  u'a1_religion2': np.dtype((str, 255)),
                  u'a2_religion1': np.dtype((str, 255)),
                  u'a2_religion2': np.dtype((str, 255)),
                  u'a1_ethnic': np.dtype((str, 16)),
                  u'a2_ethnic': np.dtype((str, 16)),
                  u'da_dateadded': np.int32,
                  u'da_sourceurl': np.dtype((str, 1024))
                  }
    useful_col = [u'ti_id', u'ti_d', u'ti_my', u'ti_y', u'a1_name',
                  u'a1_country', u'a2_name', u'a2_country', u'ac_goldsteinscale',
                  u'ac_nummentions', u'ac_numsources', u'ac_numarticles', u'ac_avgtone',
                  u'da_dateadded', u'da_sourceurl']
    start_num_e = 335
    index_e = 1
    for i, fn in enumerate(files_data_existing_csv, start=1):
        if index_e<=start_num_e:
            index_e += 1
            continue
        output_child_path = os.path.join(output_path, os.curdir, fn[0:9])
        if os.path.isdir(output_child_path):
            print "The directory exists."
        else:
            shutil.rmtree(output_child_path, True)
            os.makedirs(output_child_path)
        df = import_from_csv(os.path.join(path_local_data, fn), sep='\t', header=None,
                             names=df_meta['flattened_shortened'].tolist(), parse_dates=['ti_d'],
                             infer_datetime_format=True, dtype=dict_dtype)

        # select only CHN
        # df=df[(df['a1_country'] =='CHN') & (df['a2_country'] =='CHN')]
        # if i == 1:
        #     df_sel = df
        # else:
        #     df_sel = df_sel.append(df, ignore_index=True)
        df_col_filter = df[useful_col]
        export_to_csv(df_col_filter,
                      ex_filename=os.path.join(output_child_path, os.curdir, "gdelt_1.tsv"),
                      sep='\t',
                      escapechar="'",
                      index=False)
        url_list = df[u'da_sourceurl'].tolist()
        num = 1
        start_num = 1
        for url in url_list:
            if start_num > num:
                num += 1
                continue
            name = "%06d" % num
            fileName = output_child_path + "\\"+ str(name) + ".txt"
            info = codecs.open(fileName, 'w', 'utf-8')
            print u'File Name: ', fileName
            if url.startswith('http'):
                article = Article(url, fetch_images=False)
                time.sleep(2)
                try:
                    article.download()
                    article.parse()
                except Exception as e:
                    print e
                # print article.title
                # print article.text
                # df_col_filter.loc[num-1, u'title'] = article.title
                # df_col_filter.loc[num-1, u'content'] = article.text
                info.write(url + '\n')
                info.write(article.title + '\n')
                info.write(article.text + '\n')
            info.close()
            num += 1

