# -*- coding: utf-8 -*-
from __future__ import print_function

import math
import pickle
import os
import re
import string
import sys
import csv
import shutil
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from nltk.tag import pos_tag
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from collections import defaultdict
import kmedoids

reload(sys)
sys.setdefaultencoding('utf-8')

# 词干还原、去除停用词、命名实体识别
# load nltk's English stopwords as variable called 'stopwords'
stopwords = nltk.corpus.stopwords.words('english')
# load nltk's SnowballStemmer as variabled 'stemmer'
stemmer = SnowballStemmer("english")


def proppers_POS(text):
    tagged = pos_tag(text.split())  # use NLTK's part of speech tagger
    propernouns = [word for word, pos in tagged if pos == 'NNP' or pos == 'NNPS']
    return propernouns


# strip any proper names from a text...unfortunately right now this is yanking the first word from a sentence too.
def strip_proppers(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word.islower()]
    return "".join([" " + i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()


def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


def extract_clusters(Z, threshold, n):
    clusters = {}
    ct = n
    for row in Z:
        if row[2] < threshold:
            n1 = int(row[0])
            n2 = int(row[1])

            if n1 >= n:
                l1 = clusters[n1]
                del (clusters[n1])
            else:
                l1 = [n1]

            if n2 >= n:
                l2 = clusters[n2]
                del (clusters[n2])
            else:
                l2 = [n2]
            l1.extend(l2)
            clusters[ct] = l1
            ct += 1
        else:
            return clusters


def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


def cluster_step_one(data_frame):
    used_col = [u'a1_name', u'a2_name',
                u'a1_type1', u'a2_type1', u'a1_geo_fullname',
                u'a2_geo_fullname']
    grouped = data_frame.groupby(used_col)
    clust_id = 1
    for name, group in grouped:
        mini_clust = group['news_order'].tolist()
        data_frame.loc[mini_clust, 'cluster_step_one'] = int(clust_id)
        clust_id += 1

    return data_frame
    # group_url = group.groupby([u'da_sourceurl'])
    # print (len(group_url))
    # if len(group_url) is not 1:
    # print (name)
    # print (group)


def cluster_step_zero(data_frame):
    num_count = {}
    used_col = [u'da_sourceurl']
    grouped = data_frame.groupby(used_col)
    clust_id = 1
    for name, group in grouped:
        mini_clust = group['news_order'].tolist()
        data_frame.loc[mini_clust, 'cluster_step_zero'] = int(clust_id)
        data_frame.loc[mini_clust, 'cluster_zero_num'] = len(mini_clust)
        if num_count.get(str(len(mini_clust))) is None:
            num_count[str(len(mini_clust))] = 1
        else:
            num_count[str(len(mini_clust))] += 1
        clust_id += 1
    return data_frame, clust_id - 1, num_count


def list_current_folders(abs_path):
    from os import listdir
    from os.path import isfile, join, isdir
    data_folders = [join(abs_path, f) for f in listdir(abs_path) if isdir(join(abs_path, f)) and f[0:3] == '201']
    return data_folders


def export_to_csv(df, ex_filename, sep=',', **kwargs):
    if sep == ',':
        df.to_csv(ex_filename, sep=sep, quoting=csv.QUOTE_ALL, na_rep='{na}', encoding='utf-8', **kwargs)  # +'.csv'
    if sep == '\t':
        df.to_csv(ex_filename, sep=sep, quoting=csv.QUOTE_NONE, na_rep='{na}', encoding='utf-8',
                  **kwargs)  # +'.tsv'  , escapechar="'", quotechar=""


if __name__ == "__main__":
    path_local_script = os.path.dirname(os.path.realpath(__file__))
    # 所有新闻语料
    path_local_data = os.path.abspath(
        os.path.join(path_local_script,
                     os.pardir, "data", "newspaper_dataset"))

    data_folders = list_current_folders(path_local_data)
    for data_folder in data_folders:
        path_result_folder = os.path.abspath(
            os.path.join(path_local_data, "results", data_folder[-8:]))
        if os.path.isdir(path_result_folder) is False:
            shutil.rmtree(path_result_folder, True)
            os.makedirs(path_result_folder)
        path_daily_num_count = os.path.abspath(os.path.join(path_result_folder, "daily_num_count.txt"))
        if int(data_folder[-8:]) != 20170201:
            continue
        # 读取原始的结构化的每日新闻数据集,
        path_tsv = os.path.abspath(os.path.join(data_folder, "daily_news.tsv"))
        df_dailynews = pd.read_csv(path_tsv, sep='\t', encoding='utf-8')
        df_dailynews['ac_goldsteinscale'] = df_dailynews['ac_goldsteinscale'].replace(u'{na}', 0.0).astype('float')
        df_dailynews['ac_avgtone'] = df_dailynews['ac_avgtone'].replace(u'{na}', 0.0).astype('float')

        df_dailynews = df_dailynews.replace(u'{na}', '')
        # df_dailynews = df_dailynews.iloc[0:10000]
        df_dailynews['news_order'] = pd.Series(range(0, len(df_dailynews)))

        dailynews_first_clusters_file = os.path.abspath(os.path.join(path_result_folder, "daily_news_first_cluster.tsv"))
        if os.path.isfile(dailynews_first_clusters_file):
            df_dailynews_first_cluster = pd.read_csv(dailynews_first_clusters_file, sep='\t')
            clust_zero_num = df_dailynews_first_cluster['cluster_step_zero'].max()
        else:
            # 根据url进行聚类
            df_dailynews_zero_cluster, clust_zero_num, num_count = cluster_step_zero(df_dailynews)

            # 记录每天重复性新闻的数量
            daily_num_count_file = file(path_daily_num_count, 'wb')
            for key, value in num_count.items():
                daily_num_count_file.write(str(key))
                daily_num_count_file.write('\t')
                daily_num_count_file.write(str(value))
                daily_num_count_file.write('\n')
            # pickle.dump(num_count, daily_num_count_file)
            daily_num_count_file.close()

            # 根据一些属性(相同的[主客体名字、类型、地理位置])先行进行聚类
            # df_dailynews_first_cluster = cluster_step_one(df_dailynews_zero_cluster)
            df_dailynews_first_cluster = df_dailynews_zero_cluster
            export_to_csv(df_dailynews_first_cluster,
                          ex_filename=os.path.abspath(
                              os.path.join(
                                  path_result_folder,
                                  "daily_news_first_cluster.tsv")),
                          sep='\t',
                          escapechar="'",
                          index=False)

        # 根据内容相似度来聚类
        # 记录真正需要处理的新闻order、url、title、contents、corpus
        news_corpus = []
        news_contents = []
        news_order = []
        news_url = []
        news_title = []
        num = 1
        news_num_test = df_dailynews.shape[0]
        clust_flag = np.zeros(clust_zero_num, dtype=int)
        while num <= news_num_test:
            name = "%06d" % num
            fileName = str(name) + ".txt"
            filename = os.path.abspath(os.path.join(data_folder, fileName))
            if os.path.isfile(filename) is False:
                continue
            cluster_zero_id = df_dailynews_first_cluster.loc[num - 1, 'cluster_step_zero']
            cluster_zero_id = int(cluster_zero_id)
            cluster_zero_num = df_dailynews_first_cluster.loc[num - 1, 'cluster_zero_num']
            cluster_zero_num = int(cluster_zero_num)
            # 过滤掉过长或者过短的新闻 :  800B < size(file) < 20000B，同时去除只被发布一次的新闻（有待实验）
            if (800 <= os.path.getsize(filename) <= 20000) and \
                    (clust_flag[cluster_zero_id - 1] == 0) and \
                    (cluster_zero_num >= 5):
                news_order.append(num)
                clust_flag[cluster_zero_id - 1] = 1
                lines = ""
                title = ""
                url_line = 1
                title_line = 2
                with open(filename, 'r') as f:
                    index = 1
                    for line in f:
                        if line != "\n" and index > 2:
                            lines = lines + line
                        elif line != "\n" and index == url_line:
                            news_url.append(line)
                            index += 1
                        elif line != "\n" and index == title_line:
                            title = line
                            news_title.append(line)
                            index += 1
                news_contents.append(lines)
                # 变相增加title的权重
                temp_corpus = lines
                news_corpus.append(temp_corpus)
            num += 1
        # 对于整个content词干还原，词性标注进行处理
        totalvocab_stemmed = []
        totalvocab_tokenized = []
        totalvocab_pos = []
        totalvocab = []
        # 对于title处理
        title_stemmed = []
        title_pos = []
        titlevocab = []
        idx = 0
        vocab_tokenized_file = os.path.abspath(
                              os.path.join(
                                  path_result_folder,
                                  "vocab_tokenized.txt"))
        news_order_vocab_file = os.path.abspath(
                              os.path.join(
                                  path_result_folder,
                                  "news_order_vocab.txt"))
        title_vocab_tokenized_file = os.path.abspath(
                os.path.join(path_result_folder,
                             "title_vocab_tokenized.txt"))
        news_order_vocab = open(news_order_vocab_file, 'w')

        if os.path.isfile(vocab_tokenized_file):
            with open(vocab_tokenized_file, 'r') as f:
                for line in f.readlines():
                    totalvocab.append(line.strip('\n'))
            with open(title_vocab_tokenized_file, 'r') as f:
                for line in f.readlines():
                    titlevocab.append(line.strip('\n'))
        else:
            for document in news_corpus:
                tmpcorpus = []
                tmptitles = []
                re_char = u'[’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
                document = re.sub(re_char, u'', document)
                allwords_stemmed = tokenize_and_stem(document)
                title_stemmed = tokenize_and_stem(news_title[idx])

                allwords_posed = proppers_POS(document)
                title_pos = proppers_POS(news_title[idx])

                tmpcorpus.extend(allwords_stemmed)
                tmpcorpus.extend(allwords_posed)

                tmptitles.extend(title_stemmed)
                tmptitles.extend(title_pos)

                titlevocab.append(' '.join(tmptitles))
                totalvocab.append(' '.join(tmpcorpus))
                idx += 1

            with open(vocab_tokenized_file, 'w') as file_vocab_tokenized:
                for i in range(len(totalvocab)):
                    file_vocab_tokenized.write(totalvocab[i] + '\n')
                    news_order_vocab.write('#')
                    news_order_vocab.write(str(news_order[i]))
                    news_order_vocab.write('\n')
                    news_order_vocab.write(totalvocab[i])
                    news_order_vocab.write('\n')
            news_order_vocab.close()
            with open(title_vocab_tokenized_file, 'w') as file_title_vocab_tokenized:
                for i in range(len(titlevocab)):
                    file_title_vocab_tokenized.write(titlevocab[i] + '\n')

        # 转化为tf-idf document-词项矩阵
        tfidf_vectorizer = TfidfVectorizer(max_df=0.5,
                                           max_features=10000,
                                           min_df=1,
                                           stop_words=stopwords,
                                           use_idf=True,
                                           tokenizer=tokenize_and_stem,
                                           ngram_range=(1, 1))

        tfidf_matrix = tfidf_vectorizer.fit_transform(totalvocab)

        with open(os.path.abspath(
                os.path.join(path_result_folder,
                             'tfidf_matrix.dat')), 'wb') as outfile:
            pickle.dump(tfidf_matrix, outfile, pickle.HIGHEST_PROTOCOL)

        tfidf_query_matrix = tfidf_vectorizer.fit_transform(titlevocab)
        with open(os.path.abspath(
                os.path.join(path_result_folder,
                             'tfidf_title_matrix.dat')), 'wb') as outfile:
            pickle.dump(tfidf_query_matrix, outfile, pickle.HIGHEST_PROTOCOL)

        words_weights = tfidf_matrix.toarray()
        sim_mat = np.zeros(words_weights.shape)
        terms = tfidf_vectorizer.get_feature_names()
        keywords_list = []
        # 选取每篇文档的top-k个关键词的tf-idf值
        # top_k = 14
        for top_k in range(9, 15, 1):
            index = 0
            for idx in words_weights:
                each_key = []
                arr = idx.tolist()
                for idy in range(top_k):
                    max_idy = arr.index(max(arr))
                    sim_mat[index][max_idy] = words_weights[index][max_idy]
                    each_key.append(terms[max_idy])
                    arr[max_idy] = 0
                keywords_list.append(each_key)
                index += 1
            # 只显示层次聚类的最后k个进行显示，对大于max_d的簇进行标注
            Z = linkage(sim_mat, 'ward')

            # 簇内最大距离,需要根据实际情况调整
            for max_dist in range(4, 20, 1):
                sse_value = 0.0
                clusters = extract_clusters(Z, 0.1 * max_dist, len(news_corpus))
                if clusters is None:
                    continue
                event_info = pd.DataFrame({'event_id': pd.Series(1, index=list(range(len(clusters))), dtype='int32'),
                                           'event_title': '',
                                           'num_mentions': 0,
                                           'num_sources': 0,
                                           'num_articles': 0,
                                           'goldsteinscale': 0.0,
                                           'event_weight': 0.0})
                # 输出聚类结果
                d_c_file = 'doc_cluster_' + str(int(top_k)) + '_' + str(int(max_dist)) + '.txt'
                cmp_file = 'cmp_' + str(int(top_k)) + '_' + str(int(max_dist)) + '.txt'
                std_q_file = 'query_' + str(int(top_k)) + '_' + str(int(max_dist)) + '.txt'

                event_info_output_file = 'daily_events_' + str(int(top_k)) + '_' + str(int(max_dist)) + '.csv'

                doc_clust_file = os.path.abspath(os.path.join(path_result_folder, d_c_file))
                compare_file = os.path.abspath(os.path.join(path_result_folder, cmp_file))
                std_query_file = os.path.abspath(os.path.join(path_result_folder, std_q_file))

                doc_clust = open(doc_clust_file, "w")
                compare_dist = open(compare_file, "w")
                std_query = open(std_query_file, "w")

                dict_cid_weight = {}
                cluster_dict = defaultdict(dict)
                cluster_id = 1
                cmp_count = 0
                c_flag = True
                for key in clusters:
                    for id in clusters[key]:
                        # 将基于文本相似度的cluster结果保存
                        df_dailynews_first_cluster.loc[news_order[id] - 1, 'cluster_step_two'] = cluster_id

                    # 找到聚类的medoid（news ID）以及该质心的title、簇间距离、簇内新闻数目、
                    cluster_data = sim_mat[clusters[key], :]
                    dist_matrix = pairwise_distances(cluster_data, metric='euclidean')
                    cluster_medoids, cluster_class = kmedoids.kMedoids(dist_matrix, 1)
                    id_medoids = clusters[key][cluster_medoids[0]]
                    df_dailynews_first_cluster.loc[news_order[id_medoids] - 1, 'clust_mediod'] = cluster_id
                    df_dailynews_first_cluster.loc[news_order[id_medoids] - 1, 'clust_cnt'] = str(len(clusters[key]))
                    df_dailynews_first_cluster.loc[news_order[id_medoids] - 1, 'clust_dst'] = str(
                        sum(dist_matrix[cluster_medoids[0], :]))

                    sse_value += sum(dist_matrix[cluster_medoids[0], :])
                    # 存储cluster_id 并将相应的title中的关键字作为标准查询
                    # std_query.write(str(cluster_id))
                    # std_query.write('\t')
                    std_query.write(titlevocab[id_medoids].strip())
                    std_query.write('\n')

                    if sum(dist_matrix[cluster_medoids[0], :]) > 0.1:
                        compare_dist.write(str(news_order[id_medoids] - 1))
                        compare_dist.write('\t')
                        compare_dist.write(str(cluster_id))
                        compare_dist.write('\t')
                        compare_dist.write(news_title[id_medoids].strip())
                        compare_dist.write('\t')
                        compare_dist.write(str(len(clusters[key])))
                        compare_dist.write('\t')
                        compare_dist.write(str(sum(dist_matrix[cluster_medoids[0], :])))
                        compare_dist.write('\n')
                        cmp_count += 1

                    # event_info 写入
                    event_info.loc[cluster_id - 1, 'event_id'] = cluster_id
                    event_info.loc[cluster_id - 1, 'news_id'] = news_order[id_medoids] - 1
                    event_info.loc[cluster_id - 1, 'event_title'] = \
                        news_title[id_medoids].strip()
                    event_info.loc[cluster_id - 1, 'num_mentions'] = \
                        df_dailynews_first_cluster.loc[
                            [news_order[id] - 1 for id in clusters[key]], 'ac_nummentions'].sum()
                    event_info.loc[cluster_id - 1, 'num_sources'] = \
                        df_dailynews_first_cluster.loc[
                            [news_order[id] - 1 for id in clusters[key]], 'ac_numsources'].sum()
                    event_info.loc[cluster_id - 1, 'num_articles'] = \
                        df_dailynews_first_cluster.loc[
                            [news_order[id] - 1 for id in clusters[key]], 'ac_numarticles'].sum()
                    event_info.loc[cluster_id - 1, 'goldsteinscale'] = \
                        df_dailynews_first_cluster.loc[
                            [news_order[id] - 1 for id in clusters[key]], 'ac_goldsteinscale'].mean()
                    event_info.loc[cluster_id - 1, 'event_weight'] = \
                        1.0 * len(clusters[key]) / (
                            sum(dist_matrix[cluster_medoids[0], :]) + 1.0)

                    # 保存cluster_id、news_title、关键词集
                    doc_clust.write('# ')
                    doc_clust.write(str(cluster_id))
                    doc_clust.write('\n')
                    doc_clust.write(news_title[id_medoids].strip())
                    doc_clust.write('\n')
                    doc_clust.write(totalvocab[id_medoids])
                    doc_clust.write('\n')
                    cluster_id += 1

                export_to_csv(event_info,
                              ex_filename=os.path.abspath(
                                  os.path.join(
                                      path_result_folder,
                                      event_info_output_file)),
                              sep='\t',
                              escapechar="'", index=False)
                doc_clust.write('\n')
                doc_clust.close()

                compare_dist.write(str(cmp_count))
                compare_dist.write('\n')
                compare_dist.close()

                std_query.close()

                print(str(int(max_dist)) + '\t' + str(sse_value) + '\t' + str(cmp_count) + '\t' + str(len(clusters)))

                # d_n_t_c_file = 'daily_news_twice_cluster_' + str(int(max_dist)) + '.csv'
                # export_to_csv(df_dailynews_first_cluster,
                #               ex_filename=os.path.abspath(
                #                   os.path.join(
                #                       path_result_folder,
                #                       d_n_t_c_file)),
                #               sep='\t',
                #               escapechar="'", index=False)
