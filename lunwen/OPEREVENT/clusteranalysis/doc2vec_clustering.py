# -*- coding: utf-8 -*-
from __future__ import print_function

import math
import os
import re
import string
import sys
import csv
import shutil
import matplotlib.pyplot as plt
import pickle
import smart_open
from gensim.models.doc2vec import Doc2Vec
import nltk
import pandas as pd
import gensim
from nltk.stem.snowball import SnowballStemmer
from nltk.tag import pos_tag
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer, TfidfTransformer
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import kmedoids

reload(sys)
sys.setdefaultencoding('utf-8')


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


# def tokenize_and_stem(text):
#     # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
#     tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
#     filtered_tokens = []
#     # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
#     for token in tokens:
#         if re.search('[a-zA-Z]', token):
#             filtered_tokens.append(token)
#     # print "1: ", filtered_tokens
#     # print "2: ", len(filtered_tokens)
#     stems = [stemmer.stem(t) for t in filtered_tokens]
#     # print "3: ", stems
#     return stems


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
    return data_frame, clust_id, num_count


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


def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="utf8") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])


if __name__ == "__main__":
    path_local_script = os.path.dirname(os.path.realpath(__file__))
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
        # 读取原始的结构化的每日新闻数据集, 根据一些属性(相同的url，相同的[主客体名字、类型、地理位置])先行进行聚类
        path_tsv = os.path.abspath(os.path.join(data_folder, "daily_news.tsv"))
        df_dailynews = pd.read_csv(path_tsv, sep='\t', encoding='utf-8')
        df_dailynews = df_dailynews.replace(u'{na}', '')
        df_dailynews = df_dailynews.fillna("")
        df_dailynews = df_dailynews.iloc[0:100]
        df_dailynews['news_order'] = pd.Series(range(0, len(df_dailynews)))
        df_dailynews_zero_cluster, clust_zero_num, num_count = cluster_step_zero(df_dailynews)

        daily_num_count_file = file(path_daily_num_count, 'wb')
        for key,value in num_count.items():
            daily_num_count_file.write(str(key))
            daily_num_count_file.write('\t')
            daily_num_count_file.write(str(value))
            daily_num_count_file.write('\n')
        # pickle.dump(num_count, daily_num_count_file)
        daily_num_count_file.close()

        df_dailynews_first_cluster = cluster_step_one(df_dailynews_zero_cluster)
        export_to_csv(df_dailynews_first_cluster,
                      ex_filename=os.path.abspath(
                          os.path.join(
                              path_result_folder,
                              "daily_news_first_cluster.tsv")),
                      sep='\t',
                      escapechar="'",
                      index=False)
        # 根据内容相似度来聚类
        # read files 新闻order、url、title、contents、corpus
        news_corpus = []
        news_contents = []
        news_order = []
        news_url = []
        news_title = []
        num = 1
        # news_num_test = df_dailynews.shape[0]
        news_num_test = 100
        clust_flag = np.zeros(clust_zero_num, dtype=int)
        while num <= news_num_test:
            name = "%06d" % num
            fileName = str(name) + ".txt"
            filename = os.path.abspath(os.path.join(data_folder, fileName))
            # filter the file :  800B < size(file) < 20000B
            if os.path.isfile(filename) is False:
                continue
            cluster_zero_id = df_dailynews_first_cluster.loc[num-1, 'cluster_step_zero']
            cluster_zero_id = int(cluster_zero_id)
            cluster_zero_num = df_dailynews_first_cluster.loc[num-1, 'cluster_zero_num']
            cluster_zero_num = int(cluster_zero_num)
            if (800 <= os.path.getsize(filename) <= 20000) and \
                    (clust_flag[cluster_zero_id-1] == 0) and \
                    (cluster_zero_num >=2):
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
                # temp_corpus = title + lines + title  # 变相增加title的权重
                temp_corpus = lines
                news_corpus.append(temp_corpus)
            num += 1

        # 词干还原、去除停用词、命名实体识别
        # load nltk's English stopwords as variable called 'stopwords'
        stopwords = nltk.corpus.stopwords.words('english')
        # load nltk's SnowballStemmer as variabled 'stemmer'
        stemmer = SnowballStemmer("english")
        totalvocab_stemmed = []
        totalvocab_tokenized = []
        totalvocab_pos = []
        totalvocab = []

        for document in news_corpus:
            tmpcorpus = []
            # allwords_stemmed = tokenize_and_stem(document)
            # totalvocab_stemmed.append(allwords_stemmed)
            allwords_tokenized = tokenize_only(document)
            # totalvocab_tokenized.append(allwords_tokenized)
            allwords_posed = proppers_POS(document)
            # totalvocab_pos.append(allwords_posed)
            # tmpcorpus.extend(allwords_stemmed)
            tmpcorpus.extend(allwords_tokenized)
            tmpcorpus.extend(allwords_posed)
            totalvocab.append(' '.join(tmpcorpus))
        with open(os.path.abspath(
                os.path.join(
                    path_result_folder,
                    "vocab_tokenized.txt")), 'w') as file_vocab_tokenized:
            for i in range(len(totalvocab)):
                file_vocab_tokenized.write(totalvocab[i] + '\n')

        # 转化为doc_id*doc2vec 矩阵
        documents = list(read_corpus(os.path.abspath(
                os.path.join(path_result_folder, "vocab_tokenized.txt"))))
        model_loaded = Doc2Vec.load("wiki_text_5e5_model_0")
        corpus_vectors = np.zeros((len(documents), 200))
        for doc_id in range(len(documents)):
            article_vector = model_loaded.infer_vector(documents[doc_id].words)
            corpus_vectors[doc_id] = article_vector

        Z = linkage(corpus_vectors, 'ward')
        # 簇内最大距离,需要根据实际情况调整
        for max_dist in range(30, 100, 1):
            sse_value = 0.0
            clusters = extract_clusters(Z, 0.15*max_dist, len(news_corpus))

            # 输出聚类结果
            res_file = 'doc2vec_result_'+str(int(max_dist))+ '.txt'
            c_r_file = 'doc2vec_clust_rank_' +str(int(max_dist))+ '.txt'
            d_c_file = 'doc2vec_doc_cluster_' + str(int(max_dist)) + '.txt'
            cmp_file = 'doc2vec_cmp_' + str(int(max_dist)) + '.txt'
            dist_file = 'doc2vec_dst_' + str(int(max_dist)) + '.txt'
            result_file = os.path.abspath(os.path.join(path_result_folder, res_file))
            clust_rank_file = os.path.abspath(os.path.join(path_result_folder, c_r_file))
            doc_clust_file = os.path.abspath(os.path.join(path_result_folder, d_c_file))
            compare_file = os.path.abspath(os.path.join(path_result_folder, cmp_file))
            clust_dist_file = os.path.abspath(os.path.join(path_result_folder, dist_file))
            f_out = open(result_file, "w")
            clust_rank = open(clust_rank_file, "w")
            doc_clust = open(doc_clust_file, "w")
            compare_dist = open(compare_file, "w")
            clust_dist = open(clust_dist_file, "w")

            dict_cid_weight = {}
            cluster_dict = {}
            cluster_id = 1
            cmp_count = 0
            c_flag = True
            for key in clusters:
                f_out.write("=============================================\n")
                for id in clusters[key]:
                    f_out.write(str(news_order[id] - 1))
                    f_out.write('\t')
                    f_out.write(news_title[id].strip())
                    f_out.write('\t')
                    f_out.write(news_url[id].strip())
                    f_out.write('\n')
                    # 将基于语义相似度的cluster结果保存
                    df_dailynews_first_cluster.loc[news_order[id] - 1, 'cluster_step_two'] = cluster_id

                # 过滤掉孤立的点
                # if len(clusters[key]) >= 2:
                # 找到聚类的medoid（news ID）以及该质心的title、簇间距离、簇内新闻数目、
                cluster_data = corpus_vectors[clusters[key], :]
                dist_matrix = pairwise_distances(cluster_data, metric='euclidean')
                cluster_medoids, cluster_class = kmedoids.kMedoids(dist_matrix, 1)
                id_medoids = clusters[key][cluster_medoids[0]]

                sse_value += sum(dist_matrix[cluster_medoids[0], :])

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
                f_out.write(str(news_order[id_medoids] - 1))
                f_out.write('\t')
                f_out.write(str(cluster_id))
                f_out.write('\t')
                f_out.write(news_title[id_medoids].strip())
                f_out.write('\t')
                f_out.write(str(len(clusters[key])))
                f_out.write('\t')
                f_out.write(str(sum(dist_matrix[cluster_medoids[0], :])))
                f_out.write('\n')
                clust_dist.write(str(cluster_id))
                clust_dist.write('\t')
                clust_dist.write(str(len(clusters[key])))
                clust_dist.write('\t')
                clust_dist.write(str(sum(dist_matrix[cluster_medoids[0], :])))
                clust_dist.write('\n')
                # 保存cluster_id、news_title、关键词集
                doc_clust.write('# ')
                doc_clust.write(str(cluster_id))
                # cluster_id、title词典
                cluster_dict[cluster_id] = news_title[id_medoids].strip()
                doc_clust.write('\n')
                doc_clust.write(news_title[id_medoids].strip())
                doc_clust.write('\n')
                doc_clust.write(totalvocab[id_medoids])
                doc_clust.write('\n')
                # cluster_id、簇凝聚系数词典
                dict_cid_weight[cluster_id] = 1.0 * math.exp(len(clusters[key])) / (
                    sum(dist_matrix[cluster_medoids[0], :]) + 1.0)
                cluster_id += 1

            print (str(int(max_dist))+'\t'+str(sse_value)+'\t'+str(cmp_count)+'\t'+str(len(clusters)))

            f_out.write("***********************************\n")
            dict_sorted = sorted(dict_cid_weight.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
            for key, value in dict_sorted:
                # print (str(key.strip()), "********", str(value), "**********", str(cluster_dict[key.strip()]))
                # clust_rank中保存 cluster_id、簇凝聚系数、cluster_title
                clust_rank.write(str(key))
                clust_rank.write('\t')
                clust_rank.write(str(value))
                clust_rank.write('\t')
                clust_rank.write(str(cluster_dict[int(key)]))
                clust_rank.write('\n')
                # print(key, value)
            clust_rank.close()
            doc_clust.close()
            f_out.close()
            clust_dist.close()
            compare_dist.write(str(cmp_count))
            compare_dist.write('\n')
            compare_dist.close()

            d_n_t_c_file = 'doc2vec_daily_news_twice_cluster_' + str(int(max_dist)) + '.csv'
            export_to_csv(df_dailynews_first_cluster,
                          ex_filename=os.path.abspath(
                              os.path.join(
                                  path_result_folder,
                                  d_n_t_c_file)),
                          sep='\t',
                          escapechar="'", index=False)
