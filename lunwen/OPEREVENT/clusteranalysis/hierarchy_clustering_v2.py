# -*- coding: utf-8 -*-
from __future__ import print_function

import cPickle as pickle
import datetime
import os
import re
import string
import sys
import math
import operator
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from nltk.tag import pos_tag
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
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


def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    # print "1: ", filtered_tokens
    # print "2: ", len(filtered_tokens)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    # print "3: ", stems
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


if __name__ == "__main__":
    path_local_script = os.path.dirname(os.path.realpath(__file__))
    path_local_data = os.path.abspath(
        os.path.join(path_local_script,
                     os.pardir, "data", "newspaper_dataset", "20170131"))  # 根据需要修改日期
    path_result_data = os.path.abspath(
        os.path.join(path_local_script,
                     os.pardir, "data", "newspaper_dataset", "results", "20170131"))
    result_file = os.path.abspath(os.path.join(path_result_data, 'result.txt'))
    clust_rank_file = os.path.abspath(os.path.join(path_result_data, 'clust_rank.csv'))
    doc_clust_file = os.path.abspath(os.path.join(path_result_data, 'doc_cluster.txt'))
    f_out = open(result_file, "w")
    clust_rank = open(clust_rank_file, "w")
    doc_clust = open(doc_clust_file, "w")

    # read files 新闻order、url、title、contents、corpus
    news_corpus = []
    news_contents = []
    news_order = []
    news_url = []
    news_title = []
    num = 1
    news_num_test = 100
    while num <= news_num_test:
        name = "%06d" % num
        fileName = str(name) + ".txt"
        filename = os.path.abspath(os.path.join(path_local_data, fileName))
        # filter the file :  800B < size(file) < 20000B
        if 800 <= os.path.getsize(filename) <= 20000:
            news_order.append(num)
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
            temp_corpus = title + lines + title  # 变相增加title的权重
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

    # if os.path.isfile(path_result_data+os.sep+"vocab_tokenized.txt") and os.path.isfile(path_result_data+os.sep+"vocab_stemmed.txt") \
    #         and os.path.isfile(path_result_data+"\\"+"vocab_pos_filtered.txt"):
    #     with open(path_result_data+os.sep+'vocab_tokenized.txt', 'r') as f:
    #         for line in f.readlines():
    #             totalvocab_tokenized.append(line.strip('\n'))
    #     with open(path_result_data+os.sep+'vocab_stemmed.txt', 'r') as f:
    #         for line in f.readlines():
    #             totalvocab_stemmed.append(line.strip('\n'))
    #     with open(path_result_data+os.sep+'vocab_pos.txt', 'r') as f:
    #         for line in f.readlines():
    #             totalvocab_pos.append(line.strip('\n'))
    # else:
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

        # with open(path_result_data+os.sep+'vocab_tokenized.txt', 'w') as file_vocab_tokenized:
        #     for i in range(len(totalvocab_tokenized)):
        #         file_vocab_tokenized.write(totalvocab_tokenized[i])
        #         file_vocab_tokenized.write("\n")
        # with open(path_result_data+os.sep+'vocab_stemmed.txt', 'w') as file_vocab_stemmed:
        #     for i in range(len(totalvocab_stemmed)):
        #         file_vocab_stemmed.write(totalvocab_stemmed[i])
        #         file_vocab_stemmed.write("\n")
        # with open(path_result_data+os.sep+'vocab_pos.txt', 'r') as file_vocab_pos:
        #     for i in range(len(totalvocab_pos)):
        #         file_vocab_pos.write(totalvocab_pos[i])
        #         file_vocab_pos.write("\n")

    # vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index=totalvocab_stemmed)
    # print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

    # 读取原始的结构化的每日新闻数据集, 根据一些属性先行进行聚类
    daily_news = os.path.abspath(os.path.join(path_local_data, 'daily_news.tsv'))
    df_dailynews = pd.read_csv(daily_news, sep='\t', encoding='utf-8')
    df_news_test = df_dailynews.loc[0:news_num_test-1][news_order]
    df_news_test['news_order'] = pd.Series(news_order)
    # 转化为tf-idf document-词项矩阵
    tfidf_vectorizer = TfidfVectorizer(max_df=0.5,
                                       max_features=10000,
                                       min_df=1,
                                       stop_words=stopwords,
                                       use_idf=True,
                                       tokenizer=tokenize_only,
                                       ngram_range=(1, 1))
    # binary_vectorizer = HashingVectorizer(
    #     encoding='utf-8',
    #     stop_words='english',
    #     decode_error='ignore',
    #     tokenizer=tokenize_and_stem,
    #     ngram_range=(1, 2),
    #     binary=True
    # )
    tfidf_matrix = tfidf_vectorizer.fit_transform(totalvocab)
    # binary_matrix = binary_vectorizer.fit_transform(news_corpus)

    # with open('tfidf_matrix.dat', 'wb') as outfile:
    #     pickle.dump(tfidf_matrix, outfile, pickle.HIGHEST_PROTOCOL)

    words_weights = tfidf_matrix.toarray()
    sim_mat = np.zeros(words_weights.shape)
    terms = tfidf_vectorizer.get_feature_names()
    keywords_list = []
    # 选取每篇文档的top-k个关键词的tf-idf值
    top_k = 10
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

    # for index in keywords_list:
    #     print(index)
    # 只显示层次聚类的最后一百个进行显示，对大于max_d的簇进行标注
    max_d = 1.0
    k = 100
    # program_end = datetime.datetime.now()
    Z = linkage(sim_mat, 'ward')
    # last_clusters = Z[-20:,2]
    # last_rev = last_clusters[::-1]
    # idxs = np.arange(1, len(last_clusters)+1)
    # plt.plot(idxs, last_rev)
    # acceleration = np.diff(last_clusters, 2)
    # acceleration_rev = acceleration[::-1]
    # plt.plot(idxs[:-2]+1, acceleration_rev)
    # plt.show()
    # k = int(acceleration_rev.argmax() + 2)
    fancy_dendrogram(
        Z,
        truncate_mode='lastp',
        p=k,
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,
        annotate_above=1.0,  # useful in small plots so annotations don't overlap
        max_d=max_d
    )
    plt.savefig(os.path.abspath(os.path.join(path_result_data, "hcluster_1.png")), dpi=800)
    # clusters = fcluster(Z, k, criterion='maxclust')
    # print(clusters)
    # plt.show()

    max_dist = 1.0  # 簇内最大距离,需要根据实际情况调整
    clusters = extract_clusters(Z, max_dist, len(news_corpus))

    # 输出聚类结果
    dict_cid_weight = {}
    cluster_dict = {}
    cluster_id = 1
    for key in clusters:
        f_out.write("=============================================\n")
        for id in clusters[key]:
            f_out.write(str(id))
            f_out.write('\t')
            f_out.write(news_title[id].strip())
            f_out.write('\t')
            f_out.write(news_url[id].strip())
            f_out.write('\n')
            # print(id, news_title[id], news_url[id])
        # 过滤掉孤立的点
        if len(clusters[key]) >= 2:
            # 找到聚类的medoid（news ID）以及该质心的title、簇间距离、簇内新闻数目、
            cluster_data = sim_mat[clusters[key], :]
            dist_matrix = pairwise_distances(cluster_data, metric='euclidean')
            cluster_medoids, cluster_class = kmedoids.kMedoids(dist_matrix, 1)
            id_medoids = clusters[key][cluster_medoids[0]]
            f_out.write(str(id_medoids))
            f_out.write('\t')
            f_out.write(str(cluster_id))
            f_out.write('\t')
            f_out.write(news_title[id_medoids].strip())
            f_out.write('\t')
            f_out.write(str(len(clusters[key])))
            f_out.write('\t')
            f_out.write(str(sum(dist_matrix[cluster_medoids[0], :])))
            f_out.write('\n')
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
