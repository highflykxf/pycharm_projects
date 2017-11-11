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
    propernouns = [word for word, pos in tagged if pos == 'NNP' and pos == 'NNPS']
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


# here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed
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
    program_start = datetime.datetime.now()
    path_local_script = os.path.dirname(os.path.realpath(__file__))
    path_local_data = os.path.abspath(
        os.path.join(path_local_script,
                     os.pardir, "data",
                     os.curdir, "newspaper_dataset",
                     os.curdir, "20170131")) # 根据需要修改日期
    output_file = 'result.txt'
    f_out = open(output_file, "w")
    # read files
    news_corpus = []
    news_contents = []
    news_order = []
    news_url = []
    news_title = []
    num = 1
    while num <= 100:
        name = "%06d" % num
        fileName = path_local_data + os.curdir + str(name) + ".txt"
        # filter the file :  800B < size(file) < 20000B
        if os.path.getsize(fileName) > 800 and os.path.getsize(fileName) <= 20000:
            news_order.append(num)
            lines = ""
            title = ""
            url_line = 1
            title_line = 2
            with open(fileName, 'r') as f:
                index = 1
                for line in f:
                    if line != "\n" and index > 2:
                        lines = lines + line
                    elif line != "\n" and index == 1:
                        news_url.append(line)
                        index += 1
                    elif line != "\n" and index == 2:
                        title = line
                        news_title.append(line)
                        index += 1
            news_contents.append(lines)
            temp_corpus = title + lines
            news_corpus.append(temp_corpus)
        num += 1
    # load nltk's English stopwords as variable called 'stopwords'
    stopwords = nltk.corpus.stopwords.words('english')

    # load nltk's SnowballStemmer as variabled 'stemmer'
    stemmer = SnowballStemmer("english")
    totalvocab_stemmed = []
    totalvocab_tokenized = []
    totalvocab_pos_filtered = []

    if os.path.isfile("vocab_tokenized.txt") and os.path.isfile("vocab_stemmed.txt") \
            and os.path.isfile("vocab_pos_filtered.txt"):
        with open('vocab_tokenized.txt', 'r') as f:
            for line in f.readlines():
                totalvocab_tokenized.append(line.strip('\n'))
        with open('vocab_stemmed.txt', 'r') as f:
            for line in f.readlines():
                totalvocab_stemmed.append(line.strip('\n'))
        with open('vocab_pos_filtered.txt', 'r') as f:
            for line in f.readlines():
                totalvocab_pos_filtered.append(line.strip('\n'))
    else:
        for document in news_corpus:
                allwords_stemmed = tokenize_and_stem(document)
                totalvocab_stemmed.extend(allwords_stemmed)
                allwords_tokenized = tokenize_only(document)
                totalvocab_tokenized.extend(allwords_tokenized)
                # allwords_posed = proppers_POS(document)
                # totalvocab_pos_filtered.extend(allwords_posed)

        with open('vocab_tokenized.txt', 'w') as file_vocab_tokenized:
            for i in range(len(totalvocab_tokenized)):
                file_vocab_tokenized.write(totalvocab_tokenized[i] + '\n')
        with open('vocab_stemmed.txt', 'w') as file_vocab_stemmed:
            for i in range(len(totalvocab_stemmed)):
                file_vocab_stemmed.write(totalvocab_stemmed[i] + '\n')

    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index=totalvocab_stemmed)
    print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
    # print(vocab_frame.head())
    tfidf_vectorizer = TfidfVectorizer(max_df=0.5,
                                       max_features=10000,
                                       min_df=1,
                                       stop_words=stopwords,
                                       use_idf=True,
                                       tokenizer=tokenize_and_stem,
                                       ngram_range=(1, 1))
    # binary_vectorizer = HashingVectorizer(
    #     encoding='utf-8',
    #     stop_words='english',
    #     decode_error='ignore',
    #     tokenizer=tokenize_and_stem,
    #     ngram_range=(1, 2),
    #     binary=True
    # )

    tfidf_construct_starttime = datetime.datetime.now()
    # construce tf-idf matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(news_corpus)
    # tfidf_matrix = binary_vectorizer.fit_transform(news_corpus)
    # print(tfidf_matrix.shape)

    tfidf_construct_endtime = datetime.datetime.now()
    # print("tfidf construct time: ", (tfidf_construct_endtime - tfidf_construct_starttime).seconds)

    with open('tfidf_matrix.dat', 'wb') as outfile:
        pickle.dump(tfidf_matrix, outfile, pickle.HIGHEST_PROTOCOL)

    words_weights = tfidf_matrix.toarray()
    sim_mat = np.zeros(words_weights.shape)
    terms = tfidf_vectorizer.get_feature_names()
    keywords_list = []
    top_k = 7
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

    max_d = 0.2
    program_end = datetime.datetime.now()
    Z = linkage(sim_mat, 'ward')
    fancy_dendrogram(
        Z,
        truncate_mode='lastp',
        p=50,
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,
        annotate_above=0.6,  # useful in small plots so annotations don't overlap
        max_d=max_d
    )
    plt.savefig("hcluster_1.png", dpi=800)
    # clusters = fcluster(Z, max_d, criterion='distance')
    # print(clusters)
    # plt.show()

    clusters = extract_clusters(Z, 1.400, len(news_corpus))
    dict_title_weight = {}
    for key in clusters:
        f_out.write("=============================================")
        for id in clusters[key]:
            f_out.write(str(id))
            f_out.write(news_title[id])
            f_out.write(news_url[id])
            # print(id, news_title[id], news_url[id])
        if len(clusters[key]) >= 2:
            cluster_data = sim_mat[clusters[key], :]
            dist_matrix = pairwise_distances(cluster_data, metric='euclidean')
            cluster_medoids, cluster_class = kmedoids.kMedoids(dist_matrix, 1)
            id_medoids = clusters[key][cluster_medoids[0]]
            f_out.write(str(id_medoids))
            f_out.write(news_title[id_medoids])
            f_out.write(str(len(clusters[key])))
            f_out.write(str(sum(dist_matrix[cluster_medoids[0], :])))
            # print(id_medoids, news_title[id_medoids], len(clusters[key]), sum(dist_matrix[cluster_medoids[0], :]))
            # print(news_url[id_medoids])
            # print(len(clusters[key]),sum(dist_matrix[cluster_medoids[0],:]))
            dict_title_weight[news_title[id_medoids]] = 1.0 * math.exp(len(clusters[key])) / (
                sum(dist_matrix[cluster_medoids[0], :]) + 1.0)
    dict_sorted = sorted(dict_title_weight.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    for key, value in dict_sorted:
        f_out.write(str(key))
        f_out.write(str(value))
        # print(key, value)
    f_out.close()
