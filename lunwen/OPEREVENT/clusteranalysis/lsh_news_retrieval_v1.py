# -*- coding:utf-8 -*-
from lshash import LSHash
import os
import pickle

if __name__ == "__main__":
    path_local_script = os.path.dirname(os.path.realpath(__file__))
    # 所有新闻语料
    path_local_data = os.path.abspath(
        os.path.join(path_local_script,
                     os.pardir, "data", "newspaper_dataset"))
    path_result_folder = os.path.abspath(
        os.path.join(path_local_data, "results", "20170131"))
    with open(os.path.abspath(
            os.path.join(path_result_folder,
                         'tfidf_matrix.dat')), 'rb') as outfile:
        tfidf_matrix = pickle.load(outfile)
    tf_idf_matrix = tfidf_matrix.toarray()
    tf_idf_matrix_bak = tf_idf_matrix
    tf_idf_matrix_bak[tf_idf_matrix_bak>0] = 1
    lsh = LSHash(10, tf_idf_matrix_bak.shape[1])

    for idx in range(1000):
        lsh.index(tf_idf_matrix_bak[idx])
    res_0 = lsh.query(tf_idf_matrix_bak[1002])
    for idx in range(1000):
        if tf_idf_matrix_bak[idx] == res_0[0]:
            print idx
    res_1 = lsh.query(tf_idf_matrix_bak[1])

