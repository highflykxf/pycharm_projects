__author__ = 'Nick Hirakawa'

from parse import *
from query import QueryProcessor
import operator
import pandas as pd
from nltk.corpus import wordnet as wn
import sys

reload(sys)
sys.setdefaultencoding('utf-8')


def query_expansion(query_list):
    query_expanded = []
    for idx in range(len(query_list)):
        for syn_set in wn.synsets(query_list[idx]):
            query_expanded.extend(syn_set.lemma_names)
    return query_expanded


def main():
    qp = QueryParser(filename='../data/newspaper_dataset/results/20170201/query_11.txt')
    query_result_file = '../data/newspaper_dataset/query_result.txt'
    query_result = open(query_result_file, 'w')

    cp = CorpusParser(filename='../data/newspaper_dataset/results/20170201/doc_cluster_test.txt')
    clust_ranker_file = '../data/newspaper_dataset/results/20170201/daily_events_11.csv'
    clust_ranker = pd.read_csv(clust_ranker_file,sep='\t')

    daily_news_clusters_file = '../data/newspaper_dataset/results/20170201/daily_news_first_cluster.tsv'
    daily_news_clusters = pd.read_csv(daily_news_clusters_file, sep='\t')

    id_title_dict = clust_ranker.set_index('event_id')['event_title'].to_dict()
    id_ccoef_dict = clust_ranker.set_index('event_id')['event_weight'].to_dict()
    order_clust_dict = daily_news_clusters.set_index('news_order')['cluster_step_zero'].to_dict()

    qp.parse()
    queries_expand = []
    queries = qp.get_queries()
    for idx in range(len(queries)):
        queries_expand.append(query_expansion(queries[idx]))

    cp.parse()
    corpus = cp.get_corpus()
    proc = QueryProcessor(queries_expand, corpus)
    results = proc.run()
    qid = 0
    value_map = 0
    for result in results:
        sorted_x = sorted(result.iteritems(), key=operator.itemgetter(1))
        sorted_x.reverse()
        order = 0
        print str(qid), sorted_x[0][0]
        for i in sorted_x:
            tmp = (qid, id_title_dict[int(i[0])], order, i[1])
            # print '{:>1}\tQ0\t{:>4}\t{:>2}\t{:>12}\tNH-BM25'.format(*tmp)
            query_result.write('{:>1}\t{:>10}\t{:>2}\t{:>12}\tNH-BM25\n'.format(*tmp))
            order += 1
        qid += 1


if __name__ == '__main__':
    main()
