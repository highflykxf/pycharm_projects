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
    qp = QueryParser(filename='../data/newspaper_dataset/results/20170203/query_12_10.txt')
    query_result_file = '../data/newspaper_dataset/results/20170203/terminal_query_result_10_10.txt'
    query_result = open(query_result_file, 'w')

    cp = CorpusParser(filename='../data/newspaper_dataset/results/20170203/news_order_vocab.txt')
    clust_event_file = '../data/newspaper_dataset/results/20170203/daily_events_12_10.csv'
    clust_event = pd.read_csv(clust_event_file, sep='\t')

    daily_news_cluster_file = '../data/newspaper_dataset/results/20170203/daily_news_twice_cluster_10.csv'
    daily_news_cluster = pd.read_csv(daily_news_cluster_file, sep='\t')

    id_title_dict = clust_event.set_index('event_id')['event_title'].to_dict()
    id_news_dict = clust_event.set_index('event_id')['news_id'].to_dict()
    id_influence_dict = clust_event.set_index('event_id')['goldsteinscale'].to_dict()
    id_news_event = daily_news_cluster.set_index('news_order')['cluster_step_two'].to_dict()

    qp.parse()
    queries_expand = []
    queries = qp.get_queries()
    queries_sub = queries
    q_id = 0
    # for event_id, news_id in id_news_dict.iteritems():
    #     # if event_id > 1000:
    #     #     break
    #     # a1_name = clust_event.loc[news_id,'a1_name']
    #     # queries_sub[event_id-1] = []
    #     queries_sub[event_id-1].append(daily_news_cluster.loc[int(news_id), 'a1_name'])
    #     queries_sub[event_id-1].append(daily_news_cluster.loc[int(news_id), 'a2_name'])
    #     queries_sub[event_id-1].append(daily_news_cluster.loc[int(news_id), 'a1_type1'])
    #     queries_sub[event_id-1].append(daily_news_cluster.loc[int(news_id), 'a2_type1'])
    #     queries_sub[event_id-1].append(daily_news_cluster.loc[int(news_id), 'a1_geo_fullname'])
    #     queries_sub[event_id-1].append(daily_news_cluster.loc[int(news_id), 'a2_geo_fullname'])

    for idx in range(len(queries_sub)):
        queries_expand.append(query_expansion(queries_sub[idx]))
        # if float(id_influence_dict[idx+1]) > 0.0:
        #     queries_expand[idx].append('+')
        # else:
        #     queries_expand[idx].append('-')

    # k = 20
    for k in [1, 5, 10, 20]:
        cp.parse(1000000)
        corpus = cp.get_corpus()
        # for doc_id in corpus:
            # if float(id_influence_dict[int(id_news_event[int(doc_id)])]) > 0.0:
            #     corpus[doc_id].append('+')
            # else:
            #     corpus[doc_id].append('-')
        corpus_sub = corpus
        proc = QueryProcessor(queries_expand, corpus_sub)
        results = proc.run()
        qid = 0
        value_map_k = 0
        for result in results:
            sorted_x = sorted(result.iteritems(), key=operator.itemgetter(1))
            sorted_x.reverse()
            order = 1
            # for i in sorted_x:
            #     tmp = (qid, order, id_news_dict[int(i[0])], id_title_dict[int(i[0])], i[1])
            #     query_result.write('{:>1}\t{:>2}\t{:>2}\t{:>20}\t{:>2}\tNH-BM25\n'.format(*tmp))
            #     order += 1
            # order = 1
            for i in sorted_x:
                # print i[0],id_news_event[int(i[0])]
                if order <= k and qid + 1 == int(round(float(id_news_event[int(i[0])-1]))):
                    value_map_k += 1
                    break
                elif order > k:
                    break
                order += 1
            qid += 1
        print 1.0 * value_map_k / len(results)


if __name__ == '__main__':
    main()
