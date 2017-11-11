__author__ = 'Nick Hirakawa'

import re
import pandas as pd


class CorpusParser:
    def __init__(self, filename):
        self.filename = filename
        self.regex = re.compile('^#\s*\d+')
        self.corpus = dict()

    def parse(self, len_k):
        # daily_news_cluster_file = '../data/newspaper_dataset/results/20170203/daily_news_twice_cluster_10.csv'
        # daily_news_cluster = pd.read_csv(daily_news_cluster_file, sep='\t')
        # id_news_event = daily_news_cluster.set_index('news_order')['cluster_step_two'].to_dict()
        with open(self.filename) as f:
            s = ''.join(f.readlines())
        blobs = s.split('# ')[1:]
        for x in blobs:
            text = x.split()
            docid = text.pop(0)
            # if id_news_event[int(docid)-1] == u'{na}':
            #     continue
            if int(docid) > len_k:
                break
            self.corpus[docid] = text

    def get_corpus(self):
        return self.corpus


class QueryParser:
    def __init__(self, filename):
        self.filename = filename
        self.queries = []

    def parse(self):
        with open(self.filename) as f:
            lines = ''.join(f.readlines())
        self.queries = [x.rstrip().split() for x in lines.split('\n')[:-1]]

    def get_queries(self):
        return self.queries


if __name__ == '__main__':
    qp = QueryParser('text/queries.txt')
    print qp.get_queries()
