# -*- coding:utf-8 -*-
import nltk
import codecs
from gensim.models import LdaModel
from gensim import corpora,models,similarities
from gensim.corpora import Dictionary

# 词干还原、去除停用词、命名实体识别
# load nltk's English stopwords as variable called 'stopwords'
train_corpus = []
stopwords = nltk.corpus.stopwords.words('english')
with codecs.open('vocab_tokenized.txt', 'r', encoding='utf8') as fp:
    for line in fp:
        line = line.split()
        train_corpus.append([w for w in line if w not in stopwords])
dictionary = corpora.Dictionary(train_corpus)
corpus = [dictionary.doc2bow(text) for text in train_corpus]
lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=100)

