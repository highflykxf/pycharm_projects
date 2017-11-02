# coding:utf-8
import matplotlib.pyplot as plt
import networkx as nx
import numpy
import numpy as np
import pandas as pd
import re


def list_current_data_files(path_this):
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [join(path_this, f) for f in listdir(path_this) if isfile(join(path_this, f))]
    return onlyfiles


def drawRttNet(file_path):
    with open(file_path, 'r') as f:
        rt_time = []
        for line in f:
            time = line.strip().split(' ')[-1]
            # print time
            # day = time[0:4] + time[5:7] + time[8:10]
            # # print day
            # # hms= time[9:17].replace(':', '')
            # hm = time[11:19].replace('-', '')
            # time = int(day + hm)
            rt_time.append(int(time))
            # 计算转发时间的先后顺序
    array = numpy.array(rt_time)
    order = array.argsort()
    ranks = order.argsort()
    G = nx.Graph()
    with open(file_path, 'r') as f:
        for position, line in enumerate(f):
            # print position, line
            rtuid, uid = line.strip().split(' ')[0:2]
            G.add_edge(uid, rtuid, time=ranks[position])
    edges, colors = zip(*nx.get_edge_attributes(G, 'time').items())
    # degree=G.degree()#计算节点的出度
    # degree=G.degree()#计算节点的出度
    degree = G.degree(G)
    # closenesss = nx.closeness_centrality(G)
    # betweenness = nx.betweenness_centrality(G)
    print str(G.number_of_edges()) + " " + str(nx.average_clustering(G))
    # if nx.is_connected(G):
    #     print str(nx.average_shortest_path_length(G))
    # str(nx.average_clustering(G)) + " " + \
    # str(nx.strongly_connected_components(G.to_undirected())) + " " + \
    # print g.diameter(G) # 2
    # path_length = nx.all_pairs_shortest_path_length(G)
    pos = nx.spring_layout(G)  # 设置网络的布局
    # fig = plt.figure(figsize=(80, 60),facecolor='white')
    # pos=nx.spring_layout(G) #设置网络的布局

    fig = plt.figure(figsize=(20, 10), facecolor='white')
    nx.draw(G, pos, nodelist=degree.keys(),
            node_size=[v * 2 for v in degree.values()], node_color='orange',
            node_shape='o', cmap=plt.cm.gray,
            edgelist=edges, edge_color='gray', width=0.5,
            with_labels=False, arrows=False)
    event_id = file_path.split("\\")[-1][:-4]
    plt.title("Event " + event_id)
    # plt.show()
    plt_out_path = "E:\\workspace\\eclipse_workspace\\event_tendency_predict\\datasets\\event_rtt_net_graph\\" + event_id + ".png"
    plt.savefig(plt_out_path)


def drawTendency(file_path):
    dig_list = re.findall(r'\d+', file_path)
    df = pd.read_table(file_path, header=None, sep='\s+', names=['event_date_st', 'cnt'])
    df['event_date_st'] = df['event_date_st'].astype('str')
    plt.figure()
    plt.plot(np.arange(df.shape[0]), df['cnt'])
    # plt.xticks(df['event_date_st'])
    # plt.xlabel('event time start')
    # plt.ylabel('cnt num')
    plt.title("Event Tendency: " + dig_list[0])
    plt_out_path = "E:\\workspace\\eclipse_workspace\\event_tendency_predict\\datasets\\event_tendency_graph\\" + \
                   dig_list[0] + "_tendency.png"
    plt.savefig(plt_out_path)



def analysisSegment(file_path):
    dig_list = re.findall(r'\d+', file_path)
    df = pd.read_table(file_path, header=None, sep='\s+',
                       names=['event_date_st', 'event_date_ed', 'avg_val', 'sum_k', 'max_k', 'min_k', 'mean_k', 'label'])
    df['event_date_st'] = df['event_date_st'].astype('str')
    df['event_date_ed'] = df['event_date_ed'].astype('str')
    df['max_avg_ratio'] = df['max_k'] / df['avg_val']
    df['min_avg_ratio'] = df['min_k'] / df['avg_val']
    df['max_avg_ratio'] = df['max_avg_ratio'].apply(lambda x: 5.0 if x > 5.0 else x)
    df['min_avg_ratio'] = df['min_avg_ratio'].apply(lambda x: 1.0 if x > 1.0 else x)
    # print df.shape[0]
    plt.figure()
    # plt.plot(np.arange(df.shape[0]), df['max_avg_ratio'])
    plt.plot(np.arange(df.shape[0]), df['min_avg_ratio'])
    # plt.plot(np.arange(df.shape[0]), df['mean_k'] / df['avg_val'])
    # plt.plot(np.arange(df.shape[0]), df['cnt'])
    # plt.xticks(df['event_date_st'])
    # plt.xlabel('event time start')
    # plt.ylabel('cnt num')
    plt.title("analysis segment: " + dig_list[0])
    plt_out_path = "E:\\workspace\\eclipse_workspace\\event_tendency_predict\\datasets\\event_segment_analysis\\" + \
                   dig_list[0] + "_segment_analysis.png"
    plt.savefig(plt_out_path)
    plt.clf()
    return df[['event_date_st', 'event_date_ed', 'avg_val', 'sum_k', 'max_k', 'min_k', 'mean_k', 'label']]


if __name__ == "__main__":
    # event_rtt_file_path = "E:\\workspace\\eclipse_workspace\\event_tendency_predict\\datasets\\event_rtt_net"
    # e_id = 1
    # for event_rtt_file in list_current_data_files(event_rtt_file_path):
    #     # if e_id < 10:
    #     #     e_id += 1
    #     #     continue
    #     # if event_rtt_file.__contains__("content_corpus.txt") is not True:
    #     count = len(open(event_rtt_file, 'rU').readlines())
    #     if count <= 0:
    #         continue
    #     drawRttNet(event_rtt_file)

    event_tendency_file_path = "E:\\workspace\\eclipse_workspace\\event_tendency_predict\\datasets\\event_tendency\\"

    df_tot = pd.DataFrame(columns=['event_date_st', 'event_date_ed', 'avg_val', 'sum_k', 'max_k', 'min_k', 'mean_k', 'label'])
    for event_tendency_file in list_current_data_files(event_tendency_file_path):
        count = len(open(event_tendency_file, 'rU').readlines())
        if count <= 0:
            continue
        if event_tendency_file.__contains__("raw") is True:
            drawTendency(event_tendency_file)
        if event_tendency_file.__contains__("segment") is True:
            df_tot = pd.concat([df_tot, analysisSegment(event_tendency_file)])

    print df_tot.groupby('label').size()