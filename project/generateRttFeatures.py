# -*- coding:utf-8 -*-
import sys
import getopt
import networkx as nx
import numpy


def list_current_data_files(path_this):
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [join(path_this, f) for f in listdir(path_this) if isfile(join(path_this, f))]
    return onlyfiles


if __name__ == "__main__":
    options, args = getopt.getopt(sys.argv[1:], "", ["event_id=", "event_rtt_net_prefix=", "st_time=", "ed_time="])
    if options != []:
        for name, value in options:
            if name == '--event_id':
                EVENT_ID = value
            if name == '--event_rtt_net_prefix':
                EVENT_RTT_NET_PREFIX = value
            if name == '--st_time':
                ST_TIME = value
            if name == '--ed_time':
                ED_TIME = value

    for event_rtt_file in list_current_data_files(EVENT_RTT_NET_PREFIX):
        if event_rtt_file.split('\\')[-1].__eq__(str(EVENT_ID) + ".txt") is not True:
            continue
        with open(event_rtt_file, 'r') as f:
            rt_time = []
            for line in f:
                time = line.strip().split(' ')[-1]
                rt_time.append(int(time))

        array = numpy.array(rt_time)
        order = array.argsort()
        ranks = order.argsort()
        G = nx.DiGraph()

        with open(event_rtt_file, 'r') as f:
            for position, line in enumerate(f):
                time = line.strip().split(' ')[-1]
                if ST_TIME <= time < ED_TIME:
                    rtuid, uid = line.strip().split(' ')[0:2]
                    G.add_edge(uid, rtuid, time=ranks[position])
        if G.number_of_edges() <= 0:
            print 0, 0.0
            exit()
        edges, colors = zip(*nx.get_edge_attributes(G, 'time').items())
        # 计算节点的出度
        degree_values = nx.degree(G).values()
        if len(degree_values) == 0:
            average_degree = 0.0
        else:
            average_degree = sum(degree_values) / len(degree_values)
        max_degree_centrality = max(nx.degree_centrality(G).values())
        min_degree_centrality = min(nx.degree_centrality(G).values())
        max_closeness_centrality = max(nx.closeness_centrality(G).values())
        min_closeness_centrality = min(nx.closeness_centrality(G).values())
        max_clustering = max(nx.clustering(G.to_undirected()).values())
        average_clustering = nx.average_clustering(G.to_undirected())
        print str(average_degree) + " " + str(max_degree_centrality) + " " + str(min_degree_centrality) + " " + str(
            max_closeness_centrality) + " " + str(min_closeness_centrality) + " " + str(max_clustering) + " " + str(
            average_clustering)
