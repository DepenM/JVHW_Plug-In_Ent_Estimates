import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from est_MI import est_MI_JVHW, est_MI_MLE

def Prim_MST(weights):
    vertices = weights.shape[0]
    adj_matrix = np.zeros((vertices, vertices))
    init_weights = np.zeros((vertices)) + np.inf
    for i in range(len(weights[0])):
        if i != 0:
            init_weights[i] = weights[0, i]

    conn_vertex = [0]*vertices
    MST_vertices = 1

    while MST_vertices < vertices:
        index = np.argmin(init_weights)
        init_weights[index] = np.inf
        adj_matrix[conn_vertex[index], index] = 1
        adj_matrix[index, conn_vertex[index]] = 1
        MST_vertices += 1

        for i in range(len(weights[index])):
            if init_weights[i] != np.inf and weights[index, i] < init_weights[i]:
                    init_weights[i] = weights[index, i]
                    conn_vertex[i] = index

    return adj_matrix

def plot_tree(adj_matrix, names):
    vertices = adj_matrix.shape[0]
    dict = {}
    for i in range(vertices):
        dict[i] = names[i]
    G = nx.Graph()
    G.add_nodes_from(list(range(vertices)))
    for i in range(vertices):
        for j in range(i):
            if adj_matrix[j, i] == 1:
                G.add_edge(j, i)

    nx.draw_networkx(G, with_labels=True, labels=dict)
    plt.show()

def load_data(path):
    input = pd.read_csv(path)
    colnames = list(input.columns)
    vals = input.values
    return colnames, vals

def get_MLE_JVHW_est(vals, colnames):
    vals2 = np.zeros((vals.shape[0], vals.shape[1])) - 1
    uniq_vals = []
    no_vars = len(colnames)
    lens = np.zeros((no_vars)).astype(int)
    for i in range(vals.shape[1]):
        uniq_vals.append(np.unique(vals[:,i]))
        lens[i] = len(np.unique(vals[:,i]))

    for i in range(vals.shape[1]):
        dict = {}
        for j in range(len(uniq_vals[i])):
            dict[uniq_vals[i][j]] = j
        for j in range(vals.shape[0]):
            vals2[j, i] = dict[vals[j, i]]

    pairwise_MI_MLE = np.zeros((no_vars, no_vars))
    pairwise_MI_JVHW = np.zeros((no_vars, no_vars))
    for i in range(vals.shape[1]):
        for j in range(i):
            pairwise_MI_MLE[j, i] = est_MI_MLE(vals2[:, i], vals2[:, j])
            pairwise_MI_MLE[i, j] = pairwise_MI_MLE[j, i]
            pairwise_MI_JVHW[j, i] = est_MI_JVHW(vals2[:, i], vals2[:, j])
            pairwise_MI_JVHW[i, j] = pairwise_MI_JVHW[j, i]

    """
    ind_prob = np.zeros((no_vars, np.amax(lens)))
    pairwise_prob = np.zeros((no_vars, no_vars, np.amax(lens), np.amax(lens)))
    pairwise_MI = np.zeros((no_vars, no_vars))
    tot_rows = vals.shape[0]

    for k in range(len(uniq_vals)):
        for i in range(lens[k]):
            ind_prob[k, i] = sum(vals[:, k] == uniq_vals[k][i])/tot_rows

    for k in range(len(uniq_vals)):
        for k1 in range(k):
            MI = 0
            for i in range(lens[k1]):
                for j in range(lens[k]):
                    pairwise_prob[k1, k, i, j] = vals[(vals[:, k1] == uniq_vals[k1][i]) & (vals[:,k] == uniq_vals[k][j])].shape[0]/tot_rows
                    if pairwise_prob[k1, k, i, j] != 0:
                        MI += pairwise_prob[k1, k, i, j] * np.log2(pairwise_prob[k1, k, i, j]/(ind_prob[k1, i]*ind_prob[k, j]))
            pairwise_MI[k1, k] = MI
    """

    #print(pairwise_MI - pairwise_MI2)
    plot_tree(Prim_MST(-pairwise_MI_MLE), colnames)
    plot_tree(Prim_MST(-pairwise_MI_JVHW), colnames)

if __name__ == '__main__':
    colnames, vals = load_data('alarm10K.csv')
    get_MLE_JVHW_est(vals, colnames)