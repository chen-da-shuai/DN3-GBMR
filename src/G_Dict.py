'''
'''

import numpy as np
from annoy import AnnoyIndex
import networkx as nx
# from CanopyForNodeAttributes import Canopy as Cluster
import time
import multiprocessing


class Graph_dict():
    def __init__(self, capacity, key_dimension, act_dimension, dist_th, batch_size=8):
        # 一个图带19个表，每个表是一种动作的特征集合
        self.capacity = capacity  # 每个表的容量
        self.key_dimension = key_dimension
        self.act_dimension = act_dimension
        self.G = nx.DiGraph()
        # self.node_features = np.zeros((capacity,key_dimension))
        self.node_features_list = [np.zeros((capacity, key_dimension)) for i in range(self.act_dimension)]
        self.indices_list = [AnnoyIndex(key_dimension, metric='euclidean') for i in
                             range(self.act_dimension)]  # 用来给节点特征快速查近邻的
        # self.index.set_seed(123)
        self.initial_update_size = batch_size
        self.min_update_size = self.initial_update_size
        self.cached_keys = [[] for i in range(self.act_dimension)]
        self.cached_indices = [[] for i in range(self.act_dimension)]
        # 下面两个用来给节点的使用频率计数的
        self.lru = [np.zeros(self.capacity) for i in range(self.act_dimension)]
        self.tm = 0.0
        self.curr_capacity = [0 for i in range(self.act_dimension)]  # 是图的容量
        self.built_capacity = [0 for i in range(self.act_dimension)]
        # self.dist_th =4
        self.eta = 0.1  # 控制边权的增长速度
        self.dist_th = dist_th

    def Graphisfull(self):
        for a in range(self.act_dimension):
            if self.built_capacity[a] < self.capacity:
                return False
        return True

    def queryable(self, act, k):
        # 返回 动作act 对应的表中是否有k 个值供查询
        if self.curr_capacity[act] > k and self.built_capacity[act] > k:
            return True
        else:
            return False

    def _insert(self, a, keys, indices):
        self.cached_keys[a] = self.cached_keys[a] + keys
        # self.cached_values = self.cached_values + values
        self.cached_indices[a] = self.cached_indices[a] + indices

        if len(self.cached_indices[a]) >= self.min_update_size:  # 为啥设置这个阈值
            self.min_update_size = max(self.initial_update_size, self.curr_capacity[a] * 0.02)
            self._update_index(a)

    def _update_index(self, a):
        # self.index.unbuild()
        self.indices_list[a].unbuild()
        for i, ind in enumerate(self.cached_indices[a]):
            new_key = self.cached_keys[a][i]
            self.node_features_list[a][ind, :] = new_key
            self.indices_list[a].add_item(ind, new_key)

        self.cached_keys[a] = []
        self.cached_indices[a] = []

        self.indices_list[a].build(10)
        # self.index.build(50)
        self.built_capacity[a] = self.curr_capacity[a]

    def add_by_features(self, features, actions, values):
        len_features = len(features)
        temp_indices = [[] for i in range(self.act_dimension)]
        temp_keys = [[] for i in range(self.act_dimension)]
        encoded_trj = []

        for i in range(len_features):
            f = features[i]
            a = actions[i]
            if self.queryable(a, 1):
                ind, dist = self.indices_list[a].get_nns_by_vector(f, 1, include_distances=True)

                if len(ind) == 0 or len(dist) == 0:
                    d = np.inf
                else:
                    d = dist[0]
                if d < self.dist_th:
                    temp_index = ind[0]
                    key_ = (self.indices_list[a].get_item_vector(temp_index) + f * d) / (1 + d)
                elif self.curr_capacity[a] >= self.capacity:
                    temp_index = np.argmin(self.lru[a])
                    self.G.remove_node(temp_index + a * self.capacity)
                    key_ = f
                else:
                    temp_index = self.curr_capacity[a]
                    self.curr_capacity[a] += 1
                    key_ = f
            else:
                temp_index = self.curr_capacity[a]
                key_ = f
                self.curr_capacity[a] += 1

            self.lru[a][temp_index] = self.tm
            temp_indices[a].append(temp_index)
            temp_keys[a].append(key_)

            index_in_Graph = temp_index + a * self.capacity
            encoded_trj.append(index_in_Graph)

            if i == 0:
                start_node = index_in_Graph
                # 【性能优化】：避免重新 add_node，直接操作节点属性字典
                if start_node in self.G:
                    self.G.nodes[start_node]["visit"] += 1
                else:
                    self.G.add_node(start_node, visit=1)
                continue
            else:
                end_node = index_in_Graph
                if end_node in self.G:
                    self.G.nodes[end_node]["visit"] += 1
                else:
                    self.G.add_node(end_node, visit=1)

                if self.G.has_edge(start_node, end_node):
                    # 【性能优化】：直接内存级更新边权重，跳过 add_edge 验证机制
                    old_weight = self.G[start_node][end_node]["weight"]
                    new_weight = old_weight + self.eta * (values[i - 1] - old_weight)
                    self.G[start_node][end_node]["weight"] = new_weight
                    self.G[start_node][end_node]["label"] = actions[i - 1]
                else:
                    self.G.add_edge(start_node, end_node, label=actions[i - 1], weight=values[i - 1])

                start_node = end_node

        for a in range(self.act_dimension):
            if len(temp_keys[a]) > 0:
                self._insert(a, temp_keys[a], temp_indices[a])
        self.tm += 0.01
        return encoded_trj

    def add_by_indices(self, start_nodes, end_nodes, actions, weights):
        for i, start_node in enumerate(start_nodes):
            old_weight = weights[i]
            q_next = 0

            # 【性能优化】：使用底层的邻接字典迭代
            if end_nodes[i] in self.G:
                for v, data in self.G[end_nodes[i]].items():
                    if q_next < data.get("weight", 0.0):
                        q_next = data.get("weight", 0.0)

            new_weight = old_weight + self.eta * (q_next - old_weight)
            self.G[start_node][end_nodes[i]]['weight'] = new_weight

            a = actions[i]
            temp_index = start_node - a * self.capacity
            self.lru[a][temp_index] = self.tm
        self.tm += 0.01


    def query_by_features(self, features, actions, k):
        Q_s = np.zeros((len(features), k))
        Embs = np.zeros((len(features), k, self.key_dimension))
        Dists = np.full((len(features), k), np.inf)

        for i, (f, a) in enumerate(zip(features, actions)):
            inds, dists = self.indices_list[a].get_nns_by_vector(f, k, include_distances=True)
            num_found = len(inds)

            if num_found > 0:
                Dists[i, :num_found] = dists
                for j, ind in enumerate(inds):
                    index = ind + a * self.capacity

                    if index in self.G:
                        q_candidates = [
                            data.get("weight", 0.0)
                            for _, data in self.G[index].items()
                            if data.get("label") == a
                        ]
                        if len(q_candidates) >= 2:
                            Q_s[i, j] = max(q_candidates)
                        elif len(q_candidates) == 1:
                            Q_s[i, j] = q_candidates[0] * 0.6
                        else:
                            Q_s[i, j] = 0.0
                Embs[i, :num_found, :] = self.node_features_list[a][inds, :]
                self.lru[a][inds] = self.tm

        self.tm += 0.01
        return Q_s, Embs, Dists
    def GetKeyPointByDegree(self, num_center):
        N = np.sum(self.built_capacity)
        self.SI = np.zeros(self.capacity * self.act_dimension)

        # 【性能优化】：提取字典级的入度/出度，避免在循环中不断创建 EdgeView
        in_degrees = dict(self.G.in_degree())
        out_degrees = dict(self.G.out_degree())

        for node in self.G.nodes():
            fb = in_degrees.get(node, 0)  # before
            fa = out_degrees.get(node, 0)  # after
            ns = self.G.nodes[node].get("visit", 1)  # 默认最少被访问过1次

            si = (fa + fb) / (np.sqrt(ns / N + 1e-6))
            self.SI[node] = si

        Sorted_si_ind = np.argsort(self.SI, axis=0)[-num_center:-1]
        center_list = list(Sorted_si_ind)
        return center_list

    def GraphCluster(self, num_center):
        # 这个操作要在内存刚存满的时候，在接下来要决定谁留下来

        cls_time_a = time.time()
        center_list = self.GetKeyPointByDegree(num_center)
        print("center list", center_list)
        cls_time_b = time.time()
        print("cls time ", cls_time_b - cls_time_a)

        # print("G.NODES",self.G.nodes())
        reconstruct_time_a = time.time()

        # pool = multiprocessing.Pool(multiprocessing.cpu_count() - 2)

        for i in range(len(center_list)):
            for j in range(len(center_list)):
                if i == j:
                    pass
                else:
                    # print(len(self.G.nodes()),i,j)
                    self.search(center_list, i, j)
                    # pool.apply_async(self.search, (i, j))
        # pool.close()
        # pool.join()

        reconstruct_time_b = time.time()
        # print("rec _time", reconstruct_time_b - reconstruct_time_a)


    # def GlobalValueIteration(self, gamma_reward=0.95, gamma_penalty=0.9, iterations=5):
    #     """【反思机制】全局Bellman更新，加入路径长度惩罚防幻觉"""
    #     for _ in range(iterations):
    #         for u in self.G.nodes():
    #             for v, data in self.G[u].items():
    #                 old_weight = data.get('weight', 0.0)
    #
    #                 if old_weight <= -50.0:
    #                     continue  # 物理撞墙的死亡印记，绝不覆盖
    #
    #                 # 寻找下一状态 (v) 的最大 Q 值
    #                 max_q_next = 0.0
    #                 if v in self.G and len(self.G[v]) > 0:
    #                     max_q_next = max([d.get('weight', 0.0) for next_v, d in self.G[v].items()])
    #
    #                 # 【核心反思优化】：传导正反馈时，强制扣除1.0的分数作为“走路的体力消耗”
    #                 # 这样图网络会自动裁剪掉那些绕圈子、反复横跳的冗余捷径
    #                 if max_q_next > 0:
    #                     target_weight = (gamma_reward * max_q_next) - 1.0
    #                 else:
    #                     # 痛觉衰减
    #                     target_weight = gamma_penalty * max_q_next
    #
    #                 # 只有价值变得更加明确（更高分或更低分）时才更新
    #                 if target_weight < -1.0 and target_weight < old_weight:
    #                     self.G[u][v]['weight'] = target_weight
    #                 elif target_weight > 1.0 and target_weight > old_weight:
    #                     self.G[u][v]['weight'] = target_weight
    def GlobalValueIteration(self,
                             path_nodes=None,
                             path_actions=None,
                             path_rewards=None,
                             terminal_type='success',
                             gamma_reward=0.95,
                             gamma_penalty=0.9,
                             success_reward=100.0,
                             collision_reward=-50.0,
                             step_cost=1.0,
                             iterations=1):
        """
        只对“当前这一条路径”做回溯，不再扫描整张图。

        参数:
            path_nodes:    节点序列
            path_actions:  动作序列
            path_rewards:  每一步即时奖励
            terminal_type: 'success' 或 'collision'
        返回:
            path_values:   对应每条边的回溯值
        """
        if path_nodes is None or path_actions is None or path_rewards is None:
            return []

        if len(path_nodes) < 2 or len(path_actions) == 0 or len(path_rewards) == 0:
            return []

        # 关键：边数只能取 节点数-1、动作数、奖励数 三者最小值
        n_edges = min(len(path_nodes) - 1, len(path_actions), len(path_rewards))
        if n_edges <= 0:
            return []

        path_values = [0.0] * n_edges

        # 终点源头
        if terminal_type == 'success':
            R = success_reward
            gamma = gamma_reward
        else:
            R = collision_reward
            gamma = gamma_penalty

        # 只沿当前路径反向回溯
        for i in range(n_edges - 1, -1, -1):
            if terminal_type == 'success':
                R = path_rewards[i] + gamma * R
            else:
                R = path_rewards[i] + gamma * R
            path_values[i] = R

        # 写回图中的对应边
        for i in range(n_edges):
            u = path_nodes[i]
            v = path_nodes[i + 1]
            a = path_actions[i]
            target_weight = path_values[i]

            if self.G.has_edge(u, v):
                old_weight = self.G[u][v].get('weight', 0.0)
                new_weight = old_weight + self.eta * (target_weight - old_weight)
                self.G[u][v]['weight'] = new_weight
                self.G[u][v]['label'] = a
            else:
                self.G.add_edge(u, v, label=a, weight=target_weight)

        # 成功时给最后节点补一个自环高值，方便后续导航找到“目标吸引子”
        if terminal_type == 'success':
            last_node = path_nodes[n_edges]
            last_action = path_actions[n_edges - 1]
            self.G.add_edge(last_node, last_node, label=last_action, weight=path_values[-1])

        return path_values
    def AddShortcuts(self, shortcuts):
        """
        【推理机制】在物理空间相近但拓扑图中不相连的节点之间，搭建捷径
        shortcuts 格式: list of (node_i, node_j, action_idx, weight)
        """
        count = 0
        for src, dst, action_idx, weight in shortcuts:
            if src in self.G and dst in self.G:
                if not self.G.has_edge(src, dst):
                    self.G.add_edge(src, dst, label=action_idx, weight=weight)
                    count += 1
                else:
                    # 如果已有边，保留价值更高的
                    old_w = self.G[src][dst].get('weight', -999.0)
                    if weight > old_w:
                        self.G[src][dst]['weight'] = weight
                        self.G[src][dst]['label'] = action_idx
        return count
    def search(self, center_list, i, j):
        if nx.algorithms.shortest_paths.generic.has_path(self.G, center_list[i], center_list[j]):
            # print("reconstruct")
            path = nx.shortest_path(self.G, center_list[i], center_list[j])
            # print("path",path)
            temp_nodes = []
            temp_actions = []
            temp_weights = []
            temp_nodes_ = []
            for idx in range(len(path) - 1):
                pair_start = path[idx]
                pair_end = path[idx + 1]
                temp_nodes.append(pair_start)
                temp_nodes_.append(pair_end)
                temp_actions.append(self.G.edges[pair_start, pair_end]['label'])
                temp_weights.append(self.G.edges[pair_start, pair_end]['weight'])

            self.add_by_indices(temp_nodes, temp_nodes_, temp_actions, temp_weights)
        else:
            # print("no path")
            pass

    def ReconstructGraph(self, center_list):
        # print("keypoints", center_list)
        reconstruct_time_a = time.time()

        # 【优化核心】：利用单源最短路一次性求出当前源点到所有点的路径，消除 N^2 次 BFS 搜索
        for src in center_list:
            try:
                # 一次搜索得出 src 到所有可达节点的路径
                paths = nx.single_source_shortest_path(self.G, src)
                for dst in center_list:
                    if src != dst and dst in paths:
                        path = paths[dst]
                        temp_nodes = path[:-1]
                        temp_nodes_ = path[1:]

                        # 批量获取动作和权重
                        temp_actions = [self.G.edges[u, v]['label'] for u, v in zip(temp_nodes, temp_nodes_)]
                        temp_weights = [self.G.edges[u, v]['weight'] for u, v in zip(temp_nodes, temp_nodes_)]

                        self.add_by_indices(temp_nodes, temp_nodes_, temp_actions, temp_weights)
            except Exception as e:
                pass

        reconstruct_time_b = time.time()
        # print("rec _time", reconstruct_time_b - reconstruct_time_a)

    def flush_index(self):
        """强制将所有缓存的数据构建到 Annoy 树中"""
        for a in range(self.act_dimension):
            if len(self.cached_keys[a]) > 0:
                self._update_index(a)

# 用来测试这个类的
if __name__ == '__main__':
    # 用来对类内新增函数作简单测试
    G = Graph_dict(70, 8, 9, 0.4)  # 7个数据，每个数据的特征是8维，有9个动作

    embs = []
    acts = []
    rews = []
    for i in range(40):
        e = np.random.randn(8)
        a = np.random.randint(9)
        r = np.random.rand()
        embs.append(e)
        acts.append(a)
        rews.append(r)
    # print(embs,acts,rews)

    G.add_by_features(embs, acts, rews)
    # for i,a in enumerate(acts):
    #     adjm = G.adj_matrix
    #     weightm = G.weight_matrix
    #     print("action ",a)
    #     print(adjm[:,:,a])
    #     #print(weightm[:,:,a])
    embs = []
    acts = []
    rews = []
    for i in range(50):
        e = np.random.randn(8)
        a = np.random.randint(9)
        r = np.random.rand()
        embs.append(e)
        acts.append(a)
        rews.append(r)
    # print(embs,acts,rews)

    G.add_by_features(embs, acts, rews)
    # for i,a in enumerate(acts):
    #     adjm = G.adj_matrix
    #     weightm = G.weight_matrix
    #     print("action ",a)
    #     print(adjm[:,:,a])
    #     #print(weightm[:,:,a])
    embs = []
    acts = []
    rews = []
    for i in range(30):
        e = np.random.randn(8)
        a = np.random.randint(9)
        r = np.random.rand()
        embs.append(e)
        acts.append(a)
        rews.append(r)
    # print(embs,acts,rews)

    G.add_by_features(embs, acts, rews)
    # for i,a in enumerate(acts):
    #     adjm = G.adj_matrix
    #     weightm = G.weight_matrix
    #     print("action ",a)
    #     print(adjm[:,:,a])
    #     #print(weightm[:,:,a])

    G.GraphCluster(5)

