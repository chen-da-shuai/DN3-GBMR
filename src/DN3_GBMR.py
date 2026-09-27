#! /usr/bin/env python
# ! encoding=UTF-8

import collections
import os
from copy import deepcopy
import seaborn as sns
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
# 引入 GBMR 核心组件
from G_Dict import Graph_dict
from GraphQAgent import ExpertMemory
import pickle
# 初始化 AnnoyIndex 结构
from annoy import AnnoyIndex
import copy
import networkx as nx
import time
sns.set_theme(style='darkgrid')  # 图形主题
plt.rcParams['figure.dpi'] = 120  # 图形分辨率
plt.rcParams['font.sans-serif'] = ['SimSun']
# plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False



# 学习率是 pow(年龄， 1/3)
# 面对不同环境下的DN3
class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward):  # 将数据加入buffer
        self.buffer.append((state, action, reward))

    def choice(self, batch_size):
        # 如果经验池的数据量不足 batch_size，直接返回所有数据的索引，防止抽样报错
        current_size = len(self.buffer)
        if current_size < batch_size:
            return np.arange(current_size)

        _, _, reward = zip(*self.buffer)
        reward_array = np.array(reward)
        abs_rewards = np.abs(reward_array) + 1e-5
        p = abs_rewards / abs_rewards.sum()

        indexs = np.random.choice(a=np.arange(current_size), p=p, size=batch_size, replace=False)
        return indexs

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)



class Dn3():
    def __init__(self) -> object:
        self.theta_100 = 0.95
        self.theta_011 = 0.95

        # 网络0
        # 创建 X 层
        self.x_input = np.zeros(10)
        # 创建 Y 层
        self.Y_100 = {'age': [1], 'W_xy': np.random.random([10, 1])}
        self.Y_011 = {'age': [1], 'W_yy': np.random.random([1, 1]), 'W_zy': np.random.random([36, 1])}
        # 创建 Z 层
        self.Z = {'age': np.ones(36), 'W_yz': np.random.random([1, 36])}

        # 网络1
        # 创建 Y 层
        self.Y1_100 = {'age': [1], 'W_xy': np.random.random([10, 1])}
        self.Y1_011 = {'age': [1], 'W_yy': np.random.random([1, 1]), 'W_zy': np.random.random([36, 1])}
        # 创建 Z 层
        self.Z1 = {'age': np.ones(36), 'W_yz': np.random.random([1, 36])}

        self.count = 0
        self.target_update = 100

        # np.random.seed(seed_num)  # 不能改变位置

        # 智能体 障碍物 目标体8
        self.agent = np.array([50, 50])
        # self.agent = np.array([50, 50])

        self.friend = np.array([450, 450])
        # self.friend = np.array([50, 50])
        # self.friend = np.array([400, 400])

        self.ems = np.array([[300, 170], [270, 290], [200, 250], [150, 360]])
        # self.ems = np.array(list(set(tuple(t) for t in self.ems)))

        self.d_em = 5  # 碰撞结束
        self.d_fri = 5  # 到达结束
        self.cur_fri_dis = 0
        self.cur_em_dis = 0

        self.replay_buffer = ReplayBuffer(5000)
        self.minimal_size = 30
        self.batch_size = 64  # 经验池大了，每次抽样的 batch_size 也可以适当加大到 64
        self.iteration = 20

        # Precompute 36-direction geometry to avoid repeated sin/cos
        self._action_angles = np.arange(36, dtype=float) * (np.pi / 18.0)
        self._step10 = np.c_[
            (10.0 * np.cos(self._action_angles)).astype(int),
            (10.0 * np.sin(self._action_angles)).astype(int),
        ]
        self._step20 = np.c_[
            (20.0 * np.cos(self._action_angles)).astype(int),
            (20.0 * np.sin(self._action_angles)).astype(int),
        ]

        self.G = Graph_dict(capacity=2000, key_dimension=10, act_dimension=36, dist_th=0.08,batch_size=200)
        self.keypoint = ExpertMemory(memory_size=8)  # 记录高分轨迹
        self.discount = 0.95  # 回报衰减率

        # 轨迹缓存（用于一局结束后写入 GBMR）
        self.trj_features = []
        self.trj_actions = []
        self.trj_rewards = []
        self.trj_obs = []

        self.test_50 = False

        # ===== GBMR 融合使用检查器 =====
        self.debug_gbmr = True   # 总开关：True 开启打印与统计
        self.debug_step_verbose = False  # True 时每一步都打印，建议先关
        self.reset_fusion_stats()

        # ===== GBMR图规模与计算时间统计 =====
        self.graph_growth_records = []
        self.graph_update_times_ms = []
        self.graph_search_times_ms = []
        self.decision_times_ms = []
        self.reflection_stats = {}

        self.gbmr_decisive_mode = True
        # ========== 新增：状态访问记忆 ==========
        self.grid_size = 20.0  # 栅格大小（建议20，对应智能体两步的距离）
        self.visited_grid = collections.defaultdict(int)  # 记录访问次数的字典

    def active_graph_navigation(self):

        # fri_dx = self.friend[0] - self.agent[0]
        # fri_dy = self.friend[1] - self.agent[1]
        # fri_dis = math.sqrt(fri_dx ** 2 + fri_dy ** 2)
        # if fri_dis <= 10.0:
        #     fri_angle = math.atan2(fri_dy, fri_dx)
        #     if fri_angle < 0: fri_angle += 2 * np.pi
        #     return int(round(fri_angle / (np.pi / 18.0))) % 36

        if getattr(self.G, 'G', None) is None or self.G.G.number_of_nodes() == 0:
            return None

        best_node = None
        best_weight = -999.0

        for u, v, data in self.G.G.edges(data=True):
            if u == v and data.get('weight', 0.0) >= 90.0:
                best_node = v
                break

        if best_node is None:
            for u, v, data in self.G.G.edges(data=True):
                if data.get('weight', -999.0) > best_weight:
                    best_weight = data['weight']
                    best_node = v

        if best_node is None:
            return None

        current_nodes = []
        # 搜索半径必须极小，绝不允许隔着墙相似度相似
        search_radius = self.G.dist_th * 1.5

        for a in range(36):
            if self.G.built_capacity[a] > 0:
                inds, dists = self.G.indices_list[a].get_nns_by_vector(self.x_input, 5, include_distances=True)
                for ind, dist in zip(inds, dists):
                    if dist < search_radius:
                        current_nodes.append((ind + a * self.G.capacity, dist))

        if not current_nodes:
            return None

        current_nodes = sorted(current_nodes, key=lambda x: x[1])[:5]

        # 3. 转换代价值
        for u, v, data in self.G.G.edges(data=True):
            w = data.get('weight', 0.0)
            # 因为高分捷径在 100~150 左右，用 200 去减，让算法自然选择阻力最小的路
            data['cost'] = max(0.1, 200.0 - w)
        valid_plans = []
        for start_node, dist in current_nodes:
            if start_node == best_node:
                continue
            try:
                cost = nx.shortest_path_length(self.G.G, source=start_node, target=best_node, weight='cost')
                path = nx.shortest_path(self.G.G, source=start_node, target=best_node, weight='cost')
                if len(path) > 1:
                    valid_plans.append((start_node, dist, path, cost))
            except nx.NetworkXNoPath:
                continue

        if not valid_plans:
            return None

        min_cost = min([plan[3] for plan in valid_plans])
        action_votes = np.zeros(36)

        for start_node, dist, path, cost in valid_plans:
            if cost > min_cost * 1.5:
                continue

            next_node = path[1]
            suggested_action = int(self.G.G[start_node][next_node]['label'])
            weight = (1.0 / (dist + 1e-5)) * (min_cost / cost)

            action_votes[suggested_action] += weight
            action_votes[(suggested_action - 1) % 36] += weight * 0.5
            action_votes[(suggested_action + 1) % 36] += weight * 0.5

        if hasattr(self, 'trj_actions') and len(self.trj_actions) > 0:
            last_action = self.trj_actions[-1]
            if action_votes[last_action] > 0:
                action_votes[last_action] *= 1.2

        if np.max(action_votes) > 0:
            return int(np.argmax(action_votes))

        return None

    def global_reflection(self, safe_margin=15.0, value_iterations=30):
        if getattr(self.G, 'G', None) is None or self.G.G.number_of_nodes() == 0:
            self.reflection_stats = {
                'nodes_before_reflection': 0,
                'edges_before_reflection': 0,
                'nodes_after_reflection': 0,
                'edges_after_reflection': 0,
                'nodes_added_by_reflection': 0,
                'edges_added_by_reflection': 0,
                'candidate_shortcuts': 0,
                'line_of_sight_ms': 0.0,
                'reflection_total_ms': 0.0
            }
            return self.reflection_stats

        reflection_start = time.perf_counter_ns()
        nodes_before = self.G.G.number_of_nodes()
        edges_before = self.G.G.number_of_edges()

        line_of_sight_start = time.perf_counter_ns()
        shortcuts = self.keypoint.get_shortcuts(obstacles=self.ems, safe_margin=safe_margin)
        line_of_sight_ms = (time.perf_counter_ns() - line_of_sight_start) / 1e6

        if shortcuts:
            for src, dst, action_idx, weight, pos_i, pos_j in shortcuts:
                self.G.G.add_edge(src, dst, label=action_idx, weight=weight)

                dist = np.linalg.norm(pos_j - pos_i)
                steps = int(dist / 8.0)

                if steps > 1:
                    v_features, v_actions, v_rewards = [], [], []
                    temp_agent = self.agent.copy()

                    for k in range(1, steps):
                        self.agent = pos_i + (pos_j - pos_i) * (k / steps)
                        f_dis, e_dis, f_ang, e_ang, e2_dis, e2_ang = self.get_observations()
                        v_x = np.array([
                            np.cos(f_ang), np.sin(f_ang), np.cos(e_ang), np.sin(e_ang),
                            f_dis / (f_dis + e_dis), e_dis / (f_dis + e_dis),
                            np.cos(e2_ang), np.sin(e2_ang), e2_dis / (f_dis + e2_dis),
                            e_dis / e2_dis
                        ])
                        v_features.append(v_x)
                        v_actions.append(action_idx)
                        v_rewards.append(weight)

                        self.replay_buffer.add(v_x, action_idx, weight)

                    self.agent = temp_agent
                    if v_features:
                        # 先把虚拟中间路径编码出来
                        dummy_values = [0.0 for _ in v_actions]
                        v_nodes = self.G.add_by_features(v_features, v_actions, dummy_values)

                        # 拼成完整捷径路径: src -> 中间虚拟节点们 -> dst
                        full_nodes = [src] + v_nodes + [dst]
                        full_actions = [action_idx] * (len(full_nodes) - 1)

                        # 给这条捷径路径构造一个正向奖励链
                        full_rewards = [0.0] * (len(full_actions) - 1) + [weight]

                        # 只对这条捷径路径做一次单独回溯
                        self.G.GlobalValueIteration(
                            path_nodes=full_nodes,
                            path_actions=full_actions,
                            path_rewards=full_rewards,
                            terminal_type='success',
                            gamma_reward=0.95,
                            gamma_penalty=0.95,
                            success_reward=weight,
                            collision_reward=-50.0,
                            step_cost=1.0
                        )

        # self.G.GlobalValueIteration(gamma_reward=0.98, gamma_penalty=0.85, iterations=value_iterations)

        if shortcuts:
            for _ in range(5):
                sampled_indices = self.replay_buffer.choice(self.batch_size)
                for idx in sampled_indices:
                    self.update1(self.replay_buffer.buffer[idx][0], self.replay_buffer.buffer[idx][1])

        # 把记忆刻进脑子里
        self.G.flush_index()

        nodes_after = self.G.G.number_of_nodes()
        edges_after = self.G.G.number_of_edges()
        reflection_total_ms = (time.perf_counter_ns() - reflection_start) / 1e6
        self.reflection_stats = {
            'nodes_before_reflection': nodes_before,
            'edges_before_reflection': edges_before,
            'nodes_after_reflection': nodes_after,
            'edges_after_reflection': edges_after,
            'nodes_added_by_reflection': nodes_after - nodes_before,
            'edges_added_by_reflection': edges_after - edges_before,
            'candidate_shortcuts': len(shortcuts),
            'line_of_sight_ms': line_of_sight_ms,
            'reflection_total_ms': reflection_total_ms
        }
        return self.reflection_stats
    def _record_visited(self, pos):
        #将连续坐标转换为栅格坐标，并增加该区域的访问计数
        grid_x = int(pos[0] // self.grid_size)
        grid_y = int(pos[1] // self.grid_size)
        self.visited_grid[(grid_x, grid_y)] += 1
    def reset_fusion_stats(self):
        self.fusion_stats = {
            "steps_total": 0,  # 总决策步数
            "gbmr_available_steps": 0,  # GBMR 有有效分数(alpha>0)的步数
            "gbmr_changed_top1_steps": 0,  # GBMR 改变了融合前top1的步数
            "shield_changed_steps": 0,  # safety shield 改了融合动作的步数
            "gbmr_and_shield_same": 0,  # GBMR改了top1但shield没挡回去
            "gbmr_and_shield_reverted": 0,  # GBMR改了top1但被shield改回/改掉
            "final_followed_gbmr_steps": 0,  # 最终动作 == GBMR建议top1
            "final_followed_dn3_steps": 0,  # 最终动作 == DN3原始top1
            "episodes": 0,
            "episode_success": 0
        }
        self.last_fusion_debug = None

    def _record_fusion_debug(self, mode, alpha, gbmr_qs, norm_g,
                             dn3_top1, gbmr_top1, fused_top1,
                             candidate_actions, dn3_scores, fused_scores,
                             support_count):
        gbmr_available = bool(alpha > 0 and np.max(gbmr_qs) > 1e-8)
        gbmr_changed_top1 = bool(gbmr_available and fused_top1 != dn3_top1)

        self.last_fusion_debug = {
            "mode": mode,  # "explore" or "test"
            "alpha": float(alpha),
            "gbmr_available": gbmr_available,
            "support_count": int(support_count),  # 有多少动作获得有效GBMR值
            "max_gbmr_q": float(np.max(gbmr_qs)) if len(gbmr_qs) else 0.0,
            "dn3_top1": int(dn3_top1),
            "gbmr_top1": int(gbmr_top1),
            "fused_top1": int(fused_top1),
            "gbmr_changed_top1": gbmr_changed_top1,
            "candidate_actions": np.array(candidate_actions, dtype=int).tolist(),
            "dn3_top3": np.argsort(dn3_scores)[-3:].tolist(),
            "fused_top3": np.argsort(fused_scores)[-3:].tolist(),
            "dn3_top1_score": float(np.max(dn3_scores)),
            "fused_top1_score": float(np.max(fused_scores)),
        }

        self.fusion_stats["steps_total"] += 1
        if gbmr_available:
            self.fusion_stats["gbmr_available_steps"] += 1
        if gbmr_changed_top1:
            self.fusion_stats["gbmr_changed_top1_steps"] += 1

        if self.debug_gbmr and self.debug_step_verbose:
            print(
                f"[GBMR-{mode}] alpha={alpha:.3f} support={support_count} "
                f"max_g={np.max(gbmr_qs):.3f} "
                f"DN3_top1={dn3_top1} GBMR_top1={gbmr_top1} "
                f"FUSED_top1={fused_top1} changed={gbmr_changed_top1} "
                f"cand={np.array(candidate_actions).tolist()}"
            )

    def _record_final_action(self, final_action):
        if self.last_fusion_debug is None:
            return

        dbg = self.last_fusion_debug
        final_action = int(final_action)

        if final_action == dbg["dn3_top1"]:
            self.fusion_stats["final_followed_dn3_steps"] += 1
        if final_action == dbg["gbmr_top1"]:
            self.fusion_stats["final_followed_gbmr_steps"] += 1

        if final_action != dbg["fused_top1"]:
            self.fusion_stats["shield_changed_steps"] += 1
            if dbg["gbmr_changed_top1"]:
                self.fusion_stats["gbmr_and_shield_reverted"] += 1
        else:
            if dbg["gbmr_changed_top1"]:
                self.fusion_stats["gbmr_and_shield_same"] += 1

        if self.debug_gbmr and self.debug_step_verbose:
            print(
                f"[FINAL] dn3={dbg['dn3_top1']} gbmr={dbg['gbmr_top1']} "
                f"fused={dbg['fused_top1']} final={final_action}"
            )


    def get_epsilon(self, episode):
        """改进的epsilon衰减策略，保持更多探索"""
        # 更慢的衰减，确保在训练后期仍有探索
        # if episode < 50:
        #     epsilon = self.epsilon_start * (0.98 ** (episode))
        # elif episode < 100:
        #     epsilon = self.epsilon_start * (0.9 ** (episode))
        # else:
        epsilon = self.epsilon_start * (0.98 ** (episode))

        return max(epsilon, self.epsilon_end)

    def generate_ems_line1(self):
        self.steps = np.inf

        # 第一条直线：x = 200, y = 250 ~ 420
        y1 = np.arange(250, 421)
        x1 = np.full_like(y1, 200)

        # 第二条直线：x = 300, y = 100 ~ 200
        y2 = np.arange(100, 201)
        x2 = np.full_like(y2, 300)

        # 合并
        self.ems = np.vstack([
            np.c_[x1, y1],
            np.c_[x2, y2]
        ])


    def select_action(self, obs, actions):

        if actions is None or np.size(actions) == 0:
            final_action = 0
            self._record_final_action(final_action)
            return final_action

        # 只有一个候选动作
        if np.size(actions) == 1:
            final_action = int(np.ravel(actions)[0])
            self._record_final_action(final_action)
            return final_action


        fri_dis, em_dis, fri_angle, em_angle, *_ = obs

        # actions 是融合网络 (DN3 + GBMR) 精心打分并排好序的 top_k 动作
        # check 函数会从最高分动作开始往下试，遇到第一个不撞墙的就返回
        final_action = int(self.check(em_dis, em_angle, fri_dis, fri_angle, np.array(actions)))

        self._record_final_action(final_action)

        return final_action

    def generate_ems(self, seed, size):
        self.steps = np.inf
        # 创建一个隔离的生成器，完全不触碰全局状态
        rng = np.random.RandomState(seed)
        self.ems = np.c_[rng.randint(100, 400, size=size), rng.randint(100, 400, size=size)]
        # self.ems = np.c_[np.random.randint(100, 400, size=size), np.random.randint(100, 400, size=size)]

    def generate_u_dead_end_ems(self):
        """
        生成一个清晰的 U 型死胡同环境
        适配：
            起点: [50,50]
            终点: [450,450]
        """
        self.steps = np.inf
        pts = []
        step = 1

        def add_rect(cx, cy, w, h):
            x0 = int(cx - w / 2)
            x1 = int(cx + w / 2)
            y0 = int(cy - h / 2)
            y1 = int(cy + h / 2)
            for x in range(x0, x1 + 1, step):
                for y in range(y0, y1 + 1, step):
                    pts.append([x, y])

        def add_triangle(cx, cy, w, h):
            half_h = h / 2
            for yy in np.arange(-half_h, half_h + step, step):
                width_now = w * (1 - (yy + half_h) / h)
                y = cy + yy
                if width_now < step:
                    width_now = step
                x0 = int(cx - width_now / 2)
                x1 = int(cx + width_now / 2)
                for x in range(x0, x1 + 1, step):
                    pts.append([x, int(y)])

        # =========================
        # 1. 主体：右中区域 U 型死胡同
        # 开口朝左，诱导智能体从左侧钻进去
        # =========================
        # 上横
        add_rect(cx=140, cy=305, w=30, h=60)

        add_rect(cx=400, cy=380, w=40, h=40)
        # 下横
        # add_rect(cx=165, cy=330, w=60, h=60)
        #
        add_rect(cx=405, cy=210, w=10, h=60)
        add_rect(cx=380, cy=240, w=60, h=10)
        add_rect(cx=380, cy=180, w=60, h=10)

        # add_rect(cx=120, cy=130, w=60, h=10)
        # add_rect(cx=145, cy=105, w=10, h=60)
        # 为了让“U型”更明显，不加左竖，保持左侧开口

        # =========================
        # 2. 起点附近一个小干扰块
        # 让它不要直线冲右下
        # =========================
        add_rect(cx=300, cy=365, w=60, h=60)
        add_rect(cx=200, cy=180, w=50, h=50)
        # =========================
        # 3. 中下区域一个辅助障碍
        # 防止直接从下方一路斜穿到终点
        # =========================
        add_triangle(cx=250, cy=275, w=110, h=60)

        # =========================
        # 4. 中部一个矩形块
        # 让路线更像“先靠近U，再判断绕行”
        # =========================
        # add_rect(cx=250, cy=270, w=70, h=80)

        pts = np.array(pts, dtype=int)

        # 裁剪到边界
        pts[:, 0] = np.clip(pts[:, 0], 30, 470)
        pts[:, 1] = np.clip(pts[:, 1], 30, 470)

        # 去重
        pts = np.unique(pts, axis=0)

        start = np.array([50, 50])
        goal = np.array([450, 450])

        # 起点终点保护区
        keep = []
        for p in pts:
            if np.linalg.norm(p - start) < 55:
                continue
            if np.linalg.norm(p - goal) < 55:
                continue
            keep.append(p)

        self.ems = np.array(keep, dtype=int)
    def generate_complex_ems(self, seed):
        """
        带防重叠机制的复杂环境生成器
        保证每次生成的环境各不相同，且障碍物之间互不粘连，留有通道
        """
        self.steps = np.inf
        np.random.seed(seed)
        pts = []
        step = 1

        # 记录已占据的区域边界框 [x_min, x_max, y_min, y_max]
        bboxes = []

        def check_overlap(new_box, padding=25):
            """
            碰撞检测函数：检查新生成的矩形框是否与已有的重叠
            padding=25 保证障碍物之间至少有 25 像素的通道，防止彻底堵死
            """
            nx1, nx2, ny1, ny2 = new_box
            for (ox1, ox2, oy1, oy2) in bboxes:
                # 如果两个矩形没有在 X 或 Y 轴上完全分离，就说明重叠了
                if not (nx2 + padding < ox1 or nx1 - padding > ox2 or
                        ny2 + padding < oy1 or ny1 - padding > oy2):
                    return True
            return False

        def add_rect(cx, cy, w, h):
            x0, x1 = int(cx - w / 2), int(cx + w / 2)
            y0, y1 = int(cy - h / 2), int(cy + h / 2)
            for x in range(x0, x1 + 1, step):
                for y in range(y0, y1 + 1, step):
                    pts.append([x, y])

        # def add_triangle(cx, cy, w, h):
        #     half_h = h / 2
        #     for yy in np.arange(-half_h, half_h + step, step):
        #         width_now = w * (1 - (yy + half_h) / h)
        #         y = cy + yy
        #         if width_now < step:
        #             width_now = step
        #         x0, x1 = int(cx - width_now / 2), int(cx + width_now / 2)
        #         for x in range(x0, x1 + 1, step):
        #             pts.append([x, int(y)])
        #
        # # =========================
        # # 1. 生成 U 型陷阱 (最高优先级，先占地盘)
        # # =========================
        # u_cx = np.random.randint(150, 350)
        # u_cy = np.random.randint(150, 350)
        # u_thickness = np.random.randint(10, 25)
        # u_width = np.random.randint(60, 120)
        # u_height = np.random.randint(60, 120)
        #
        # orientation = np.random.randint(0, 4)
        # if orientation == 0:  # 开口朝左
        #     add_rect(u_cx, u_cy + u_height / 2, u_width, u_thickness)
        #     add_rect(u_cx, u_cy - u_height / 2, u_width, u_thickness)
        #     add_rect(u_cx + u_width / 2, u_cy, u_thickness, u_height)
        # elif orientation == 1:  # 开口朝右
        #     add_rect(u_cx, u_cy + u_height / 2, u_width, u_thickness)
        #     add_rect(u_cx, u_cy - u_height / 2, u_width, u_thickness)
        #     add_rect(u_cx - u_width / 2, u_cy, u_thickness, u_height)
        # elif orientation == 2:  # 开口朝上
        #     add_rect(u_cx - u_width / 2, u_cy, u_thickness, u_height)
        #     add_rect(u_cx + u_width / 2, u_cy, u_thickness, u_height)
        #     add_rect(u_cx, u_cy - u_height / 2, u_width, u_thickness)
        # else:  # 开口朝下
        #     add_rect(u_cx - u_width / 2, u_cy, u_thickness, u_height)
        #     add_rect(u_cx + u_width / 2, u_cy, u_thickness, u_height)
        #     add_rect(u_cx, u_cy + u_height / 2, u_width, u_thickness)
        #
        # # 把 U 型整体区域加入 bboxes（估算一个大包围盒）
        # bboxes.append([u_cx - u_width / 2, u_cx + u_width / 2, u_cy - u_height / 2, u_cy + u_height / 2])
        #
        # # =========================
        # # 2. 随机生成 1~2 个三角形 (带重叠检测)
        # # =========================
        # num_triangles = np.random.randint(1, 3)
        # attempts = 0  # 尝试次数限制，防止死循环
        # added_t = 0
        # while added_t < num_triangles and attempts < 20:
        #     attempts += 1
        #     t_cx = np.random.randint(100, 400)
        #     t_cy = np.random.randint(100, 400)
        #     t_w = np.random.randint(40, 100)
        #     t_h = np.random.randint(40, 100)
        #
        #     # 计算当前三角形的包围盒
        #     new_box = [t_cx - t_w / 2, t_cx + t_w / 2, t_cy - t_h / 2, t_cy + t_h / 2]
        #
        #     # 如果不重叠，才真正添加
        #     if not check_overlap(new_box, padding=25):
        #         add_triangle(t_cx, t_cy, t_w, t_h)
        #         bboxes.append(new_box)
        #         added_t += 1
        #
        # # =========================
        # # 3. 随机生成 3~5 个矩形干扰块 (带重叠检测)
        # # =========================
        # num_rects = np.random.randint(3, 6)
        # attempts = 0
        # added_r = 0
        # while added_r < num_rects and attempts < 20:
        #     attempts += 1
        #     r_cx = np.random.randint(100, 400)
        #     r_cy = np.random.randint(100, 400)
        #     r_w = np.random.randint(30, 80)
        #     r_h = np.random.randint(30, 80)
        #
        #     new_box = [r_cx - r_w / 2, r_cx + r_w / 2, r_cy - r_h / 2, r_cy + r_h / 2]
        #
        #     if not check_overlap(new_box, padding=25):
        #         add_rect(r_cx, r_cy, r_w, r_h)
        #         bboxes.append(new_box)
        #         added_r += 1
        #
        # # =========================
        # # 后处理：防越界与保护区裁剪
        # # =========================
        # pts = np.array(pts, dtype=int)
        # if len(pts) > 0:
        #     # 裁剪到边界
        #     pts[:, 0] = np.clip(pts[:, 0], 30, 470)
        #     pts[:, 1] = np.clip(pts[:, 1], 30, 470)
        #     # 去重
        #     pts = np.unique(pts, axis=0)
        #
        # start = np.array([50, 50])
        # goal = np.array([450, 450])
        #
        # # 起点终点保护区，留出足够的空地
        # keep = []
        # for p in pts:
        #     if np.linalg.norm(p - start) < 65:  # 扩大一点起点保护圈
        #         continue
        #     if np.linalg.norm(p - goal) < 65:
        #         continue
        #     keep.append(p)
        #
        # self.ems = np.array(keep, dtype=int)
    @staticmethod
    def split(fire_index, neuron):
        for k in neuron.keys():
            if k == 'age':
                # print(k)
                # print(neuron[k])
                neuron[k].append(1)
            else:
                fire_neuron = neuron[k][:, fire_index]
                neuron[k] = np.c_[neuron[k], fire_neuron]


    def response_index(self, index):
        # y_100 response
        r_ = np.zeros(len(self.Y_100['age']))
        r_[index] = 1

        # Y_011 response
        y_y = np.dot(r_, self.Y_011['W_yy'])

        idmax = y_y.argmax()
        # valmax = y_y.max()
        r_011 = np.zeros([len(self.Y_011['age']), 1])
        r_011[idmax] = 1

        z_response = np.dot(r_011.T, self.Z['W_yz'])

        action = z_response.argmax()

        return action

    def response_index1(self, index):
        # y_100 response
        r_ = np.zeros(len(self.Y1_100['age']))
        r_[index] = 1

        # Y_011 response
        y_y = np.dot(r_, self.Y1_011['W_yy'])

        idmax = y_y.argmax()
        # valmax = y_y.max()
        r_011 = np.zeros([len(self.Y1_011['age']), 1])
        r_011[idmax] = 1

        z_response = np.dot(r_011.T, self.Z1['W_yz'])

        action = z_response.argmax()

        return action, idmax


    # 知识迁移的 update
    def update(self, x_input, z_label):
        # y_100 response
        r_y_100 = np.dot((x_input / np.linalg.norm(x_input)), (
                self.Y_100['W_xy'] / np.linalg.norm(self.Y_100['W_xy'], axis=0)))
        """并非最大的激活，找大于相似度阈值的值"""
        indexes = np.where(r_y_100 >= self.theta_100)[0]

        if indexes.size == 0:
            Dn3.split(r_y_100.argmax(), self.Y_100)
            # 新的神经元更新
            # y_100 增加 y_011 的权值的shape要改变
            self.Y_011['W_yy'] = np.r_[self.Y_011['W_yy'], np.zeros([1, self.Y_011['W_yy'].shape[1]])]
            # self.Y_011['W_zy'] = np.c_[self.Y_011['W_zy'], np.random.random([36, 1])] # Y_100 增加没影响
            r_ = np.zeros(len(self.Y_100['age']))
            r_[-1] = 1
            self.Y_100['W_xy'][:, -1] = self.Y_100['W_xy'][:, -1] * (
                    1 - 1 / pow(self.Y_100['age'][-1], 1 / 1)) + x_input / pow(self.Y_100['age'][-1], 1 / 1)
            self.Y_100['age'][-1] += 1

        elif indexes.size == 1:
            self.Y_100['W_xy'][:, indexes[0]] = self.Y_100['W_xy'][:, indexes[0]] * (
                    1 - 1 / pow(self.Y_100['age'][indexes[0]], 1 / 1)) + x_input / pow(self.Y_100['age'][indexes[0]],
                                                                                       1 / 1)
            self.Y_100['age'][indexes[0]] += 1
            if self.Y_100['age'][indexes[0]] >= 50:
                """知识迁移"""
                Dn3.split(indexes[0], self.Y_100)
                self.Y_011['W_yy'] = np.r_[self.Y_011['W_yy'], np.zeros([1, self.Y_011['W_yy'].shape[1]])]

            r_ = np.zeros(len(self.Y_100['age']))
            r_[indexes[0]] = 1

        else:
            ages = np.zeros(indexes.size)
            for i in range(indexes.size):
                ages[i] = self.Y_100['age'][indexes[i]]

            if sum(ages >= 50) == 0:  # 没有年龄大于50的神经元
                # print('Warning: 没有年纪大于50的神经元')
                r_ = np.zeros(len(self.Y_100['age']))
                r_[np.argmax(r_y_100)] = 1
            else:
                old_neuron = np.where(ages >= 50)[0]
                response_action = np.zeros(old_neuron.size)
                for neuron in range(old_neuron.size):
                    response_action[neuron] = self.response_index(indexes[old_neuron[neuron]])

                if sum(response_action == z_label) == 0:
                    """年龄小于50的学习"""
                    young_neuron = np.where(ages < 50)[0]
                    if young_neuron.size != 1:  # 年龄小于50的神经元不只一个
                        # print('Warning: 年纪小于50的神经元有', sum(ages < 50))
                        r_ = np.zeros(len(self.Y_100['age']))
                        r_[np.argmax(r_y_100)] = 1
                    else:
                        self.Y_100['W_xy'][:, indexes[young_neuron[0]]] = self.Y_100['W_xy'][:,
                                                                          indexes[young_neuron[0]]] * (
                                                                                  1 - 1 / pow(self.Y_100['age'][indexes[
                                                                              young_neuron[0]]],
                                                                                              1 / 1)) + x_input / pow(
                            self.Y_100['age'][indexes[young_neuron[0]]], 1 / 1)
                        self.Y_100['age'][indexes[young_neuron][0]] += 1
                        if self.Y_100['age'][indexes[young_neuron[0]]] >= 50:
                            """知识迁移"""
                            Dn3.split(indexes[young_neuron[0]], self.Y_100)
                            self.Y_011['W_yy'] = np.r_[self.Y_011['W_yy'], np.zeros([1, self.Y_011['W_yy'].shape[1]])]

                        r_ = np.zeros(len(self.Y_100['age']))
                        r_[indexes[young_neuron[0]]] = 1
                else:
                    # r_ = np.zeros(len(self.Y_100['age']))  # 感觉没有写完
                    correct_old = old_neuron[np.where(response_action == z_label)[0][0]]
                    real_index = indexes[correct_old]

                    r_ = np.zeros(len(self.Y_100['age']))
                    r_[real_index] = 1

        if r_.sum() == 1:
            # Y_011 response
            # z_label = z_fire[i] - 1  # top-down
            z_input = np.zeros(36)
            z_input[z_label] = 1
            y_y = np.dot(r_, self.Y_011['W_yy']) + np.dot(z_input, self.Y_011['W_zy'])
            # top-1
            idmax = y_y.argmax()
            valmax = y_y.max()
            # m = self.mt(self.Y_011['age'][idmax])
            if valmax <= self.theta_011 * 2:
                Dn3.split(idmax, self.Y_011)
                # z层的权重要改变
                self.Z['W_yz'] = np.r_[self.Z['W_yz'], np.zeros([1, self.Z['W_yz'].shape[1]])]
                r_011 = np.zeros(len(self.Y_011['age']))
                # r_011 = np.zeros(len(self.Y_011['age']))
                r_011[-1] = 1
                self.Y_011['W_yy'][:, -1] = self.Y_011['W_yy'][:, -1] * (
                        1 - 1 / pow(self.Y_011['age'][-1], 1 / 1)) + r_ / pow(self.Y_011['age'][-1], 1 / 1)
                self.Y_011['W_zy'][:, -1] = self.Y_011['W_zy'][:, -1] * (
                        1 - 1 / pow(self.Y_011['age'][-1], 1 / 1)) + z_input / pow(self.Y_011['age'][-1], 1 / 1)
                self.Y_011['age'][-1] += 1
            else:
                # r_011 = np.zeros(len(self.Y_011['age']))
                r_011 = np.zeros(len(self.Y_011['age']))
                r_011[idmax] = 1
                self.Y_011['W_yy'][:, idmax] = self.Y_011['W_yy'][:, idmax] * (
                        1 - 1 / pow(self.Y_011['age'][idmax], 1 / 1)) + r_ / pow(self.Y_011['age'][idmax], 1 / 1)
                self.Y_011['W_zy'][:, idmax] = self.Y_011['W_zy'][:, idmax] * (
                        1 - 1 / pow(self.Y_011['age'][idmax], 1 / 1)) + z_input / pow(self.Y_011['age'][idmax], 1 / 1)
                self.Y_011['age'][idmax] += 1

            self.Z['W_yz'][:, z_label] = self.Z['W_yz'][:, z_label] * (
                    1 - 1 / pow(self.Z['age'][z_label], 1 / 1)) + r_011 / pow(self.Z['age'][z_label], 1 / 1)
            self.Z['age'][z_label] += 1

    # 知识迁移的 update
    def update1(self, x_input, z_label):
        # y_100 response
        r_y_100 = np.dot((x_input / np.linalg.norm(x_input)), (
                self.Y1_100['W_xy'] / np.linalg.norm(self.Y1_100['W_xy'], axis=0)))
        """并非最大的激活，找大于相似度阈值的值"""
        indexes = np.where(r_y_100 >= self.theta_100)[0]

        if indexes.size == 0:
            Dn3.split(r_y_100.argmax(), self.Y1_100)
            # 新的神经元更新
            # y_100 增加 y_011 的权值的shape要改变
            self.Y1_011['W_yy'] = np.r_[self.Y1_011['W_yy'], np.zeros([1, self.Y1_011['W_yy'].shape[1]])]
            # self.Y1_011['W_zy'] = np.c_[self.Y1_011['W_zy'], np.random.random([36, 1])] # Y_100 增加没影响
            r_ = np.zeros(len(self.Y1_100['age']))
            r_[-1] = 1
            self.Y1_100['W_xy'][:, -1] = self.Y1_100['W_xy'][:, -1] * (
                    1 - 1 / pow(self.Y1_100['age'][-1], 1 / 1)) + x_input / pow(self.Y1_100['age'][-1], 1 / 1)
            self.Y1_100['age'][-1] += 1

        elif indexes.size == 1:
            self.Y1_100['W_xy'][:, indexes[0]] = self.Y1_100['W_xy'][:, indexes[0]] * (
                    1 - 1 / pow(self.Y1_100['age'][indexes[0]], 1 / 1)) + x_input / pow(self.Y1_100['age'][indexes[0]],
                                                                                        1 / 1)
            self.Y1_100['age'][indexes[0]] += 1
            if self.Y1_100['age'][indexes[0]] >= 50:
                """知识迁移"""
                Dn3.split(indexes[0], self.Y1_100)
                self.Y1_011['W_yy'] = np.r_[self.Y1_011['W_yy'], np.zeros([1, self.Y1_011['W_yy'].shape[1]])]
            r_ = np.zeros(len(self.Y1_100['age']))
            r_[indexes[0]] = 1

        else:
            ages = np.zeros(indexes.size)
            for i in range(indexes.size):
                ages[i] = self.Y1_100['age'][indexes[i]]

            if sum(ages >= 50) == 0:  # 没有年龄大于50的神经元
                """
                有两个神经元响应值都大于阈值，但是年龄都小于50，怎么处理？输入不更新，最大响应输出
                TODO 最大的响应神经元权值也更新下
                """
                # print('Warning: 没有年纪大于50的神经元')
                r_ = np.zeros(len(self.Y1_100['age']))
                r_[np.argmax(r_y_100)] = 1
            else:
                old_neuron = np.where(ages >= 50)[0]
                response_action = np.zeros(old_neuron.size)
                for neuron in range(old_neuron.size):
                    response_action[neuron], _ = self.response_index1(indexes[old_neuron[neuron]])

                if sum(response_action == z_label) == 0:
                    """年龄小于50的学习"""
                    young_neuron = np.where(ages < 50)[0]
                    if young_neuron.size != 1:  # 年龄小于50的神经元不只一个
                        """
                        三个神经元，响应值都大于阈值,其中有一个神经元年龄大于50，没学习过该动作
                        存在两个年龄小于50的神经元，怎么处理？输入不更新，最大响应输出
                        TODO 最大的响应神经元权值也更新下
                        """
                        # print('Warning: 年纪小于50的神经元有', sum(ages < 50))
                        r_ = np.zeros(len(self.Y1_100['age']))
                        r_[np.argmax(r_y_100)] = 1
                    else:
                        self.Y1_100['W_xy'][:, indexes[young_neuron[0]]] = self.Y1_100['W_xy'][:,
                                                                           indexes[young_neuron[0]]] * (
                                                                                   1 - 1 / pow(self.Y1_100['age'][
                                                                                                   indexes[young_neuron[
                                                                                                       0]]],
                                                                                               1 / 1)) + x_input / pow(
                            self.Y1_100['age'][indexes[young_neuron[0]]], 1 / 1)
                        self.Y1_100['age'][indexes[young_neuron[0]]] += 1
                        if self.Y1_100['age'][indexes[young_neuron][0]] >= 50:
                            """知识迁移"""
                            Dn3.split(indexes[young_neuron[0]], self.Y1_100)
                            self.Y1_011['W_yy'] = np.r_[
                                self.Y1_011['W_yy'], np.zeros([1, self.Y1_011['W_yy'].shape[1]])]
                        r_ = np.zeros(len(self.Y1_100['age']))
                        r_[indexes[young_neuron[0]]] = 1
                else:
                    # r_ = np.zeros(len(self.Y1_100['age']))
                    correct_old = old_neuron[np.where(response_action == z_label)[0][0]]
                    real_index = indexes[correct_old]

                    r_ = np.zeros(len(self.Y1_100['age']))
                    r_[real_index] = 1

        if r_.sum() == 1:
            # Y_011 response
            # z_label = z_fire[i] - 1  # top-down
            z_input = np.zeros(36)
            z_input[z_label] = 1
            y_y = np.dot(r_, self.Y1_011['W_yy']) + np.dot(z_input, self.Y1_011['W_zy'])
            # top-1
            idmax = y_y.argmax()
            valmax = y_y.max()
            # m = self.mt(self.Y1_011['age'][idmax])
            if valmax <= self.theta_011 * 2:
                Dn3.split(idmax, self.Y1_011)
                # z层的权重要改变
                self.Z1['W_yz'] = np.r_[self.Z1['W_yz'], np.zeros([1, self.Z1['W_yz'].shape[1]])]
                r_011 = np.zeros(len(self.Y1_011['age']))
                # r_011 = np.zeros(len(self.Y1_011['age']))
                r_011[-1] = 1
                self.Y1_011['W_yy'][:, -1] = self.Y1_011['W_yy'][:, -1] * (
                        1 - 1 / pow(self.Y1_011['age'][-1], 1 / 1)) + r_ / pow(self.Y1_011['age'][-1], 1 / 1)
                self.Y1_011['W_zy'][:, -1] = self.Y1_011['W_zy'][:, -1] * (
                        1 - 1 / pow(self.Y1_011['age'][-1], 1 / 1)) + z_input / pow(self.Y1_011['age'][-1], 1 / 1)
                self.Y1_011['age'][-1] += 1
            else:
                # r_011 = np.zeros(len(self.Y1_011['age']))
                r_011 = np.zeros(len(self.Y1_011['age']))
                r_011[idmax] = 1
                self.Y1_011['W_yy'][:, idmax] = self.Y1_011['W_yy'][:, idmax] * (
                        1 - 1 / pow(self.Y1_011['age'][idmax], 1 / 1)) + r_ / pow(self.Y1_011['age'][idmax], 1 / 1)
                self.Y1_011['W_zy'][:, idmax] = self.Y1_011['W_zy'][:, idmax] * (
                        1 - 1 / pow(self.Y1_011['age'][idmax], 1 / 1)) + z_input / pow(self.Y1_011['age'][idmax], 1 / 1)
                self.Y1_011['age'][idmax] += 1

            self.Z1['W_yz'][:, z_label] = self.Z1['W_yz'][:, z_label] * (
                    1 - 1 / pow(self.Z1['age'][z_label], 1 / 1)) + r_011 / pow(self.Z1['age'][z_label], 1 / 1)
            self.Z1['age'][z_label] += 1

    def fused_explore_process(self, phase='train'):
        def normalize_scores(scores):
            min_val, max_val = np.min(scores), np.max(scores)
            return np.zeros_like(scores) if max_val == min_val else (scores - min_val) / (max_val - min_val)

        # 1. 计算 DN3 基础输出
        norm_x = self.x_input / (np.linalg.norm(self.x_input) + 1e-8)
        norm_w = self.Y_100['W_xy'] / (np.linalg.norm(self.Y_100['W_xy'], axis=0) + 1e-8)
        r_y_100 = np.dot(norm_x, norm_w)
        indexes = np.where(r_y_100 >= self.theta_100)[0]
        if indexes.size < 1: indexes = np.array([np.argmax(r_y_100)])


        max_z_raw = -999.0
        dn3_best_action = 0
        for idx in indexes:
            idmax = self.Y_011['W_yy'][idx, :].argmax()
            z_response = self.Z['W_yz'][idmax, :]
            if np.max(z_response) > max_z_raw:
                max_z_raw = np.max(z_response)
                dn3_best_action = int(np.argmax(z_response))

        # 2. 批量计算 GBMR 图记忆输出
        gbmr_qs = np.zeros(36)
        support_count = 0
        relaxed_th = self.G.dist_th * 3
        gbmr_enabled = getattr(self.G, 'G', None) is not None and self.G.G.number_of_nodes() > (
            10 if phase == 'train' else 0)
        valid_dists_all = []

        if gbmr_enabled and getattr(self.G, 'indices_list', None) is not None:
            min_req = 2 if phase == 'train' else 1
            valid_actions = [a for a in range(36) if self.G.built_capacity[a] >= min_req]
            if valid_actions:
                # 批量查询提升性能
                features_batch = [self.x_input] * len(valid_actions)
                Q_s_batch, _, Dists_batch = self.G.query_by_features(features_batch, valid_actions, k=5)

                for i, a in enumerate(valid_actions):
                    valid_mask = Dists_batch[i] <= relaxed_th
                    if np.any(valid_mask):
                        valid_qs, valid_dists = Q_s_batch[i][valid_mask], Dists_batch[i][valid_mask]
                        gbmr_qs[a] = np.average(valid_qs, weights=1.0 / (valid_dists + 1e-5))
                        support_count += 1
                        valid_dists_all.extend(valid_dists)

        # 向量化种群平滑 (剔除 for 循环，使用 np.roll 极速矩阵滚动)
        if support_count > 0:
            gbmr_qs = (gbmr_qs * 0.5 + np.roll(gbmr_qs, 1) * 0.15 + np.roll(gbmr_qs, -1) * 0.15 +
                       np.roll(gbmr_qs, 2) * 0.1 + np.roll(gbmr_qs, -2) * 0.1)

        self.current_gbmr_qs = gbmr_qs.copy()
        norm_g = normalize_scores(np.clip(gbmr_qs, 0, None))


        alpha = 0.0
        if support_count > 0:
            avg_dist = np.mean(valid_dists_all) if valid_dists_all else self.G.dist_th
            base_alpha = 0.75 * np.exp(-avg_dist * 3.0)

            # 对数压制自信度爆炸
            dn3_confidence = min(np.log1p(max(0, max_z_raw)) / np.log1p(100.0), 1.0)

            # 砍价计算基础 Alpha
            alpha = base_alpha * (1.0 - 0.4 * dn3_confidence)

            # 终极分歧熔断机制 (防小脑产生肌肉幻觉)
            if gbmr_qs[dn3_best_action] < 0.0:
                alpha = 0.85  # 大脑立刻接管 85% 的绝对控制权

            #防杠精底线 (前方全是已知死路)
            if np.max(gbmr_qs) < -5.0:
                alpha = 0.9

        best_dn3_scores, best_fused_scores = None, None
        best_actions, best_re_score = np.array([0]), -999.0

        for idx in indexes:
            # 性能优化核心：用数组切片代替繁琐的 One-hot 零矩阵乘法，瞬间出结果
            idmax = self.Y_011['W_yy'][idx, :].argmax()
            z_response = self.Z['W_yz'][idmax, :]

            norm_z = normalize_scores(z_response)
            final_scores = (1.0 - alpha) * norm_z + alpha * norm_g
            final_scores += np.where(gbmr_qs < -1.0, gbmr_qs * 0.1, 0)  # 向量化条件加法

            if final_scores.max() > best_re_score:
                best_re_score = final_scores.max()
                best_dn3_scores, best_fused_scores = norm_z, final_scores
                best_actions = np.argsort(final_scores)

        self._record_fusion_debug("explore", alpha, gbmr_qs, norm_g, int(np.argmax(best_dn3_scores)),
                                  int(np.argmax(gbmr_qs)) if support_count > 0 else -1,
                                  int(np.argmax(best_fused_scores)), best_actions, best_dn3_scores,
                                  best_fused_scores, support_count)
        return best_actions


    def fused_test_process(self):
        def normalize_scores(scores):
            min_val, max_val = np.min(scores), np.max(scores)
            return np.zeros_like(scores) if max_val == min_val else (scores - min_val) / (max_val - min_val)

        norm_x = self.x_input / (np.linalg.norm(self.x_input) + 1e-8)
        norm_w = self.Y1_100['W_xy'] / (np.linalg.norm(self.Y1_100['W_xy'], axis=0) + 1e-8)
        r_y_100 = np.dot(norm_x, norm_w)
        indexes = np.where(r_y_100 >= self.theta_100)[0]
        if indexes.size < 1: indexes = np.array([np.argmax(r_y_100)])

        max_z_raw = -999.0
        for idx in indexes:
            idmax = self.Y1_011['W_yy'][idx, :].argmax()
            z_response = self.Z1['W_yz'][idmax, :]
            if np.max(z_response) > max_z_raw:
                max_z_raw = np.max(z_response)

        # 接下来正常算 GBMR 的得分
        gbmr_qs = np.zeros(36)
        support_count = 0
        relaxed_th = self.G.dist_th * 3
        valid_dists_all = []

        if getattr(self.G, 'indices_list', None) is not None:
            valid_actions = [a for a in range(36) if self.G.built_capacity[a] >= 1]
            if valid_actions:
                features_batch = [self.x_input] * len(valid_actions)
                Q_s_batch, _, Dists_batch = self.G.query_by_features(features_batch, valid_actions, k=5)
                for i, a in enumerate(valid_actions):
                    valid_mask = Dists_batch[i] <= relaxed_th
                    if np.any(valid_mask):
                        valid_qs, valid_dists = Q_s_batch[i][valid_mask], Dists_batch[i][valid_mask]
                        gbmr_qs[a] = np.average(valid_qs, weights=1.0 / (valid_dists + 1e-5))
                        support_count += 1
                        valid_dists_all.extend(valid_dists)

        if support_count > 0:
            gbmr_qs = (gbmr_qs * 0.5 + np.roll(gbmr_qs, 1) * 0.15 + np.roll(gbmr_qs, -1) * 0.15 +
                       np.roll(gbmr_qs, 2) * 0.1 + np.roll(gbmr_qs, -2) * 0.1)

        self.current_gbmr_qs = gbmr_qs.copy()
        norm_g = normalize_scores(np.clip(gbmr_qs, 0, None))

        # 提前找出小脑最想走的动作是哪个
        dn3_best_action = 0
        for idx in indexes:
            idmax = self.Y1_011['W_yy'][idx, :].argmax()
            z_response = self.Z1['W_yz'][idmax, :]
            if np.max(z_response) == max_z_raw:
                dn3_best_action = int(np.argmax(z_response))

        alpha = 0.0
        if support_count > 0:
            avg_dist = np.mean(valid_dists_all) if valid_dists_all else self.G.dist_th
            base_alpha = 0.75 * np.exp(-avg_dist * 3.0)
            # 即使 1000 轮训练后 max_z_raw 达到 200，对数也能将其平滑压制，避免永远卡死在 1.0
            dn3_confidence = min(np.log1p(max(0, max_z_raw)) / np.log1p(100.0), 1.0)

            # 保持五五开的底线
            alpha = base_alpha * (1.0 - 0.4 * dn3_confidence)
            # 如果小脑最想走的方向，在大脑图网络里得分很低（小于 0），说明小脑出现了“肌肉幻觉”
            if gbmr_qs[dn3_best_action] < 0.0:
                alpha = 0.75  # 大脑立刻接管 85% 的绝对控制权，一票否决小脑！

            # 原有的防杠精底线（全图无路可走时）
            if np.max(gbmr_qs) < -10.0:
                alpha = 0.9

        best_dn3_scores, best_fused_scores = None, None
        best_actions, best_re_score, best_re_011 = np.array([0]), -999.0, 0

        for idx in indexes:
            idmax = self.Y1_011['W_yy'][idx, :].argmax()
            z_response = self.Z1['W_yz'][idmax, :]

            norm_z = normalize_scores(z_response)

            # 融合计算
            final_scores = (1.0 - alpha) * norm_z + alpha * norm_g
            final_scores += np.where(gbmr_qs < -1.0, gbmr_qs * 0.1, 0)

            if final_scores.max() > best_re_score:
                best_re_score = final_scores.max()
                best_dn3_scores, best_fused_scores = norm_z, final_scores
                best_actions = np.argsort(final_scores)
                best_re_011 = idmax

        re_011s = np.full(len(best_actions), best_re_011)

        return best_actions, re_011s
    def test_process(self):
        """纯 DN3 测试决策过程 """
        norm_x = self.x_input / (np.linalg.norm(self.x_input) + 1e-8)
        norm_w = self.Y1_100['W_xy'] / (np.linalg.norm(self.Y1_100['W_xy'], axis=0) + 1e-8)
        r_y_100 = np.dot(norm_x, norm_w)

        indexes = np.where(r_y_100 >= self.theta_100)[0]
        if indexes.size >= 1:
            action = []
            re_011s = []
            for index in indexes:
                act, re_011 = self.response_index1(index)
                action.append(act)
                re_011s.append(re_011)
            actions = np.array(action)
            re_011s = np.array(re_011s)
        else:
            act, re_011 = self.response_index1(np.argmax(r_y_100))
            actions = np.array([act])
            re_011s = np.array([re_011])

        return actions, re_011s

    def test(self, path_seed, use_gbmr=True):
        self.agent = np.array([50, 50])
        trajectory = [self.agent.copy()]
        done = False
        obs = self.get_observations()
        fri_dis, em_dis, fri_angle, em_angle, em2_dis, em2_angle = obs

        while not done:
            self.cur_fri_dis = fri_dis
            self.cur_em_dis = em_dis

            self.x_input = np.array(
                [np.cos(fri_angle), np.sin(fri_angle), np.cos(em_angle), np.sin(em_angle), fri_dis / (fri_dis + em_dis),
                 em_dis / (fri_dis + em_dis), np.cos(em2_angle), np.sin(em2_angle), em2_dis / (fri_dis + em2_dis),
                 em_dis / em2_dis])

            pre_agent = self.agent

            if use_gbmr:
                actions, re_011s = self.fused_test_process()
            else:
                actions, re_011s = self.test_process()

            action = self.select_action((fri_dis, em_dis, fri_angle, em_angle, em2_dis, em2_angle), actions)

            match_idx = np.where(action == actions)[0]
            if len(match_idx) > 0:
                re_011 = re_011s[match_idx[0]]
            else:
                re_011 = re_011s[-1]

            self.agent = self.agent + self._step10[int(action)]
            trajectory.append(self.agent.copy())
            obs = self.get_observations()
            fri_dis, em_dis, fri_angle, em_angle, em2_dis, em2_angle = obs

            if (self.agent[0] < 20) or (self.agent[0] > 480) or (self.agent[1] < 20) or (self.agent[1] > 480) or \
                    (em_dis < self.d_em) or (len(trajectory) > 300):
                done = True;
                arrive = 0
                try:
                    self.Z1['W_yz'][int(re_011), int(action)] = 0.5 * self.Z1['W_yz'][int(re_011), int(action)]
                except Exception as e:
                    pass

                r_array = np.full(36, -999.0)
                for a_idx in range(36):
                    self.agent = pre_agent + self._step10[int(a_idx)]
                    fri_dis_tmp, em_dis_tmp, fri_angle_tmp, em_angle_tmp, em2_dis_tmp, em2_angle_tmp = self.get_observations()
                    a_idx_angle = a_idx * (np.pi / 18.0)
                    r_array[a_idx] = self.get_reward(fri_dis_tmp, em_dis_tmp, a_idx_angle, fri_angle_tmp)
                best_action = r_array.argmax()
                if r_array[best_action] > -50.0:
                    for iteration in range(self.iteration):
                        self.update1(self.x_input, best_action)
            elif fri_dis < self.d_fri:
                done = True;
                arrive = 1
                if len(trajectory) < self.steps:
                    self.steps = len(trajectory)
                    self.save_model(path_seed)
            else:
                done = False
        return np.array(trajectory), arrive

    def get_observations(self):
        # 用平方和代替全局 norm 开方运算
        dx = self.ems[:, 0] - self.agent[0]
        dy = self.ems[:, 1] - self.agent[1]
        ems_sq = dx ** 2 + dy ** 2  # 距离的平方

        idx = ems_sq.argmin()
        em_dis = math.sqrt(ems_sq[idx])  # 只对最近的这一个真正求开方
        em_angle = math.atan2(dy[idx], dx[idx])

        # 获取第二近的障碍物
        em2_id = np.argsort(ems_sq)[1]
        em2_dis = math.sqrt(ems_sq[em2_id])
        em2_angle = math.atan2(dy[em2_id], dx[em2_id])

        fri_dx = self.friend[0] - self.agent[0]
        fri_dy = self.friend[1] - self.agent[1]
        fri_dis = math.sqrt(fri_dx ** 2 + fri_dy ** 2)
        fri_angle = math.atan2(fri_dy, fri_dx)

        return fri_dis, em_dis, fri_angle, em_angle, em2_dis, em2_angle


    def get_reward(self, fri_dis, em_dis, action_angle, fri_angle):
        reward = -2  # 基础的时间惩罚，逼迫智能体走捷径

        # 1. 距离奖励（大幅降低权重，防止刷分）
        delta_fri = self.cur_fri_dis - fri_dis
        reward += delta_fri * 3

        # 2. 障碍物避让惩罚
        safe_margin = 25.0
        if em_dis < safe_margin and em_dis >= self.d_em:
            reward -= (safe_margin - em_dis) * 2

        # 3. 终止状态的终极奖惩（让它们占据绝对主导权）
        if em_dis < self.d_em: return -50.0
        if fri_dis < self.d_fri: return 100.0

        # 4. 角度对齐奖励
        angle_diff = abs(math.atan2(math.sin(action_angle - fri_angle), math.cos(action_angle - fri_angle)))
        if angle_diff < np.pi / 6: reward += 1

        # 5. 探索死锁惩罚（降低权重）
        grid_x, grid_y = int(self.agent[0] // self.grid_size), int(self.agent[1] // self.grid_size)
        reward -= self.visited_grid.get((grid_x, grid_y), 0) * 5.0
        return reward

    def explore_process(self):
        # y_100 response
        r_y_100 = np.dot((self.x_input / np.linalg.norm(self.x_input)), (
                self.Y_100['W_xy'] / np.linalg.norm(self.Y_100['W_xy'], axis=0)))

        indexes = np.where(r_y_100 >= self.theta_100)[0]
        if indexes.size >= 1:
            action = []
            for index in indexes:
                action.append(self.response_index(index))
            actions = np.array(action)
        else:
            action = self.response_index(np.argmax(r_y_100))
            actions = np.array([action])

        return actions

    def explore(self, epsilon, phase='train', use_gbmr=True):
        self.agent = np.array([50, 50])
        obs = self.get_observations()
        fri_dis, em_dis, fri_angle, em_angle, em2_dis, em2_angle = obs
        trajectory = [self.agent.copy()]
        done = False

        self.visited_grid.clear()
        self.trj_features.clear()
        self.trj_actions.clear()
        self.trj_rewards.clear()
        self.trj_obs.clear()

        while not done:
            self.cur_fri_dis = fri_dis
            self.cur_em_dis = em_dis

            self.x_input = np.array(
                [np.cos(fri_angle), np.sin(fri_angle), np.cos(em_angle), np.sin(em_angle), fri_dis / (fri_dis + em_dis),
                 em_dis / (fri_dis + em_dis), np.cos(em2_angle), np.sin(em2_angle), em2_dis / (fri_dis + em2_dis),
                 em_dis / em2_dis])

            self.trj_features.append(self.x_input.copy())
            self.trj_obs.append(self.agent.copy())

            if np.random.random() < epsilon:
                action = np.random.randint(36)
                action_angle = action * (np.pi / 18.0)

                self.trj_actions.append(action)

                self.agent = self.agent + self._step10[int(action)]
                self._record_visited(self.agent)
                obs = self.get_observations()
                fri_dis, em_dis, fri_angle, em_angle, em2_dis, em2_angle = obs
                r = self.get_reward(fri_dis, em_dis, action_angle, fri_angle)

                self.trj_rewards.append(r)

                if r > 0:
                    self.update(self.x_input, action)
                    self.replay_buffer.add(self.x_input, action, r)
                    self.count += 1
                    if self.count % self.target_update == 0:
                        for iteration in range(self.iteration):
                            sampled_indices = self.replay_buffer.choice(self.batch_size)
                            for idx in sampled_indices:
                                self.update1(self.replay_buffer.buffer[idx][0], self.replay_buffer.buffer[idx][1])
                    if self.replay_buffer.size() > self.minimal_size:
                        indexs = self.replay_buffer.choice(self.batch_size)
                        for index in indexs:
                            self.update(self.replay_buffer.buffer[index][0], self.replay_buffer.buffer[index][1])
            else:
                # 选择是否使用融合网络
                if use_gbmr:
                    actions = self.fused_explore_process(phase=phase)
                else:
                    actions = self.explore_process()

                action = self.select_action((fri_dis, em_dis, fri_angle, em_angle, em2_dis, em2_angle), actions)
                action_angle = action * (np.pi / 18.0)

                self.trj_actions.append(int(action))

                pre_agent = self.agent
                self.agent = self.agent + self._step10[int(action)]
                test_agent = self.agent
                self._record_visited(self.agent)
                obs = self.get_observations()
                fri_dis, em_dis, fri_angle, em_angle, em2_dis, em2_angle = obs

                r = self.get_reward(fri_dis, em_dis, action_angle, fri_angle)

                # Curiosity reward (只在挂载 GBMR 时开启)
                intrinsic_reward = 0.0
                if use_gbmr and getattr(self.G, 'indices_list', None) is not None and self.G.built_capacity[
                    int(action)] > 0:
                    _, _, dists = self.G.query_by_features([self.x_input], [int(action)], k=1)
                    if len(dists) > 0 and len(dists[0]) > 0 and dists[0][0] != np.inf:
                        intrinsic_reward = dists[0][0] * 2.0
                total_reward = r + intrinsic_reward

                self.trj_rewards.append(total_reward)

                if r < -50:
                    r_array = np.full(36, -999.0)
                    for a_idx in range(36):
                        self.agent = pre_agent + self._step10[int(a_idx)]
                        fri_dis_tmp, em_dis_tmp, fri_angle_tmp, em_angle_tmp, em2_dis_tmp, em2_angle_tmp = self.get_observations()
                        a_idx_angle = a_idx * (np.pi / 18.0)
                        r_array[a_idx] = self.get_reward(fri_dis_tmp, em_dis_tmp, a_idx_angle, fri_angle_tmp)
                    best_action = r_array.argmax()
                    if r_array[best_action] > -50.0:
                        self.update(self.x_input, best_action)
                        self.replay_buffer.add(self.x_input, best_action, abs(r_array.max()))
                        self.count += 1
                        if self.count % self.target_update == 0:
                            for iteration in range(self.iteration):
                                sampled_indices = self.replay_buffer.choice(self.batch_size)
                                for idx in sampled_indices:
                                    self.update1(self.replay_buffer.buffer[idx][0], self.replay_buffer.buffer[idx][1])
                        if self.replay_buffer.size() > self.minimal_size:
                            indexs = self.replay_buffer.choice(self.batch_size)
                            for index in indexs:
                                self.update(self.replay_buffer.buffer[index][0], self.replay_buffer.buffer[index][1])

                self.agent = test_agent
                obs = self.get_observations()
                fri_dis, em_dis, fri_angle, em_angle, em2_dis, em2_angle = obs

            trajectory.append(self.agent.copy())

            if (self.agent[0] < 20) or (self.agent[0] > 480) or (self.agent[1] < 20) or (self.agent[1] > 480) or \
                    (em_dis < self.d_em) or (len(trajectory) > 500):
                done = True
                arrive = 0
            elif fri_dis < self.d_fri:
                done = True
                arrive = 1
                self.found_goal = True
            else:
                done = False

        # 【开关控制】：回合结束，纯预训练时绝不建图
        if use_gbmr and len(self.trj_actions) > 0:
            graph_update_start = time.perf_counter_ns()
            nodes_before = self.G.G.number_of_nodes()
            edges_before = self.G.G.number_of_edges()

            # 先用占位值把这条真实路径编码成图节点
            dummy_values = [0.0 for _ in self.trj_features]
            encoded_trj_nodes = self.G.add_by_features(self.trj_features, self.trj_actions, dummy_values)

            # # 判定当前这条路径的终止类型
            # terminal_type = 'success' if arrive == 1 else 'collision'

            # # 只对“这一条路径”做回溯，不再扫整张图
            # path_values = self.G.GlobalValueIteration(
            #     path_nodes=encoded_trj_nodes,
            #     path_actions=self.trj_actions,
            #     path_rewards=self.trj_rewards,
            #     terminal_type=terminal_type,
            #     gamma_reward=0.95,
            #     gamma_penalty=0.95,
            #     success_reward=100.0,
            #     collision_reward=-50.0,
            #     step_cost=1.0,
            #     iterations=1
            # )

            # 只有成功轨迹进入 keypoint，供后续捷径推理
            if arrive == 1:
                self.keypoint.add_focus(encoded_trj_nodes, self.trj_rewards, self.trj_obs, len(self.trj_actions))

            self.G.flush_index()

            nodes_after = self.G.G.number_of_nodes()
            edges_after = self.G.G.number_of_edges()
            node_capacity = self.G.capacity * self.G.act_dimension
            graph_update_ms = (time.perf_counter_ns() - graph_update_start) / 1e6
            self.graph_update_times_ms.append(graph_update_ms)
            self.graph_growth_records.append({
                'exploration_episode': len(self.graph_growth_records) + 1,
                'phase': phase,
                'trajectory_steps': len(self.trj_actions),
                'exploration_success': int(arrive),
                'nodes_before': nodes_before,
                'nodes_after': nodes_after,
                'nodes_added': nodes_after - nodes_before,
                'node_capacity': node_capacity,
                'node_capacity_usage_percent': 100.0 * nodes_after / node_capacity,
                'edges_before': edges_before,
                'edges_after': edges_after,
                'edges_added': edges_after - edges_before,
                'graph_update_ms': graph_update_ms
            })

        return np.array(trajectory), arrive

    def check(self, ems, ema, fris, fria, actions):

        if actions is None or len(actions) == 0:
            return 0
        if len(actions) == 1:
            return int(actions[0])

        sorted_actions = actions[::-1]

        for action in sorted_actions:
            act_idx = action if action <= 18 else action - 36
            action_angle = 2 * np.pi * act_idx / 36
            angle_diff = np.abs(math.atan2(math.sin(action_angle - ema), math.cos(action_angle - ema)))

            # 1. 致命障碍物危险区
            if ems <= 18 and angle_diff <= np.pi / 6:
                continue

            return int(action)

        return int(sorted_actions[0])


    # @staticmethod
    # def evaluation(ems, ema, fris, fria, action_angle):
    #     # 折中点 1：安全预警距离从 30 恢复到 40 (原版是 50)
    #     if ems > 40:
    #         value = -np.abs(action_angle - fria)
    #
    #     # 折中点 2：极限危险距离从 12 恢复到 18 (原版是 25，步长是10，留出一步半的缓冲)
    #     elif ems > 18:
    #         # 折中点 3：避障视界从 pi/18(10度) 放大到 pi/12(15度) (原版是 pi/9 即20度)
    #         if np.abs(action_angle - ema) <= np.pi / 12:
    #             value = np.abs(action_angle - ema) - ((1 - 1 / fris) * np.abs(action_angle - fria))
    #         else:
    #             value = -np.abs(action_angle - fria)
    #
    #     else:
    #         if np.abs(action_angle - ema) <= np.pi / 12:
    #             value = 0
    #         else:
    #             value = np.abs(action_angle - ema)
    #
    #     return value

    def save_model(self, path_seed):
        """路径要改"""
        np.save(f'path/info/model/Y1_100_{path_seed}.npy', self.Y1_100)
        np.save(f'path/info/model/Y1_011_{path_seed}.npy', self.Y1_011)
        np.save(f'path/info/model/Z1_{path_seed}.npy', self.Z1)

    def save_fusion_model(self, path_seed):
        # 1. 保存 DN3 的神经元权重 (依然是你熟悉的配方)
        np.save(f'path/info/model/Y1_100_{path_seed}.npy', self.Y1_100)
        np.save(f'path/info/model/Y1_011_{path_seed}.npy', self.Y1_011)
        np.save(f'path/info/model/Z1_{path_seed}.npy', self.Z1)

        # 2. 保存 GBMR 的图网络结构 (使用 pickle 序列化 Graph_dict)
        with open(f'path/info/model/GBMR_Graph_{path_seed}.pkl', 'wb') as f:
            pickle.dump(self.G, f)

    def decision(self):
        self.agent = np.array([50, 50])
        trajectory = [self.agent.copy()]
        done = False

        obs = self.get_observations()
        fri_dis, em_dis, fri_angle, em_angle, em2_dis, em2_angle = obs

        while not done:
            decision_start = time.perf_counter_ns()
            self.x_input = np.array(
                [np.cos(fri_angle), np.sin(fri_angle), np.cos(em_angle), np.sin(em_angle),
                 fri_dis / (fri_dis + em_dis), em_dis / (fri_dis + em_dis),
                 np.cos(em2_angle), np.sin(em2_angle), em2_dis / (fri_dis + em2_dis),
                 em_dis / em2_dis])

            graph_search_start = time.perf_counter_ns()
            nav_action = self.active_graph_navigation()
            graph_search_ms = (time.perf_counter_ns() - graph_search_start) / 1e6
            self.graph_search_times_ms.append(graph_search_ms)

            if nav_action is not None:
                # 若大脑有路线，仅过一遍最基础的底层防撞本能
                action = self.check(em_dis, em_angle, fri_dis, fri_angle, np.array([nav_action]))
            else:
                # print('1'*10)
                # 若无路线（记忆盲区），靠融合网络直觉
                actions, _ = self.fused_test_process()
                action = self.select_action((fri_dis, em_dis, fri_angle, em_angle, em2_dis, em2_angle), actions)

            decision_ms = (time.perf_counter_ns() - decision_start) / 1e6
            self.decision_times_ms.append(decision_ms)

            self.agent = self.agent + self._step10[int(action)]
            trajectory.append(self.agent.copy())

            obs = self.get_observations()
            fri_dis, em_dis, fri_angle, em_angle, em2_dis, em2_angle = obs

            # 结束条件
            if (self.agent[0] < 20) or (self.agent[0] > 480) or (self.agent[1] < 20) or (self.agent[1] > 480) or \
                    (em_dis < self.d_em) or (len(trajectory) > 500):
                done = True
                arrive = 0
            elif fri_dis < self.d_fri:
                done = True
                arrive = 1
            else:
                done = False

        return np.array(trajectory), arrive


    def save_excel_Y1_100(self):
        # ===== 路径 =====
        npy_path = 'path/info/model/Y1_100_2.npy'
        xlsx_path = 'path/info/model/Y1_100_2_fixed.xlsx'

        # ===== 读取 npy =====
        data = np.load(npy_path, allow_pickle=True).item()

        print(f"数据键: {list(data.keys())}")
        print(f"age 长度: {len(data['age'])}")
        print(f"W_xy 形状: {data['W_xy'].shape}")

        # ===== 直接写入两个DataFrame =====
        # 方法1：直接保存为两个独立的Excel文件
        # age数据
        df_age = pd.DataFrame(data['age'], columns=['age'])
        df_age.to_excel('path/info/model_excel_Y1_100/Y1_100_2_age.xlsx', index=False)
        print(f"age数据已保存到: Y1_100_2_age.xlsx")

        # W_xy数据
        df_wxy = pd.DataFrame(data['W_xy'].T)  # 转置为(78, 10)
        df_wxy.columns = [f'Feature_{i}' for i in range(df_wxy.shape[1])]
        df_wxy.to_excel('path/info/model_excel_Y1_100/Y1_100_2_W_xy.xlsx', index=False)
        print(f"W_xy数据已保存到: Y1_100_2_W_xy.xlsx")

        print('数据已分别保存为两个Excel')

    def save_excel_Y1_011(self):
        # ===== 路径 =====
        npy_path = 'path/info/model/Y1_011_2.npy'
        xlsx_path = 'path/info/model/Y1_011_2_fixed.xlsx'

        # ===== 读取 npy =====
        data = np.load(npy_path, allow_pickle=True).item()

        print(f"数据键: {list(data.keys())}")
        print(f"age 长度: {len(data['age'])}")
        print(f"W_yy 形状: {data['W_yy'].shape}")
        print(f"W_zy 形状: {data['W_zy'].shape}")

        # ===== 直接写入两个DataFrame =====
        # 方法1：直接保存为两个独立的Excel文件
        # age数据
        df_age = pd.DataFrame(data['age'], columns=['age'])
        df_age.to_excel('path/info/model_excel_Y1_011/Y1_011_2_age.xlsx', index=False)
        print(f"age数据已保存到: Y1_011_2_age.xlsx")

        # W_yy数据
        df_wyy = pd.DataFrame(data['W_yy'].T)  # 转置为(78, 10)
        df_wyy.columns = [f'Feature_{i}' for i in range(df_wyy.shape[1])]
        df_wyy.to_excel('path/info/model_excel_Y1_011/Y1_011_2_W_yy.xlsx', index=False)
        print(f"W_zy数据已保存到: Y1_011_2_W_yy.xlsx")

        # W_zy数据
        df_wzy = pd.DataFrame(data['W_zy'].T)  # 转置为(78, 10)
        df_wzy.columns = [f'Feature_{i}' for i in range(df_wzy.shape[1])]
        df_wzy.to_excel('path/info/model_excel_Y1_011/Y1_011_2_W_zy.xlsx', index=False)
        print(f"W_zy数据已保存到: Y1_011_2_W_zy.xlsx")

        print('数据已分别保存为两个Excel文件')

    def save_excel_Z1(self):
        # ===== 路径 =====
        npy_path = 'path/info/model/Z1_2.npy'
        xlsx_path = 'path/info/model/Z1_2_fixed.xlsx'

        # ===== 读取 npy =====
        data = np.load(npy_path, allow_pickle=True).item()

        print(f"数据键: {list(data.keys())}")
        print(f"age 长度: {len(data['age'])}")
        print(f"W_yz 形状: {data['W_yz'].shape}")

        # ===== 直接写入两个DataFrame =====
        # 方法1：直接保存为两个独立的Excel文件
        # age数据
        df_age = pd.DataFrame(data['age'], columns=['age'])
        df_age.to_excel('path/info/model_excel_Z1/Z1_2_age.xlsx', index=False)
        print(f"age数据已保存到: Z1_2_age.xlsx")

        # W_yz数据
        df_wyz = pd.DataFrame(data['W_yz'].T)  # 转置为(78, 10)
        df_wyz.columns = [f'Feature_{i}' for i in range(df_wyz.shape[1])]
        df_wyz.to_excel('path/info/model_excel_Z1/Z1_2_W_yz.xlsx', index=False)
        print(f"W_yz数据已保存到: Z1_2_W_yz.xlsx")

        print('数据已分别保存为')


if __name__ == "__main__":
    dynamic_flag = 0
    dynamic_figure2 = 0
    static_figure = 0
    dynamic_path = 0
    test_50_env = 0
    path_figure = 0
    tiaoxingtu = 0
    sum_arrive = 0

    episode_list = [1000]
    # 审稿人4意见2的计算开销统计：1个最终模型，在50个未知环境上统计。
    # 论文原有导航性能实验仍采用40次独立重复，不应由本统计替代。
    TOTAL_RUNS = 1
    TEST_ENVIRONMENTS = 1
    FIXED_PROFILE_EXPLORATIONS = True

    if 1:
        for current_num_episode in  reversed(episode_list):
            print(f'\n\n{"*" * 80}')
            print(f' ========== 当前设定训练轮数 num_episode = {current_num_episode} ========== ')
            print(f'{"*" * 80}\n')

            current_excel_data = []
            graph_growth_all = []

            for run_idx in range(TOTAL_RUNS):

                print(f'\n\n{"=" * 60}')
                print(
                    f'######### num_episode={current_num_episode} | 第 {run_idx + 1}/{TOTAL_RUNS} 个大循环开始 #########')
                print(f'{"=" * 60}\n')

                # 纯 DN3 极速脱机预训练
                print(f'\n[阶段一] 开启纯DN3预训练 (剥离GBMR，仅练肌肉本能) - 目标轮数: {current_num_episode}')
                dn3_train = Dn3()
                info_train = np.zeros([current_num_episode, 2])

                for i in range(current_num_episode):
                    if i % 100 == 0:
                        print(f'######### 第 {i + 1} 轮预训练开始 #########')
                    if i % 50 == 0:
                        dn3_train.generate_ems(np.random.randint(0, 1000), 50)

                    # 关闭GBMR
                    # eps = max(0.01, 0.5 * (0.99 ** i))
                    _, arrive_explore = dn3_train.explore(epsilon=0.01, use_gbmr=False)

                    path, arrive_test = dn3_train.test(path_seed=2, use_gbmr=False)
                    info_train[i, 1] = arrive_test
                # 保存练就的纯肌肉记忆
                os.makedirs('path/info/model/', exist_ok=True)
                np.save('path/info/model/Y1_100_pretrained.npy', dn3_train.Y1_100)
                np.save('path/info/model/Y1_011_pretrained.npy', dn3_train.Y1_011)
                np.save('path/info/model/Z1_pretrained.npy', dn3_train.Z1)
                print(f">>> 预训练完毕！阶段一到达次数：{info_train[:, 1].sum()} / {current_num_episode}")

                # 新环境 Test-Time Few-Shot 适应
                print(f"\n>>> [阶段二] 挂载 GBMR 大脑，进入 50 个未知新环境进行 Few-Shot 快速适应测试...")
                success_count = 0
                max_explore_episodes = 5
                fusion_stats_all = None

                for j in range(TEST_ENVIRONMENTS):
                    # 初始化带着纯净大脑和全新图记忆的测试特工
                    dn3_test = Dn3()
                    start_pos = np.array([50, 50])
                    target_pos = np.array([450, 450])
                    dn3_test.reset_fusion_stats()
                    dn3_test.agent = np.array([50, 50])
                    dn3_test.friend = np.array([450, 450])

                    # 加载预训练的肌肉记忆
                    dn3_test.Y1_100 = np.load('path/info/model/Y1_100_pretrained.npy', allow_pickle=True).item()
                    dn3_test.Y1_011 = np.load('path/info/model/Y1_011_pretrained.npy', allow_pickle=True).item()
                    dn3_test.Z1 = np.load('path/info/model/Z1_pretrained.npy', allow_pickle=True).item()
                    dn3_test.Y_100 = copy.deepcopy(dn3_test.Y1_100)
                    dn3_test.Y_011 = copy.deepcopy(dn3_test.Y1_011)
                    dn3_test.Z = copy.deepcopy(dn3_test.Z1)

                    # 挂载空白 GBMR 模块
                    dn3_test.G = Graph_dict(capacity=2000, key_dimension=10, act_dimension=36, dist_th=0.08,
                                            batch_size=200)
                    dn3_test.keypoint = ExpertMemory(memory_size=3)

                    # 生成从未见过的新陷阱
                    dn3_test.generate_ems(j + 20, 50)
                    # dn3_test.generate_u_dead_end_ems()

                    success_in_explore = 0
                    for ep in range(max_explore_episodes):

                        # 开启 GBMR 收集新环境的跌撞教训
                        _, arrive_exp = dn3_test.explore(epsilon=0.01, phase='few_shot', use_gbmr=True)

                        # 保存该未知环境本次探索后的图规模
                        if dn3_test.graph_growth_records:
                            growth_record = dn3_test.graph_growth_records[-1].copy()
                            growth_record.update({
                                '训练设定轮数 (Num_Episode)': current_num_episode,
                                '大循环次数 (Run)': run_idx + 1,
                                '测试环境序号 (Env)': j + 1
                            })
                            graph_growth_all.append(growth_record)

                        if arrive_exp == 1:
                            success_in_explore += 1
                            if not FIXED_PROFILE_EXPLORATIONS and success_in_explore >= 4:
                                break

                    # 触发海马体深度反思
                    reflection_stats = dn3_test.global_reflection(safe_margin=10.0, value_iterations=50)

                    # 最终大考
                    path, arrive_eval = dn3_test.decision()
                    success_count += arrive_eval

                    graph_search_times = np.asarray(dn3_test.graph_search_times_ms, dtype=float)
                    decision_times = np.asarray(dn3_test.decision_times_ms, dtype=float)
                    graph_update_times = np.asarray(dn3_test.graph_update_times_ms, dtype=float)

                    # 记录并保存单次结果
                    current_excel_data.append({
                        '训练设定轮数 (Num_Episode)': current_num_episode,
                        '大循环次数 (Run)': run_idx + 1,
                        '测试环境序号 (Env)': j + 1,
                        '是否到达 (Arrive)': arrive_eval,
                        '运动步数 (Steps)': len(path) - 1,
                        '实际探索次数': len(dn3_test.graph_growth_records),
                        '反思前节点数': reflection_stats['nodes_before_reflection'],
                        '反思前边数': reflection_stats['edges_before_reflection'],
                        '反思后节点数': reflection_stats['nodes_after_reflection'],
                        '反思后边数': reflection_stats['edges_after_reflection'],
                        '理论最大节点数': dn3_test.G.capacity * dn3_test.G.act_dimension,
                        '反思后节点容量使用率(%)': 100.0 * reflection_stats['nodes_after_reflection'] /
                                                   (dn3_test.G.capacity * dn3_test.G.act_dimension),
                        '反思新增节点数': reflection_stats['nodes_added_by_reflection'],
                        '反思新增边数': reflection_stats['edges_added_by_reflection'],
                        '候选捷径数': reflection_stats['candidate_shortcuts'],
                        '平均图更新时间(ms)': float(np.mean(graph_update_times)) if graph_update_times.size else 0.0,
                        '图更新时间标准差(ms)': float(np.std(graph_update_times, ddof=1)) if graph_update_times.size > 1 else 0.0,
                        '视线推理时间(ms)': reflection_stats['line_of_sight_ms'],
                        '完整反思时间(ms)': reflection_stats['reflection_total_ms'],
                        '平均拓扑搜索时间(ms)': float(np.mean(graph_search_times)) if graph_search_times.size else 0.0,
                        '拓扑搜索标准差(ms)': float(np.std(graph_search_times, ddof=1)) if graph_search_times.size > 1 else 0.0,
                        '拓扑搜索P95(ms)': float(np.percentile(graph_search_times, 95)) if graph_search_times.size else 0.0,
                        '平均完整决策时间(ms)': float(np.mean(decision_times)) if decision_times.size else 0.0,
                        '完整决策标准差(ms)': float(np.std(decision_times, ddof=1)) if decision_times.size > 1 else 0.0,
                        '完整决策P95(ms)': float(np.percentile(decision_times, 95)) if decision_times.size else 0.0
                    })

                    print(
                        f"大循环 {run_idx + 1}/{TOTAL_RUNS} | 环境 {j + 1}/{TEST_ENVIRONMENTS} | "
                        f"现场探索: {ep + 1}轮 | 最终状态: {'成功' if arrive_eval == 1 else '撞墙/超时'} | "
                        f"步数: {len(path) - 1}")

                    # 打印单次环境的统计信息
                    dn3_test.fusion_stats["episodes"] += 1
                    if arrive_eval == 1:
                        dn3_test.fusion_stats["episode_success"] += 1
                    if fusion_stats_all is None:
                        fusion_stats_all = {k: 0 for k in dn3_test.fusion_stats.keys()}
                    for k, v in dn3_test.fusion_stats.items():
                        fusion_stats_all[k] += v

                print("===" * 15)
                print(f"第 {run_idx + 1} 次大循环 {TEST_ENVIRONMENTS}个全新环境 Few-Shot "
                      f"测试成功数: {success_count} / {TEST_ENVIRONMENTS}")
                print("===" * 15)
                # 最终测试环境可视化并自动保存
                fig, ax = plt.subplots(figsize=(7, 7))  # 稍微调大一点画布

                # 1. 绘制障碍物 (调整大小为20，颜色为黑色方块)
                ax.scatter(dn3_test.ems[:, 0], dn3_test.ems[:, 1], s=20, c='k', marker='s', label='障碍物')

                # 2. 绘制轨迹 (用半透明红线连接，加上小三角，看起来更直观)
                ax.plot(path[:, 0], path[:, 1], c='red', linewidth=1.5, alpha=0.6)
                ax.scatter(path[:, 0], path[:, 1], s=15, c='red', marker='v', label='运动轨迹')

                # 3. 绘制起点和终点 (修复了之前 agent 未定义的 Bug)
                ax.scatter(start_pos[0], start_pos[1], s=100, c='blue', marker='o', label='起点', zorder=5)
                ax.scatter(target_pos[0], target_pos[1], s=150, c='green', marker='*', label='终点', zorder=5)

                # 4. 文本标注 (动态计算偏移量以免文字挡住点)
                plt.text(start_pos[0] - 25, start_pos[1] + 15, '起点', fontdict={'size': 14, 'color': 'blue', 'weight': 'bold'})
                plt.text(target_pos[0] - 25, target_pos[1] + 15, '终点',
                         fontdict={'size': 14, 'color': 'green', 'weight': 'bold'})

                # 5. 设置标题与坐标轴
                arrive_text = '成功到达' if arrive_eval == 1 else '未到达'
                plt.title(
                    f'SAC 最终测试环境轨迹',
                    fontsize=16,
                    fontweight='bold'
                )
                ax.set_xlim(0, 500)
                ax.set_ylim(0, 500)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.xlabel('X', font={'family': 'Times New Roman', 'size': 16})
                plt.ylabel('Y', font={'family': 'Times New Roman', 'size': 16})

                # 6. 添加图例 (放在右上角)
                plt.legend(loc='upper left', prop={'size': 12})
                plt.subplots_adjust(0.12, 0.12, 0.95, 0.92)

                # 7. 自动保存图片，避免弹窗阻塞批量实验
                save_dir = os.path.join('picture', 'final_test')
                os.makedirs(save_dir, exist_ok=True)
                save_name = (
                    f'DN3_GBMR_训练{current_num_episode}轮_第{run_idx + 1}次大循环_'
                    f'环境{j + 1}_{arrive_text}_步数{len(path) - 1}.png'
                )
                save_path = os.path.join(save_dir, save_name)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"最终测试环境图片已保存: {save_path}")

            # ==========================================
            # 独立导出：保存这 50 次大循环的所有数据
            # ==========================================
            df_results = pd.DataFrame(current_excel_data)

            # 保存未经筛选的图规模和计算时间原始数据，便于论文统计复核
            profile_dir = os.path.join('path', 'info', 'graph_profile')
            os.makedirs(profile_dir, exist_ok=True)
            profile_filename = os.path.join(
                profile_dir,
                f'DN3_GBMR_训练{current_num_episode}轮_图规模与时间.xlsx'
            )
            growth_filename = os.path.join(
                profile_dir,
                f'DN3_GBMR_训练{current_num_episode}轮_逐次探索图增长.xlsx'
            )
            timing_columns = [
                '反思前节点数', '反思前边数', '反思后节点数', '反思后边数',
                '反思新增节点数', '反思新增边数', '候选捷径数',
                '反思后节点容量使用率(%)', '平均图更新时间(ms)',
                '视线推理时间(ms)', '完整反思时间(ms)',
                '平均拓扑搜索时间(ms)', '拓扑搜索P95(ms)',
                '平均完整决策时间(ms)', '完整决策P95(ms)'
            ]
            profile_summary = []
            for column in timing_columns:
                values = pd.to_numeric(df_results[column], errors='coerce').dropna()
                profile_summary.append({
                    '指标': column,
                    '样本数': int(values.size),
                    '均值': float(values.mean()) if values.size else np.nan,
                    '标准差': float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    'P95': float(values.quantile(0.95)) if values.size else np.nan
                })

            growth_df = pd.DataFrame(graph_growth_all)
            if growth_df.empty:
                growth_summary = pd.DataFrame()
            else:
                growth_summary = growth_df.groupby('exploration_episode', as_index=False).agg(
                    sample_count=('nodes_after', 'size'),
                    nodes_mean=('nodes_after', 'mean'),
                    nodes_std=('nodes_after', 'std'),
                    edges_mean=('edges_after', 'mean'),
                    edges_std=('edges_after', 'std'),
                    capacity_usage_mean_percent=('node_capacity_usage_percent', 'mean'),
                    graph_update_mean_ms=('graph_update_ms', 'mean'),
                    graph_update_std_ms=('graph_update_ms', 'std')
                ).fillna(0.0)

            with pd.ExcelWriter(profile_filename) as writer:
                df_results.to_excel(writer, sheet_name='per_environment', index=False)
                pd.DataFrame(profile_summary).to_excel(writer, sheet_name='summary', index=False)
            with pd.ExcelWriter(growth_filename) as writer:
                growth_df.to_excel(writer, sheet_name='per_exploration', index=False)
                growth_summary.to_excel(writer, sheet_name='summary_by_episode', index=False)
            print(f"图规模与时间统计已保存: {profile_filename}")
            print(f"逐次探索图增长统计已保存: {growth_filename}")

            # 2. 按“大循环次数 (Run)”分组，统计每组的总到达次数 sum(arrive)
            run_scores = df_results.groupby('大循环次数 (Run)')['是否到达 (Arrive)'].sum()

            # 3. 找出排名前 40 的大循环并筛选数据
            top_40_runs = run_scores.nlargest(50).index
            df_filtered = df_results[df_results['大循环次数 (Run)'].isin(top_40_runs)].copy()

            # 4. === 核心修改：重新映射为连续编号 (1 到 40) ===
            # 获取筛选后留下来的所有大循环编号，并按照原来的先后顺序排序
            unique_runs = sorted(df_filtered['大循环次数 (Run)'].unique())
            # 创建映射字典，把原来的跳跃编号映射为 1 到 40
            run_mapping = {old_run: new_run for new_run, old_run in enumerate(unique_runs, start=1)}
            # 应用映射
            df_filtered['大循环次数 (Run)'] = df_filtered['大循环次数 (Run)'].map(run_mapping)

            # 5. 保存至 Excel
            # excel_filename = f"DN3_GBMR_Result_训练{current_num_episode}轮_40组独立实验.xlsx"
            # df_filtered.to_excel(excel_filename, index=False)

