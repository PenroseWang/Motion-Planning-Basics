import os
import copy
from collections import deque
import numpy as np
from utils import tic, collision_free, in_boundary, heuristic, dist, dist_sq, Node, draw_map, plt, save_gif


class RobotPlannerRTAA:
    # search-based RTAA* algorithm
    def __init__(self, boundary, blocks, res=0.1):
        self.boundary = boundary
        self.blocks = blocks
        self.res = res                  # grid resolution
        self.h_table = {}               # table to save updated heuristics
        self.connectivity_table = {}    # table to save connectivity with nodes
        self.start = None
        self.goal = None
        self.timer = None
        self.numofdirs = None
        self.dR = None
        self.to_be_continued = False
        self.continue_plan_data = None

    def plan(self, start, goal):
        self.timer = tic()
        self.start = start
        self.goal = goal

        if self.to_be_continued:
            return self.continue_planning(), tic() - self.timer

        if not self.numofdirs:
            self.numofdirs = 26
            [dX, dY, dZ] = np.meshgrid([-self.res, 0, self.res], [-self.res, 0, self.res],
                                       [-self.res, 0, self.res])
            self.dR = np.vstack((dX.flatten(), dY.flatten(), dZ.flatten()))
            self.dR = np.delete(self.dR, 13, axis=1)

        # 1. expand N nodes using A* algorithm (adaptive N depending on time limit)
        arrived = False
        node_i = None
        opened, closed = {}, {}
        opened[tuple(start)] = Node(start, g=0, h=self.h(start))
        while tic() - self.timer < 1:
            # remove node_i with smallest f_i and put into closed
            f = np.inf
            for i_rp in opened:
                f_i = opened[i_rp].g + opened[i_rp].h
                if f_i < f:
                    f = f_i
                    node_i = opened[i_rp]
            node_i_rp = tuple(node_i.pos)
            opened.pop(node_i_rp)
            if node_i.h == 0:
                arrived = True
                break
            closed[node_i_rp] = node_i
            self.connect_children(node_i)

            for newrp in self.connectivity_table[node_i_rp]:
                if newrp in closed:
                    continue
                if newrp not in opened:
                    opened[newrp] = Node(np.array(newrp), node_i_rp, node_i.g + dist(newrp, node_i_rp),
                                         self.h(newrp))
                else:
                    node_newrp_j = node_i.g + dist(newrp, node_i_rp)
                    if opened[newrp].g > node_newrp_j:
                        opened[newrp].g = node_newrp_j
                        opened[newrp].parent = node_i_rp

        # 2. update heuristics in closed list
        if arrived:
            f_jstar = node_i.g + node_i.h
            node_j = node_i
        else:
            f_jstar = np.inf
            node_j = None
            for j_rp in opened:
                if opened[j_rp].g + opened[j_rp].h < f_jstar:
                    f_jstar = opened[j_rp].g + opened[j_rp].h
                    node_j = opened[j_rp]
        for i_rp in closed:
            self.h_table[i_rp] = f_jstar - closed[i_rp].g

        # 3. find optimal path using A*
        closed_pre = closed
        closed_pre[tuple(node_j.pos)] = node_j
        self.connect_children(node_j)
        opened, closed = {}, {}
        opened[tuple(start)] = Node(start, g=0, h=self.h(start))
        while True:
            # exit and continue planning on next move if time exceeds limit
            if tic() - self.timer > 1.5:
                self.to_be_continued = True
                self.continue_plan_data = [copy.deepcopy(node_j), copy.deepcopy(opened),
                                           copy.deepcopy(closed), copy.deepcopy(closed_pre)]
                return start, tic() - self.timer

            # remove node_i with smallest f_i and put into closed
            node_i = None
            f = np.inf
            for i_rp in opened:
                f_i = opened[i_rp].g + opened[i_rp].h
                if f_i < f:
                    f = f_i
                    node_i = opened[i_rp]
            node_i_rp = tuple(node_i.pos)
            opened.pop(node_i_rp)
            closed[node_i_rp] = node_i

            if tuple(node_j.pos) in closed:
                break

            for newrp in self.connectivity_table[node_i_rp]:
                if newrp not in closed_pre or newrp in closed:
                    continue
                if newrp not in opened:
                    opened[newrp] = closed_pre[newrp]
                    opened[newrp].h = self.h(newrp)
                else:
                    node_newrp_j = node_i.g + dist(newrp, node_i_rp)
                    if opened[newrp].g > node_newrp_j:
                        opened[newrp].g = node_newrp_j
                        opened[newrp].parent = node_i_rp
        cur_node = closed[tuple(node_j.pos)]
        while cur_node.parent != tuple(start):
            cur_node = closed_pre[cur_node.parent]
        return cur_node.pos, tic() - self.timer

    def h(self, position):
        rp = tuple(position)
        if rp not in self.h_table:
            self.h_table[rp] = heuristic(position, self.goal)
        return self.h_table[rp]

    def connect_children(self, node):
        node_rp = tuple(node.pos)
        if node_rp not in self.connectivity_table:
            self.connectivity_table[node_rp] = set()
            for j in range(self.numofdirs):
                newrp = tuple(node.pos + self.dR[:, j])
                if not collision_free(node_rp, newrp, self.blocks) or \
                        not in_boundary(newrp, self.boundary):
                    continue
                self.connectivity_table[node_rp].add(newrp)
        return

    def continue_planning(self):
        self.timer = tic()
        node_j, opened, closed, closed_pre = self.continue_plan_data
        while True:
            if tic() - self.timer > 1.5:
                self.continue_plan_data = [copy.deepcopy(node_j), copy.deepcopy(opened),
                                           copy.deepcopy(closed), copy.deepcopy(closed_pre)]
                return self.start

            # remove node_i with smallest f_i and put into closed
            node_i = None
            f = np.inf
            for i_rp in opened:
                f_i = opened[i_rp].g + opened[i_rp].h
                if f_i < f:
                    f = f_i
                    node_i = opened[i_rp]
            node_i_rp = tuple(node_i.pos)
            opened.pop(node_i_rp)
            closed[node_i_rp] = node_i

            if tuple(node_j.pos) in closed:
                break

            for newrp in self.connectivity_table[node_i_rp]:
                if newrp not in closed_pre or newrp in closed:
                    continue
                if newrp not in opened:
                    opened[newrp] = closed_pre[newrp]
                    opened[newrp].h = self.h(newrp)
                else:
                    node_newrp_j = node_i.g + dist(newrp, node_i_rp)
                    if opened[newrp].g > node_newrp_j:
                        opened[newrp].g = node_newrp_j
                        opened[newrp].parent = node_i_rp
        cur_node = closed[tuple(node_j.pos)]
        while cur_node.parent != tuple(self.start):
            cur_node = closed_pre[cur_node.parent]
        self.to_be_continued = False
        return cur_node.pos


class RobotPlannerRRT:
    # sampling-based RRT* algorithm
    def __init__(self, boundary, blocks, map_name, display=True, map_data=None):
        self.boundary = boundary
        self.blocks = blocks
        self.map_name = map_name
        self.tree = dict()
        self.edges = set()
        self.vol_free = -1.0
        self.V = -1
        self.r = -1.0
        self.epsilon = 0.5
        self.timer = -1.0
        self.root = None
        self.goal = None
        self.map_completed = False
        self.trajectory = deque([])
        self.trajectory_smoothed = deque([])
        self.trajectory_completed = False
        self.plan_completed = False
        # display properties
        self.display = display
        if self.display:
            self.fig, self.ax = map_data
        else:
            self.fig, self.ax = None, None
        self.displayed_nodes = set()

    def plan(self, start, goal):
        self.timer = tic()
        # if self.trajectory_completed:       # return original trajectory
        #     return self.trajectory.popleft(), tic() - self.timer
        if self.plan_completed:             # return smoothed trajectory
            return self.trajectory_smoothed.popleft(), tic() - self.timer
        if len(self.tree) == 0:
            # initialization
            self.root = tuple(start)
            self.goal = tuple(goal)
            self.tree[self.root] = [0, None]  # [cost, parent]
            self.vol_free = (self.boundary[0, 3] - self.boundary[0, 0]) * \
                            (self.boundary[0, 4] - self.boundary[0, 1]) * \
                            (self.boundary[0, 5] - self.boundary[0, 2])
            for k in range(self.blocks.shape[0]):
                self.vol_free -= (self.blocks[k, 3] - self.blocks[k, 0]) * \
                                 (self.blocks[k, 4] - self.blocks[k, 1]) * \
                                 (self.blocks[k, 5] - self.blocks[k, 2])
            self.V = int(self.vol_free)
            self.r = 1.1 * 2 * ((1 + 1/3)*(self.vol_free*3/(4*np.pi))*np.log(self.V)/self.V)**(1/3)
        if not self.map_completed:
            self.construct_tree()
            move_time = tic() - self.timer
            if self.display:
                self.display_tree()
                if self.map_completed:
                    plt.savefig(os.path.join('results', self.map_name + '_tree'))
            else:
                if self.map_completed:
                    for node in self.tree.keys():
                        self.ax.plot(node[0:1], node[1:2], node[2:], 'go', markersize=1)
                    plt.savefig(os.path.join('results', self.map_name + '_tree'))
            return start, move_time
        if not self.trajectory_completed:
            # planning
            cur_node = self.goal
            while cur_node:
                self.trajectory.appendleft(np.array(cur_node))
                cur_node = self.tree[cur_node][1]
                if tic() - self.timer > 1.5:    # timer, exit if exceeds 1.5 s
                    self.goal = cur_node
                    return start, tic() - self.timer
            self.trajectory_completed = True
            return start, tic() - self.timer
        if not self.plan_completed:
            # smooth trajectory
            while len(self.trajectory) > 1:
                if len(self.trajectory_smoothed) == 0:
                    self.trajectory_smoothed.append(self.trajectory.popleft())
                if collision_free(self.trajectory_smoothed[-1], self.trajectory[1], self.blocks):
                    self.trajectory.popleft()
                else:
                    next_move = self.trajectory[0]
                    if dist_sq(next_move, self.trajectory_smoothed[-1]) > 1:
                        self.trajectory_smoothed.append(self.trajectory_smoothed[-1] + .8 *
                            (next_move - self.trajectory_smoothed[-1]) / dist(next_move, self.trajectory_smoothed[-1]))
                    else:
                        self.trajectory_smoothed.append(self.trajectory.popleft())
                if tic() - self.timer > 1.5:
                    return start, tic() - self.timer

            goal = self.trajectory.popleft()
            while dist_sq(goal, self.trajectory_smoothed[-1]) > 1:
                self.trajectory_smoothed.append(self.trajectory_smoothed[-1] + .8 *
                        (goal - self.trajectory_smoothed[-1]) / dist(goal, self.trajectory_smoothed[-1]))
            self.trajectory_smoothed.append(goal)
            self.plan_completed = True
            return start, tic() - self.timer

    def construct_tree(self):
        while tic() - self.timer < 1.5:     # timer, exit if exceeds 1.5 s
            x_rand = self.sample_free()
            x_nearest, edge_nearest = self.nearest(x_rand)
            x_new = self.steer(x_nearest, x_rand, edge_nearest)
            if x_new is None:
                continue
            X_near = self.near(x_new, min(self.r, self.epsilon))
            self.tree[x_new] = [self.tree[x_nearest][0] + dist(x_nearest, x_new), None]
            # extend along a minimum-cost path
            x_min = x_nearest
            for x_near in X_near:
                if collision_free(x_near, x_new, self.blocks):
                    c = self.tree[x_near][0] + dist(x_near, x_new)
                    if c < self.tree[x_new][0]:
                        self.tree[x_new][0] = c
                        x_min = x_near
            self.add_edge(x_min, x_new)
            self.tree[x_new][1] = x_min
            # rewire the tree
            for x_near in X_near:
                if collision_free(x_near, x_new, self.blocks) and \
                        self.tree[x_new][0] + dist(x_new, x_near) < self.tree[x_near][0]:
                    self.delete_edge(x_near, self.tree[x_near][1])
                    self.tree[x_near][1] = x_new
                    self.add_edge(x_new, x_near)
            # check if goal in tree:
            if self.goal in self.tree:
                self.map_completed = True
                return

    def sample_free(self):
        # goal-biased sampling
        if np.random.rand(1) < 0.01:
            return self.goal
        sample = np.empty(3)
        success = False
        while not success:
            sample[0] = self.boundary[0, 0] + (self.boundary[0, 3] - self.boundary[0, 0])*np.random.rand(1)
            sample[1] = self.boundary[0, 1] + (self.boundary[0, 4] - self.boundary[0, 1])*np.random.rand(1)
            sample[2] = self.boundary[0, 2] + (self.boundary[0, 5] - self.boundary[0, 2])*np.random.rand(1)
            # check if in free space
            for k in range(self.blocks.shape[0]):
                if (self.blocks[k, 0] < sample[0] < self.blocks[k, 3] and
                        self.blocks[k, 1] < sample[1] < self.blocks[k, 4] and
                        self.blocks[k, 2] < sample[2] < self.blocks[k, 5]):
                    break
            success = True
        return tuple(sample)

    def nearest(self, node):
        d_min = np.inf
        d_min_free = np.inf
        edge = None
        x_nearest = self.root
        edge_free = None
        x_nearest_free = self.root
        for (node1, node2) in self.edges:
            px, py, pz = node[0], node[1], node[2]
            x1, y1, z1 = node1[0], node1[1], node1[2]
            x2, y2, z2 = node2[0], node2[1], node2[2]
            t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1) + (pz - z1) * (z2 - z1)) / dist_sq(node1, node2)
            t = max(0, min(1, t))
            node_x = tuple(np.array(node1) + t * (np.array(node2) - np.array(node1)))
            d = dist_sq(node_x, node)
            if d < d_min:
                x_nearest = node_x
                d_min = d
                edge = (node1, node2)
            if collision_free(node_x, node, self.blocks) and d < d_min_free:
                x_nearest_free = node_x
                d_min_free = d
                edge_free = (node1, node2)
        if edge_free:
            return x_nearest_free, edge_free
        else:
            return x_nearest, edge

    def steer(self, node_start, node_goal, edge_start):
        if node_start == node_goal:
            return None
        d_sq = dist_sq(node_start, node_goal)
        if node_goal == self.goal and d_sq < self.epsilon ** 2:
            if not collision_free(node_start, self.goal, self.blocks):
                return None
            # construct node and edge if node_start not in tree
            if node_start not in self.tree:
                self.add_node(node_start, edge_start)
            return tuple(self.goal)
        else:
            x_new = np.array(node_start) + \
                    (np.array(node_goal) - np.array(node_start))/np.sqrt(d_sq) * self.epsilon
            if not collision_free(node_start, x_new, self.blocks) or not in_boundary(x_new, self.boundary):
                return None
            # construct node and edge if node_start not in tree
            if node_start not in self.tree:
                self.add_node(node_start, edge_start)
            return tuple(x_new)

    # def steer(self, node_start, node_goal, edge_start):
    #     if node_start == node_goal:
    #         return None
    #     d_sq = dist_sq(node_start, node_goal)
    #     if node_goal == self.goal and d_sq < self.epsilon ** 2:
    #         if not collision_free(node_start, self.goal, self.blocks):
    #             return None
    #         # construct node and edge if node_start not in tree
    #         if node_start not in self.tree:
    #             self.add_node(node_start, edge_start)
    #         return tuple(self.goal)
    #     else:
    #         x_new = np.array(node_start) + \
    #                 (np.array(node_goal) - np.array(node_start))/np.sqrt(d_sq) * self.epsilon
    #         if not collision_free(node_start, x_new, self.blocks) or not in_boundary(x_new, self.boundary):
    #             return None
    #         for node in self.tree:
    #             if dist_sq(node, x_new) < self.epsilon ** 2:
    #                 return None
    #         # construct node and edge if node_start not in tree
    #         if node_start not in self.tree:
    #             self.add_node(node_start, edge_start)
    #         return tuple(x_new)

    def near(self, node1, r):
        X_near = set()
        for node in self.tree:
            if dist_sq(node1, node) <= r ** 2:
                X_near.add(node)
        return X_near

    def add_node(self, node, edge):
        self.delete_edge(edge[0], edge[1])

        if self.tree[edge[1]][1] == edge[0]:
            self.tree[node] = [self.tree[edge[0]][0] + dist(edge[0], node), edge[0]]
            self.tree[edge[1]][1] = node
        else:
            self.tree[node] = [self.tree[edge[1]][0] + dist(edge[1], node), edge[1]]
            self.tree[edge[0]][1] = node
        self.add_edge(edge[0], node)
        self.add_edge(edge[1], node)

    def add_edge(self, node1, node2):
        self.edges.add((node1, node2))
        return

    def delete_edge(self, node1, node2):
        self.edges.discard((node1, node2))
        self.edges.discard((node2, node1))
        return

    def display_tree(self):
        for node in self.tree.keys():
            if node in self.displayed_nodes:
                continue
            self.ax.plot(node[0:1], node[1:2], node[2:], 'go', markersize=1)
            self.displayed_nodes.add(node)
        self.fig.canvas.flush_events()


class RobotPlannerGreedy:
    __slots__ = ['boundary', 'blocks']

    def __init__(self, boundary, blocks):
        self.boundary = boundary
        self.blocks = blocks

    def plan(self, start, goal):
        # for now greedily move towards the goal
        newrobotpos = np.copy(start)

        numofdirs = 26
        [dX, dY, dZ] = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])
        dR = np.vstack((dX.flatten(), dY.flatten(), dZ.flatten()))
        dR = np.delete(dR, 13, axis=1)
        dR = dR / np.sqrt(np.sum(dR ** 2, axis=0)) / 2.0

        mindisttogoal = 1000000
        for k in range(numofdirs):
            newrp = start + dR[:, k]

            # Check if this direction is valid
            if (newrp[0] < self.boundary[0, 0] or newrp[0] > self.boundary[0, 3] or
                    newrp[1] < self.boundary[0, 1] or newrp[1] > self.boundary[0, 4] or
                    newrp[2] < self.boundary[0, 2] or newrp[2] > self.boundary[0, 5]):
                continue

            valid = True
            for k in range(self.blocks.shape[0]):
                if (self.blocks[k, 0] < newrp[0] < self.blocks[k, 3] and
                        self.blocks[k, 1] < newrp[1] < self.blocks[k, 4] and
                        self.blocks[k, 2] < newrp[2] < self.blocks[k, 5]):
                    valid = False
                    break
            if not valid:
                break

            # Update newrobotpos
            disttogoal = sum((newrp - goal) ** 2)
            if (disttogoal < mindisttogoal):
                mindisttogoal = disttogoal
                newrobotpos = newrp

        return newrobotpos


