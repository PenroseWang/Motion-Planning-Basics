import os
import sys
import time
import imageio
import numpy as np
import numpy.lib.recfunctions as nprec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def tic():
    return time.time()


def toc(tstart, nm=""):
    print('%s took: %s sec.\n' % (nm, (time.time() - tstart)))


def load_map(fname):
    mapdata = np.loadtxt(fname, dtype={'names': ('type', 'xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax', 'r', 'g', 'b'),
                                       'formats': ('S8', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f')})
    blockIdx = mapdata['type'] == b'block'
    # works on numpy-1.16.3
    boundary = nprec.structured_to_unstructured(mapdata[~blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax',
                                                                    'r', 'g', 'b']])
    blocks = nprec.structured_to_unstructured(mapdata[blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax', 'r',
                                                                 'g', 'b']])
    return boundary, blocks


def draw_map(boundary, blocks, start, goal, figname=''):
    fig = plt.figure(figname)
    ax = fig.add_subplot(111, projection='3d')
    hb = draw_block_list(ax, blocks)
    ax.plot(start[0:1], start[1:2], start[2:], 'ro', markersize=7, markeredgecolor='k', alpha=0.3)
    hs = ax.plot(start[0:1], start[1:2], start[2:], 'ro', markersize=7, markeredgecolor='k')
    hg = ax.plot(goal[0:1], goal[1:2], goal[2:], 'go', markersize=7, markeredgecolor='k')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(boundary[0, 0], boundary[0, 3])
    ax.set_ylim(boundary[0, 1], boundary[0, 4])
    ax.set_zlim(boundary[0, 2], boundary[0, 5])
    return fig, ax, hb, hs, hg


def draw_block_list(ax, blocks):
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
                 dtype='float')
    f = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [0, 1, 2, 3], [4, 5, 6, 7]])
    clr = blocks[:, 6:] / 255
    n = blocks.shape[0]
    d = blocks[:, 3:6] - blocks[:, :3]
    vl = np.zeros((8 * n, 3))
    fl = np.zeros((6 * n, 4), dtype='int64')
    fcl = np.zeros((6 * n, 3))
    for k in range(n):
        vl[k * 8:(k + 1) * 8, :] = v * d[k] + blocks[k, :3]
        fl[k * 6:(k + 1) * 6, :] = f + k * 8
        fcl[k * 6:(k + 1) * 6, :] = clr[k, :]

    if type(ax) is Poly3DCollection:
        ax.set_verts(vl[fl])
    else:
        pc = Poly3DCollection(vl[fl], alpha=0.25, linewidths=1, edgecolors='k')
        pc.set_facecolor(fcl)
        h = ax.add_collection3d(pc)
        return h


def save_gif(plots_path, save_path):
    i = 0
    images = [[]] * 99999  # at most 99,999 frames
    for file_name in sorted(os.listdir(plots_path)):
        if file_name.endswith('.png'):
            file_path = os.path.join(plots_path, file_name)
            images[i] = imageio.imread(file_path)
            os.remove(os.path.join(plots_path, file_name))
            i += 1
    os.rmdir(plots_path)
    imageio.mimsave(save_path, images[:i])


def print_results(success, numofmoves, distance, total_time, max_time, txt_name):
    print('\rSuccess: %r' % success)
    print('Number of Moves: %i' % numofmoves)
    print('Total time: %.2f s.' % total_time)
    print('Average time per move: %.2f s/move.' % (total_time/numofmoves))
    print('Max time per move: %.2f s.' % max_time)
    print('Travel distance: %.2f' % distance)
    with open(txt_name, 'w') as f:
        f.write(txt_name.split('.')[-2].split('/')[-1].upper() + ' TEST:')
        f.write('\nSuccess: %r' % success)
        f.write('\nNumber of Moves: %i' % numofmoves)
        f.write('\nTotal time: %.2f s.' % total_time)
        f.write('\nAverage time per move: %.2f s/move.' % (total_time / numofmoves))
        f.write('\nMax time per move: %.2f s.' % max_time)
        f.write('\nTravel distance: %.2f' % distance)
        f.close()


def check(robotpos, newrobotpos, blocks, boundary):
    success = True
    if sum((newrobotpos - robotpos) ** 2) > 1:
        print('ERROR: the robot cannot move too fast.')
        return not success
    if not in_boundary(newrobotpos, boundary):
        print('ERROR: out-of-map robot position commanded.')
        return not success
    if not collision_free(robotpos, newrobotpos, blocks):
        print('ERROR: collision... BOOM, BAAM, BLAAM!!!')
        return not success
    return success


def in_boundary(newrobotpos, boundary):
    if (newrobotpos[0] < boundary[0, 0] or newrobotpos[0] > boundary[0, 3] or
            newrobotpos[1] < boundary[0, 1] or newrobotpos[1] > boundary[0, 4] or
            newrobotpos[2] < boundary[0, 2] or newrobotpos[2] > boundary[0, 5]):
        return False
    return True


def collision_free(start, end, blocks):
    # reference: http://www.garagegames.com/community/blogs/view/309
    start, end = np.array(start), np.array(end)
    for block in blocks:
        success = False
        bmin, bmax = (block[0], block[1], block[2]), (block[3], block[4], block[5])
        fst, fet = 0, 1
        for i in range(3):
            if start[i] < end[i]:
                if start[i] > bmax[i] or end[i] < bmin[i]:
                    success = True
                    break
                d = end[i] - start[i]
                st = (bmin[i] - start[i])/d if start[i] < bmin[i] else 0
                et = (bmax[i] - start[i])/d if end[i] > bmax[i] else 1
            else:
                if end[i] > bmax[i] or start[i] < bmin[i]:
                    success = True
                    break
                d = end[i] - start[i]
                st = (bmax[i] - start[i])/d if start[i] > bmax[i] else 0
                et = (bmin[i] - start[i])/d if end[i] < bmin[i] else 1
            fst = max(fst, st)
            fet = min(fet, et)
            if fet < fst:
                success = True
                break
        if not success:
            return False
    return True


def heuristic(point1, point2):
    p1, p2 = np.array(point1), np.array(point2)
    # d = dist(p1, p2)                # Euclidean distance
    d = np.max(np.abs(p1 - p2))     # diagonal distance
    # d = np.sum(np.abs(p1 - p2))     # Manhattan distance
    return d if d > 0.1 else 0


def dist(point1, point2):
    return np.sqrt(dist_sq(point1, point2))


def dist_sq(point1, point2):
    return np.sum(np.square(np.array(point1) - np.array(point2)))


class Node:
    def __init__(self, position, parent=None, g=np.inf, h=0.0):
        self.pos = position
        self.parent = parent
        self.g = g
        self.h = h

if __name__ == '__main__':
    print(collision_free((10, 10, 0), (0, 10, 0), [[0, 1, 0, 5, 5, 5]]))



