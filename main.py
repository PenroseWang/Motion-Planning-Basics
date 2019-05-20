import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from RobotPlanner import RobotPlannerRTAA, RobotPlannerGreedy, RobotPlannerRRT
from utils import load_map, draw_map, tic, check, save_gif, print_results

matplotlib.use("TkAgg")
plt.ion()


def runtest(mapfile, start, goal, verbose=True):
    # directory to save results
    result_dir = 'results'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    map_name = mapfile.split('.')[-2].split('/')[-1]
    plots_save_path = os.path.join(result_dir, map_name)
    if os.path.exists(plots_save_path):
        for file_name in os.listdir(plots_save_path):
            os.remove(os.path.join(plots_save_path, file_name))
        os.rmdir(plots_save_path)
    os.mkdir(plots_save_path)

    # display the environment
    boundary, blocks = load_map(mapfile)
    if verbose:
        fig, ax, hb, hs, hg = draw_map(boundary, blocks, start, goal, 'map')
        # mng = plt.get_current_fig_manager()
        # mng.resize(*mng.window.maxsize())
        plt.savefig(os.path.join(plots_save_path, 'result%04d.png' % 0))

    # instantiate a robot planner
    # RP = RobotPlannerGreedy(boundary, blocks)
    # RP = RobotPlannerRTAA(boundary, blocks)
    RP = RobotPlannerRRT(boundary, blocks, map_name, display=True, map_data=[fig, ax])

    # main loop
    robotpos = np.copy(start)
    numofmoves = 0
    max_time = 0
    total_time = 0
    trajectory = np.copy(start)
    while True:

        # call the robot planner
        newrobotpos, move_time = RP.plan(robotpos, goal)
        total_time += move_time
        max_time = max(move_time, max_time)
        movetime = max(1, np.ceil(move_time/2.0))

        # check if the planner was done on time
        if movetime > 1:
            newrobotpos = robotpos - 0.5 + np.random.rand(3)
            print('\nWarning: move time exceeds limit.')

        # check if the commanded position is valid
        if not check(robotpos, newrobotpos, blocks, boundary):
            success = False
            break

        # make the move
        robotpos = newrobotpos
        numofmoves += 1
        trajectory = np.vstack((trajectory, robotpos))
        print('\rNum of moves: %04d, move time: %.2f s, max time: %.2f s' % (numofmoves, move_time, max_time),
              end='', flush=True)

        # update plot
        if verbose:
            hs[0].set_xdata(robotpos[0])
            hs[0].set_ydata(robotpos[1])
            hs[0].set_3d_properties(robotpos[2])
            ax.plot(trajectory[-2:, 0], trajectory[-2:, 1], trajectory[-2:, 2], 'r')
            fig.canvas.flush_events()
            plt.savefig(os.path.join(plots_save_path, 'result%05d.png' % numofmoves))

        # check if the goal is reached
        if sum((robotpos - goal) ** 2) <= 0.1:
            success = True
            break

        # exit if number of moves exceeds limit
        if numofmoves >= 9999:
            success = False
            break

    # calculate distance
    distance = np.sum(np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=1))

    # save video & plot
    if verbose:
        print('\rSaving results...', end='', flush=True)
        result_name = os.path.join(result_dir, map_name)
        # save plot
        hs[0].set_xdata(start[0])
        hs[0].set_ydata(start[1])
        hs[0].set_3d_properties(start[2])
        ax.plot(trajectory[-2:, 0], trajectory[-2:, 1], trajectory[-2:, 2], 'r')
        fig.canvas.flush_events()
        plt.savefig(result_name + '.png')
        save_gif(plots_save_path, result_name + '.gif')
        print('\rResults have been saved to \'' + result_name + '.gif|png\'.', end='', flush=True)
        plt.close(fig)

    return success, numofmoves, distance, total_time, max_time


def test_single_cube():
    print('SINGLE_CUBE TEST:')
    start = np.array([2.3, 2.3, 1.3])
    goal = np.array([7.0, 7.0, 6.0])
    success, numofmoves, distance, total_time, max_time = runtest('./maps/single_cube.txt', start, goal, True)
    print_results(success, numofmoves, distance, total_time, max_time, './results/single_cube.txt')


def test_maze():
    print('MAZE TEST:')
    start = np.array([0.0, 0.0, 1.0])
    goal = np.array([12.0, 12.0, 5.0])
    success, numofmoves, distance, total_time, max_time = runtest('./maps/maze.txt', start, goal, True)
    print_results(success, numofmoves, distance, total_time, max_time, './results/maze.txt')


def test_window():
    print('WINDOW TEST:')
    start = np.array([0.2, -4.9, 0.2])
    goal = np.array([6.0, 18.0, 3.0])
    success, numofmoves, distance, total_time, max_time = runtest('./maps/window.txt', start, goal, True)
    print_results(success, numofmoves, distance, total_time, max_time, './results/window.txt')


def test_tower():
    print('TOWER TEST:')
    start = np.array([2.5, 4.0, 0.5])
    goal = np.array([4.0, 2.5, 19.5])
    success, numofmoves, distance, total_time, max_time = runtest('./maps/tower.txt', start, goal, True)
    print_results(success, numofmoves, distance, total_time, max_time, './results/tower.txt')


def test_flappy_bird():
    print('FLAPPY_BIRD TEST:')
    start = np.array([0.5, 2.5, 5.5])
    goal = np.array([19.0, 2.5, 5.5])
    success, numofmoves, distance, total_time, max_time = runtest('./maps/flappy_bird.txt', start, goal, True)
    print_results(success, numofmoves, distance, total_time, max_time, './results/flappy_bird.txt')


def test_room():
    print('ROOM TEST:')
    start = np.array([1.0, 5.0, 1.5])
    goal = np.array([9.0, 7.0, 1.5])
    success, numofmoves, distance, total_time, max_time = runtest('./maps/room.txt', start, goal, True)
    print_results(success, numofmoves, distance, total_time, max_time, './results/room.txt')


def test_monza():
    print('MONZA TEST:')
    start = np.array([0.5, 1.0, 4.9])
    goal = np.array([3.8, 1.0, 0.1])
    success, numofmoves, distance, total_time, max_time = runtest('./maps/monza.txt', start, goal, True)
    print_results(success, numofmoves, distance, total_time, max_time, './results/monza.txt')


if __name__ == "__main__":
    test_single_cube()
    test_flappy_bird()
    test_window()
    test_tower()
    test_room()
    test_monza()
    test_maze()

