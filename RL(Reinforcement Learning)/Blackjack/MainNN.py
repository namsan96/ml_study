from casino import *
import numpy as np
from tempfile import *
import time
from NNPlayer import *
import matplotlib.pyplot as plt

def trainBJ(n_games, tlimit, fn):
    stime = time.time()
    plt.ion()
    dealer = CasinoBJ()
    player = PlayerMLP(dealer)
    # try:  # try to open the file fn and continue from it
    #     player.load(fn)
    # except FileNotFoundError:
    #     print ("%s does not exist.  Start from scratch.")

    player.run_simulation(n_games, tlimit)
    player.save(fn)
    etime = time.time()
    print ("Training time = ", etime - stime)
    plt.ioff()
    return player

def testBJ(n_games, fn):
    stime = time.time()
    dealer = CasinoBJ()
    player = PlayerMLP(dealer)
    player.load(fn)
    plt.ioff()
    player.plot_Q()
    player.test_performance(n_games)
    etime = time.time()
    print ("Test time = ", etime - stime)

# main program
if __name__ == '__main__':
    n_train = 5*10**4
    n_test = 3*10**4
    tlimit = 6000000000
    fname = 'Try'
    trainBJ(n_train,tlimit, fname)
    testBJ(n_test, fname)
