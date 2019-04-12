import os
import time

import math
import numpy as np
import scipy.io as sio
import scipy.sparse as ss

root = os.getcwd()
dataset_path = 'suite_matrix_dataset'


def load_data():
    blank = ''
    print(f'{blank:20}')

    for dirpath, dirnames, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.mtx'):
                abspath = os.path.normpath(os.path.join(root, dirpath, file))
                mm, bf, t, ob = format_comparison(read_matrix_market(abspath))

                # print to the screen
                shape = mm.get_shape()
                print(f'{file:20} {shape[0]:>10}:{shape[1]:<10} {t:<10.2} {bf:10}', end='')
                for (k, v) in ob.items():
                    print(f'{v:10.2}', end='')
                print()


def read_matrix_market(file=None):
    if not file:
        file = '/media/toto/WORKSPACE/CSI5610/' \
               'SpMv/suite_matrix_dataset/mm/rgg010/rgg010.mtx'
    mm = sio.mmread(file)
    if type(mm) is np.ndarray:
        mm = ss.coo_matrix(mm)
    return mm


def format_comparison(m: ss.spmatrix):
    # make the operand to be the unit vector in associate shape
    multiplier = np.ones((m.get_shape()[1], 1))

    observation = dict()

    # ordinary dot product in COOrdinate format
    p, f, t = measure_multiplication(m.copy().tocoo(), multiplier)
    observation[str(f)] = t

    # dot product in Compressed Sparse Row format
    p, f, t = measure_multiplication(m.copy().tocsr(), multiplier)
    observation[str(f)] = t

    # dot product in Compressed Sparse Column format
    p, f, t = measure_multiplication(m.copy().tocsc(), multiplier)
    observation[str(f)] = t

    # dot product in sparse DIAgonal format
    try:
        mf = m.copy().todia()
        p, f, t = measure_multiplication(mf, multiplier)
    except RuntimeError:
        # print(RuntimeError)
        f = 'dia'
        t = float(math.inf)
    observation[str(f)] = t

    # dot product in Block Sparse Row format
    p, f, t = measure_multiplication(m.copy().tobsr(), multiplier)
    observation[str(f)] = t

    # dot product in Dictionary Of Key format
    p, f, t = measure_multiplication(m.copy().todok(), multiplier)
    observation[str(f)] = t

    # dot product in LInked List format
    p, f, t = measure_multiplication(m.copy().tolil(), multiplier)
    observation[str(f)] = t

    # dot product in full dense matrix
    # p, f, t = measure_multiplication(m.copy().todense(), multiplier)
    # observation[str(f)] = t

    # best format and the elapsed time
    k = 'coo'
    v = observation['coo']

    # one loop to find the best format
    for (key, value) in observation.items():
        if value < v:
            k = key
            v = value
        # print the result
        # print(f'{v:10.2}', end='')
    print()

    return m, k, v, observation


def measure_multiplication(m: ss.spmatrix, operand: object) -> object:
    start = time.time()
    product = m.dot(operand)
    end = time.time()

    return product, m.getformat(), (end - start) * 1000


if __name__ == '__main__':
    load_data()
