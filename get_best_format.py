import sys
import os
import time
import datetime

import math
import numpy as np
import scipy.io as sio
import scipy.sparse as ss

import csv

matrix_formats = ['coo', 'csr', 'csc', 'dia', 'bsr', 'dok', 'lil']
DENSE = 'dense'


def load_data(dataset=None, result=None):
    if not dataset:
        root_path = os.getcwd()
        dataset_folder = 'test/dataset'
        dataset = os.path.join(root_path, dataset_folder)

    if not result:
        root_path = os.getcwd()
        result_folder = 'test/result'
        result = os.path.join(root_path, result_folder)

    date_string = datetime.datetime.now().isoformat()
    output_file_name = 'best_formats_' + date_string + '.csv'
    output_path = os.path.join(result, output_file_name)

    with open(output_path, 'w', newline='') as csvfile:
        fn = ['FILE', 'N_ROWS', 'N_COLUMNS', 'SHORTEST_ELAPSED_TIME',
              'BEST_FORMAT'] + matrix_formats
        writer = csv.DictWriter(csvfile, dialect=csv.unix_dialect,
                                fieldnames=fn, delimiter=',')

        # table head
        writer.writeheader()

        # print to screen
        print('{:30}'.format('FILE'), end='')
        print('{:^21} {:^10} {:^10}'
              .format('SHAPE', 'BEST TIME', 'BEST FORMAT'), end='')
        for f in matrix_formats:
            print(f'{f.upper():>10}', end='')
        print()

        for dirpath, dirnames, files in os.walk(dataset):
            for file in files:
                if file.endswith('.mtx'):
                    abspath = os.path.join(dirpath, file)
                    mm, p, bf, bt, ob = format_comparison(
                        read_matrix_market(abspath))

                    shape = mm.get_shape()

                    # write to csv file
                    buffer = {
                        'FILE': f'{file:<30}',
                        'N_ROWS': f'{shape[0]:>10d}',
                        'N_COLUMNS': f'{shape[1]:>10d}',
                        'SHORTEST_ELAPSED_TIME': f'{bt:>10.4f}',
                        'BEST_FORMAT': f'{bf:>10s}',
                    }
                    formatted_observation = ob.copy()
                    for (k, v) in formatted_observation.items():
                        formatted_observation[k] = f'{v:>10.4f}'
                    buffer.update(formatted_observation)
                    writer.writerow(buffer)

                    # print to the screen
                    shape = mm.get_shape()
                    print(f'''{file:30} {shape[0]:>10}:{shape[
                        1]:<10} {bt:^10.4f} {bf:10}''', end='')
                    for (k, v) in ob.items():
                        print(f'{v:10.4f}', end='')
                    print()


def read_matrix_market(file=None):
    if not file:
        pass

    mm = sio.mmread(file)
    if type(mm) is np.ndarray:
        mm = ss.coo_matrix(mm)
    return mm


def format_comparison(m: ss.spmatrix):
    # make the operand to be the unit vector in associate shape
    multiplier = np.ones((m.get_shape()[1], 1))

    observation = dict()
    for f in matrix_formats:
        m, p, af, t = measure_multiplication(m, multiplier, f)
        observation[af] = t

    # best format and the elapsed time
    k = matrix_formats[0]
    v = observation[matrix_formats[0]]

    # one loop to find the best format
    for (key, value) in observation.items():
        if value < v:
            k = key
            v = value

    return m, p, k, v, observation


def measure_multiplication(m: ss.spmatrix or np.ndarray, operand: object,
                           f: str = 'coo') -> object:
    mm = m
    actual_f = f
    if type(m) == ss.spmatrix:
        try:
            mm = m.asformat(f.lower(), copy=True)
        except RuntimeError:
            return m, DENSE, math.inf
    elif type(m) == np.ndarray:
        actual_f = DENSE

    start = time.time()
    product = mm.dot(operand)
    end = time.time()
    elapse_time = (end - start) * 1000  # in millisecond

    return mm, product, actual_f, elapse_time


if __name__ == '__main__':
    root = os.getcwd()
    if len(sys.argv) > 2:
        dataset = sys.argv[1]
        result = sys.argv[2]
        if not os.path.isdir(dataset):
            raise RuntimeError(dataset + 'is not found')
        if not os.path.isdir(result):
            raise RuntimeError(result + 'is not found')
    elif len(sys.argv) > 1:
        dataset = sys.argv[1]
        if not os.path.isdir(dataset):
            raise RuntimeError(dataset + 'is not found')
        result = os.path.join(root, 'result')
        if not os.path.exists(result):
            os.makedirs(result)
    else:
        dataset = os.path.join(root, 'test', 'dataset')
        result = os.path.join(root, 'test', 'result')

        if not os.path.exists(dataset):
            os.makedirs(dataset)
        if not os.path.exists(result):
            os.makedirs(result)

    load_data(dataset, result)