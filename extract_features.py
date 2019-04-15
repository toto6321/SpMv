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
    output_file_name = 'feature_tables_' + date_string + '.csv'
    output_path = os.path.join(result, output_file_name)

    with open(output_path, 'w', newline='') as csvfile:
        fn = ['FILE', 'N_ROWS', 'N_COLS', 'NNZ_TOTAL', 'DENSITY',
              'NNZ_MAX', 'NNZ_MEAN', 'NNZ_STD', 'BEST_TIME', 'BEST_FORMAT'] \
             + matrix_formats
        writer = csv.DictWriter(csvfile, dialect=csv.unix_dialect,
                                fieldnames=fn, delimiter=',')

        # table head
        writer.writeheader()

        # print to screen
        string = (
            f'{fn[0]:<30s}'
            f'{fn[1]:>10s}'
            f'{fn[2]:>10s}'
            f'{fn[3]:>10s}'
            f'{fn[4]:>10s}'
            f'{fn[5]:>10s}'
            f'{fn[6]:>10s}'
            f'{fn[7]:>10s}'
            f'{fn[8]:>10s}'
            f'{fn[9]:>10s}'
        )
        print(string, end='')
        for f in matrix_formats:
            print(f'{f.upper():>10s}', end='')
        print()

        for dirpath, dirnames, files in os.walk(dataset):
            for file in files:
                if file.endswith('.mtx'):
                    abspath = os.path.join(dirpath, file)
                    mm = read_matrix_market(abspath)

                    # regular feature extraction
                    n_rows, n_cols, nnz_total, density, nnz_max, nnz_mean, \
                    nnz_std = regular_feature_extraction(mm)
                    _, p, bf, bt, ob = format_comparison(mm)

                    shape = mm.get_shape()

                    # write to csv file
                    buffer = {
                        fn[0]: f'{file:<30}',
                        fn[1]: f'{n_rows:>10d}',
                        fn[2]: f'{n_cols:>10d}',
                        fn[3]: f'{nnz_total:>10.0f}',
                        fn[4]: f'{density:>10.2f}',
                        fn[5]: f'{nnz_max:>10.0f}',
                        fn[6]: f'{nnz_mean:>10.2f}',
                        fn[7]: f'{nnz_std:>10.2f}',
                        fn[8]: f'{bt:>10.4f}',
                        fn[9]: f'{bf:>10s}',
                    }
                    formatted_observation = ob.copy()
                    for (k, v) in formatted_observation.items():
                        formatted_observation[k] = f'{v:>10.4f}'
                    buffer.update(formatted_observation)
                    writer.writerow(buffer)

                    # print to the screen
                    string = (
                        f'{file:<30s}'
                        f'{n_rows:>10d}'
                        f'{n_cols:>10d}'
                        f'{nnz_total:>10.0f}'
                        f'{density:>10.2f}'
                        f'{nnz_max:>10.0f}'
                        f'{nnz_mean:>10.2f}'
                        f'{nnz_std:>10.2f}'
                        f'{bt:>10.4f}'
                        f'{bf:>10s}'
                    )
                    print(string, end='')

                    for (k, v) in ob.items():
                        print(f'{v:>10.4f}', end='')
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
    # multiplier = np.fromfunction(lambda i, j: i, (m.get_shape()[1], 1))
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

    return multiplier, p, k, v, observation


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


def regular_feature_extraction(m: ss.spmatrix or np.ndarray = None):
    if m is None:
        return

    if isinstance(m, ss.spmatrix):
        n_rows, n_cols = m.get_shape()
        nnz_list = np.zeros(n_rows)

        # generate nnz list by scanning non-zero element indexes list
        row_indexes = m.row
        for i in range(0, len(row_indexes)):
            nnz_list[row_indexes[i]] += 1
    else:
        # then it should be a np.ndarray
        n_rows, n_cols = m.shape
        nnz_list = np.zeros(n_rows)

        # generate nnz list by scanning non-zero element row by row
        for r in range(0, n_rows):
            for c in range(0, n_cols):
                if m[r, c] != 0:
                    nnz_list[r] += 1

    nnz_mean = nnz_list.mean(0)
    nnz_total = nnz_list.sum(0)
    density = nnz_total * 100 / (n_rows * n_cols)
    nnz_max = nnz_list.max(0)
    nnz_std = nnz_list.std(0)

    return n_rows, n_cols, nnz_total, density, nnz_max, nnz_mean, nnz_std


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
