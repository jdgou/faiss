#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 jack <jack@6k>
#
# Distributed under terms of the MIT license.

"""
filter with hamming binaries
"""

import faiss
import numpy as np

ptr = faiss.swig_ptr

d = 4096
n = 1_000_000

xb = np.random.rand(n // 8 // 8, d).view('uint8').reshape(n, d // 8)
index = faiss.index_binary_factory(d, 'BFlat')

bsc = faiss.BinSC(d)

k = 2000
ntotal = n
dist = np.empty(k, dtype='i4')
_ids = np.empty(k, dtype='i8')
xq = np.random.rand(d).astype('f4')
faiss.omp_set_num_threads(1)

# timeit
bsc.assign(ntotal, ptr(xq), ptr(xb), k, ptr(_ids))

# ref
index.add(xb)
index.search(xb[:1], k)

