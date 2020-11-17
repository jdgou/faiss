#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 jack <jack@6k>
#
# Distributed under terms of the MIT license.

"""
rerank with fp16 vectors
"""

import faiss
import numpy as np

ptr = faiss.swig_ptr

d = 4096
n = 100_000

xb = np.random.rand(n, d).astype('f4')
index = faiss.index_factory(d, 'SQfp16')
index.metric_type = 0
sq = faiss.SQS(index)

k = 35
codes = index.sa_encode(xb)
ntotal = 10000
ids = np.random.randint(0, n, ntotal)
dist = np.empty(k, dtype='f4')
_ids = np.empty(k, dtype='i8')
faiss.omp_set_num_threads(1)

# timeit
sq.search(ntotal, ptr(xb[:1]), ptr(codes), ptr(ids), k, ptr(dist), ptr(_ids))

# ref
index.add(xb[:ntotal])
index.search(xb[:1], k)

