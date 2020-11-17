#! /bin/sh
#
# build.sh
# Copyright (C) 2020 jack <jack@6k>
#
# Distributed under terms of the MIT license.
#

set -eu


# Build libfaiss_avx2.so.
cmake -B _build_avx2 \
      -DBUILD_TESTING=OFF \
      -DFAISS_OPT_LEVEL=avx2 \
      -DFAISS_ENABLE_GPU=OFF \
      -DFAISS_ENABLE_PYTHON=ON \
      -DCMAKE_BUILD_TYPE=Release .

cmake --build _build_avx2 -j

