//
// Created by jack on 2020/11/18.
//

#ifndef FAISS_SQ_H
#define FAISS_SQ_H

#include <memory>

#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/FaissException.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/utils.h>

namespace faiss {
    using idx_t = Index::idx_t;

    struct SQS : ScalarQuantizer {
        const IndexScalarQuantizer *index;
        InvertedListScanner *scanner;

        explicit SQS(const IndexScalarQuantizer *index) : index(index), ScalarQuantizer(index->sq) {
            scanner = select_Scanner(index->metric_type, nullptr, true);
        }

        InvertedListScanner *select_Scanner(MetricType mt, const Index *quantizer, bool store_pairs,
                                            bool by_residual = false) const;

        void search(idx_t ntotal, const float *x, const uint8_t *codes, const idx_t *ids, idx_t k, float *distances,
                    idx_t *labels) {

            // adaptor
            auto metric_type = index->metric_type;
            auto d = index->d;
            auto n = 1;

#pragma omp parallel
            {
#pragma omp for
                for (idx_t i = 0; i < n; i++) {
                    float *D = distances + k * i;
                    idx_t *I = labels + k * i;
                    // re-order heap
                    if (metric_type == METRIC_L2) {
                        maxheap_heapify(k, D, I);
                    } else {
                        minheap_heapify(k, D, I);
                    }
                    scanner->set_query(x + i * d);
                    scanner->scan_codes(ntotal, codes,
                                        ids, D, I, k);

                    // re-order heap
                    if (metric_type == METRIC_L2) {
                        maxheap_reorder(k, D, I);
                    } else {
                        minheap_reorder(k, D, I);
                    }
                }
            }
        }
    };

    struct BinSC {
        size_t d;
        size_t code_size;

        explicit BinSC(int d) : d(d), code_size(d / 8) {}

        void assign(idx_t ntotal, const float *q, const uint8_t *codes, idx_t k, idx_t *labels) const {

            // adaptor
            auto nn = 1;
            auto s = 0;
            auto xb = codes;
            auto x = std::unique_ptr<uint8_t[]>(new uint8_t[code_size]);
            auto distances = std::unique_ptr<int32_t[]>(new int32_t[k]);
            real_to_binary(d, q, x.get());

            int_maxheap_array_t res = {
                    size_t(nn), size_t(k), labels + s * k, distances.get() + s * k
            };

            hammings_knn_hc(&res, x.get() + s * code_size, xb, ntotal, code_size,
                    /* ordered = */ true);
        }
    };
}

#endif //FAISS_SQ_H
