//
// Created by jack on 2020/7/21.
//

#ifndef FAISS_ATTRSCANNER_H
#define FAISS_ATTRSCANNER_H


#include <faiss/MetricType.h>
#include <faiss/extra/AttrIndex.h>
#include <faiss/utils/distances.h>
#include <faiss/impl/FaissAssert.h>

namespace faiss {
    namespace {
        template<MetricType metric, class C>
        struct AttrIVFFlatScanner : AttrInvertedListScanner {

            size_t d;
            bool store_pairs;

            AttrIVFFlatScanner(size_t d, bool store_pairs) :
                    d(d), store_pairs(store_pairs) {}

            const float *xi;

            void set_query(const float *query) override {
                this->xi = query;
            }

            idx_t list_no;

            void set_list(idx_t list_no, float /* coarse_dis */) override {
                this->list_no = list_no;
            }

            float distance_to_code(const uint8_t *code) const override {
                const float *yj = (float *) code;
                float dis = metric == METRIC_INNER_PRODUCT ?
                            fvec_inner_product(xi, yj, d) : fvec_L2sqr(xi, yj, d);
                return dis;
            }

            size_t scan_codes(size_t list_size,
                              const uint8_t *codes,
                              const idx_t *ids,
                              float *simi, idx_t *idxi,
                              size_t k) const override {
                const float *list_vecs = (const float *) codes;
                size_t nup = 0;
                for (size_t j = 0; j < list_size; j++) {
                    const float *yj = list_vecs + d * j;
                    float dis = metric == METRIC_INNER_PRODUCT ?
                                fvec_inner_product(xi, yj, d) : fvec_L2sqr(xi, yj, d);
                    if (C::cmp(simi[0], dis)) {
                        heap_pop<C>(k, simi, idxi);
                        int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                        heap_push<C>(k, simi, idxi, dis, id);
                        nup++;
                    }
                }
                return nup;
            }

            size_t scan_codes_with_filter(size_t list_size,
                                          const enc_t *attrs,
                                          const uint8_t *codes,
                                          const idx_t *ids,
                                          float *simi, idx_t *idxi,
                                          size_t k) const override {
                const float *list_vecs = (const float *) codes;
                size_t nup = 0;
                for (size_t j = 0; j < list_size; j++) {
                    if (not check_attr(attr, mask, attrs + j)) {
                        continue;
                    }
                    const float *yj = list_vecs + d * j;
                    float dis = metric == METRIC_INNER_PRODUCT ?
                                fvec_inner_product(xi, yj, d) : fvec_L2sqr(xi, yj, d);
                    if (C::cmp(simi[0], dis)) {
                        heap_pop<C>(k, simi, idxi);
                        int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                        heap_push<C>(k, simi, idxi, dis, id);
                        nup++;
                    }
                }
                return nup;
            }

            void scan_codes_range(size_t list_size,
                                  const uint8_t *codes,
                                  const idx_t *ids,
                                  float radius,
                                  RangeQueryResult &res) const override {
                const float *list_vecs = (const float *) codes;
                for (size_t j = 0; j < list_size; j++) {
                    const float *yj = list_vecs + d * j;
                    float dis = metric == METRIC_INNER_PRODUCT ?
                                fvec_inner_product(xi, yj, d) : fvec_L2sqr(xi, yj, d);
                    if (C::cmp(radius, dis)) {
                        int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                        res.add(dis, id);
                    }
                }
            }


        };
    } // anonymous namespace

    namespace {


        template<class DCClass>
        struct IVFSQScannerIP : InvertedListScanner {
            DCClass dc;
            bool store_pairs, by_residual;

            size_t code_size;

            idx_t list_no;  /// current list (set to 0 for Flat index
            float accu0;    /// added to all distances

            IVFSQScannerIP(int d, const std::vector<float> &trained,
                           size_t code_size, bool store_pairs,
                           bool by_residual) :
                    dc(d, trained), store_pairs(store_pairs),
                    by_residual(by_residual),
                    code_size(code_size), list_no(0), accu0(0) {}


            void set_query(const float *query) override {
                dc.set_query(query);
            }

            void set_list(idx_t list_no, float coarse_dis) override {
                this->list_no = list_no;
                accu0 = by_residual ? coarse_dis : 0;
            }

            float distance_to_code(const uint8_t *code) const final {
                return accu0 + dc.query_to_code(code);
            }

            size_t scan_codes(size_t list_size,
                              const uint8_t *codes,
                              const idx_t *ids,
                              float *simi, idx_t *idxi,
                              size_t k) const override {
                size_t nup = 0;

                for (size_t j = 0; j < list_size; j++) {

                    float accu = accu0 + dc.query_to_code(codes);

                    if (accu > simi[0]) {
                        minheap_pop(k, simi, idxi);
                        int64_t id = store_pairs ? (list_no << 32 | j) : ids[j];
                        minheap_push(k, simi, idxi, accu, id);
                        nup++;
                    }
                    codes += code_size;
                }
                return nup;
            }

            void scan_codes_range(size_t list_size,
                                  const uint8_t *codes,
                                  const idx_t *ids,
                                  float radius,
                                  RangeQueryResult &res) const override {
                for (size_t j = 0; j < list_size; j++) {
                    float accu = accu0 + dc.query_to_code(codes);
                    if (accu > radius) {
                        int64_t id = store_pairs ? (list_no << 32 | j) : ids[j];
                        res.add(accu, id);
                    }
                    codes += code_size;
                }
            }


        };


        template<class DCClass>
        struct IVFSQScannerL2 : InvertedListScanner {

            DCClass dc;

            bool store_pairs, by_residual;
            size_t code_size;
            const Index *quantizer;
            idx_t list_no;    /// current inverted list
            const float *x;   /// current query

            std::vector<float> tmp;

            IVFSQScannerL2(int d, const std::vector<float> &trained,
                           size_t code_size, const Index *quantizer,
                           bool store_pairs, bool by_residual) :
                    dc(d, trained), store_pairs(store_pairs), by_residual(by_residual),
                    code_size(code_size), quantizer(quantizer),
                    list_no(0), x(nullptr), tmp(d) {
            }


            void set_query(const float *query) override {
                x = query;
                if (!quantizer) {
                    dc.set_query(query);
                }
            }


            void set_list(idx_t list_no, float /*coarse_dis*/) override {
                if (by_residual) {
                    this->list_no = list_no;
                    // shift of x_in wrt centroid
                    quantizer->compute_residual(x, tmp.data(), list_no);
                    dc.set_query(tmp.data());
                } else {
                    dc.set_query(x);
                }
            }

            float distance_to_code(const uint8_t *code) const final {
                return dc.query_to_code(code);
            }

            size_t scan_codes(size_t list_size,
                              const uint8_t *codes,
                              const idx_t *ids,
                              float *simi, idx_t *idxi,
                              size_t k) const override {
                size_t nup = 0;
                for (size_t j = 0; j < list_size; j++) {

                    float dis = dc.query_to_code(codes);

                    if (dis < simi[0]) {
                        maxheap_pop(k, simi, idxi);
                        int64_t id = store_pairs ? (list_no << 32 | j) : ids[j];
                        maxheap_push(k, simi, idxi, dis, id);
                        nup++;
                    }
                    codes += code_size;
                }
                return nup;
            }

            void scan_codes_range(size_t list_size,
                                  const uint8_t *codes,
                                  const idx_t *ids,
                                  float radius,
                                  RangeQueryResult &res) const override {
                for (size_t j = 0; j < list_size; j++) {
                    float dis = dc.query_to_code(codes);
                    if (dis < radius) {
                        int64_t id = store_pairs ? (list_no << 32 | j) : ids[j];
                        res.add(dis, id);
                    }
                    codes += code_size;
                }
            }


        };

        template<class DCClass>
        InvertedListScanner *sel2_InvertedListScanner
                (const ScalarQuantizer *sq,
                 const Index *quantizer, bool store_pairs, bool r) {
            if (DCClass::Sim::metric_type == METRIC_L2) {
                return new IVFSQScannerL2<DCClass>(sq->d, sq->trained, sq->code_size,
                                                   quantizer, store_pairs, r);
            } else if (DCClass::Sim::metric_type == METRIC_INNER_PRODUCT) {
                return new IVFSQScannerIP<DCClass>(sq->d, sq->trained, sq->code_size,
                                                   store_pairs, r);
            } else {
                FAISS_THROW_MSG("unsupported metric type");
            }
        }

        template<class Similarity, class Codec, bool uniform>
        InvertedListScanner *sel12_InvertedListScanner
                (const ScalarQuantizer *sq,
                 const Index *quantizer, bool store_pairs, bool r) {
            constexpr int SIMDWIDTH = Similarity::simdwidth;
            using QuantizerClass = QuantizerTemplate<Codec, uniform, SIMDWIDTH>;
            using DCClass = DCTemplate<QuantizerClass, Similarity, SIMDWIDTH>;
            return sel2_InvertedListScanner<DCClass>(sq, quantizer, store_pairs, r);
        }


        template<class Similarity>
        InvertedListScanner *sel1_InvertedListScanner
                (const ScalarQuantizer *sq, const Index *quantizer,
                 bool store_pairs, bool r) {
            constexpr int SIMDWIDTH = Similarity::simdwidth;
            switch (sq->qtype) {
                case ScalarQuantizer::QT_8bit_uniform:
                    return sel12_InvertedListScanner
                            <Similarity, Codec8bit, true>(sq, quantizer, store_pairs, r);
                case ScalarQuantizer::QT_4bit_uniform:
                    return sel12_InvertedListScanner
                            <Similarity, Codec4bit, true>(sq, quantizer, store_pairs, r);
                case ScalarQuantizer::QT_8bit:
                    return sel12_InvertedListScanner
                            <Similarity, Codec8bit, false>(sq, quantizer, store_pairs, r);
                case ScalarQuantizer::QT_4bit:
                    return sel12_InvertedListScanner
                            <Similarity, Codec4bit, false>(sq, quantizer, store_pairs, r);
                case ScalarQuantizer::QT_6bit:
                    return sel12_InvertedListScanner
                            <Similarity, Codec6bit, false>(sq, quantizer, store_pairs, r);
                case ScalarQuantizer::QT_fp16:
                    return sel2_InvertedListScanner
                            <DCTemplate < QuantizerFP16 < SIMDWIDTH>, Similarity, SIMDWIDTH > >
                                                                                  (sq, quantizer, store_pairs, r);
                case ScalarQuantizer::QT_8bit_direct:
                    if (sq->d % 16 == 0) {
                        return sel2_InvertedListScanner
                                       <DistanceComputerByte < Similarity, SIMDWIDTH> >
                               (sq, quantizer, store_pairs, r);
                    } else {
                        return sel2_InvertedListScanner
                                <DCTemplate < Quantizer8bitDirect < SIMDWIDTH>,
                                Similarity, SIMDWIDTH > >
                                            (sq, quantizer, store_pairs, r);
                    }

            }

            FAISS_THROW_MSG ("unknown qtype");
            return nullptr;
        }

        template<int SIMDWIDTH>
        InvertedListScanner *sel0_InvertedListScanner
                (MetricType mt, const ScalarQuantizer *sq,
                 const Index *quantizer, bool store_pairs, bool by_residual) {
            if (mt == METRIC_L2) {
                return sel1_InvertedListScanner<SimilarityL2 < SIMDWIDTH> >
                       (sq, quantizer, store_pairs, by_residual);
            } else if (mt == METRIC_INNER_PRODUCT) {
                return sel1_InvertedListScanner<SimilarityIP < SIMDWIDTH> >
                       (sq, quantizer, store_pairs, by_residual);
            } else {
                FAISS_THROW_MSG("unsupported metric type");
            }
        }

    } // anonymous namespace

    InvertedListScanner *ScalarQuantizer::select_InvertedListScanner
            (MetricType mt, const Index *quantizer,
             bool store_pairs, bool by_residual) const {
#ifdef USE_F16C
        if (d % 8 == 0) {
        return sel0_InvertedListScanner<8>
            (mt, this, quantizer, store_pairs, by_residual);
    } else
#endif
        {
            return sel0_InvertedListScanner<1>
                    (mt, this, quantizer, store_pairs, by_residual);
        }
    }

}

#endif //FAISS_ATTRSCANNER_H
