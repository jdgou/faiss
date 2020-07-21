//
// Created by jack on 2020/7/19.
//

#include <faiss/extra/AttrIndex.h>
#include <faiss/IndexIVF.h>


#include <omp.h>

#include <cstdio>
#include <memory>

#include <faiss/utils/utils.h>
#include <faiss/utils/hamming.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/extra/AttrIndex.h>
#include <faiss/extra/AttrScanner.h>


namespace faiss {
    // attribute core filter function
    bool check_attr(const enc_t *query, const enc_t *mask, const enc_t *attr) {
        if (*attr % 100 == 0)
            return true;
        else
            return false;
    }

    // forward declarations

    template<>
    void AttrIndex<IndexIVFFlat>::add_with_ids_attr(idx_t n, const float *x, const idx_t *xids, const enc_t *attrs);

    // IndexIVF

    template<typename IndexType>
    void AttrIndex<IndexType>::add_attr(idx_t n, const float *x, const enc_t *a) {
        add_with_ids_attr(n, x, nullptr, a);
    }

    template<typename IndexType>
    void AttrIndex<IndexType>::add_with_ids_attr(idx_t n, const float *x, const idx_t *xids, const enc_t *attrs) {
        // do some blocking to avoid excessive allocs
        idx_t bs = 65536;
        if (n > bs) {
            for (idx_t i0 = 0; i0 < n; i0 += bs) {
                idx_t i1 = std::min(n, i0 + bs);
                if (this->verbose) {
                    printf("   IndexIVF::add_with_ids %ld:%ld\n", i0, i1);
                }
                add_with_ids_attr(i1 - i0, x + i0 * this->d,
                                  xids ? xids + i0 : nullptr, attrs + i0);
            }
            return;
        }

        FAISS_THROW_IF_NOT (this->is_trained);
        this->direct_map.check_can_add(xids);

        std::unique_ptr<idx_t[]> idx(new idx_t[n]);
        this->quantizer->assign(n, x, idx.get());
        size_t nadd = 0, nminus1 = 0;

        for (size_t i = 0; i < n; i++) {
            if (idx[i] < 0) nminus1++;
        }

        std::unique_ptr<uint8_t[]> flat_codes(new uint8_t[n * this->code_size]);
        this->encode_vectors(n, x, idx.get(), flat_codes.get());

        DirectMapAdd dm_adder(this->direct_map, n, xids);

#pragma omp parallel reduction(+: nadd)
        {
            int nt = omp_get_num_threads();
            int rank = omp_get_thread_num();

// each thread takes care of a subset of lists
            for (size_t i = 0; i < n; i++) {
                idx_t list_no = idx[i];
                if (list_no >= 0 && list_no % nt == rank) {
                    idx_t id = xids ? xids[i] : this->ntotal + i;
                    size_t ofs = dynamic_cast<AttrArrayInvertedLists *>(this->invlists)->add_entry_attr(
                            list_no, id,
                            flat_codes.get() + i * this->code_size,
                            *(attrs + i)
                    );

                    dm_adder.add(i, list_no, ofs);

                    nadd++;
                } else if (rank == 0 && list_no == -1) {
                    dm_adder.add(i, -1, 0);
                }
            }
        }


        if (this->verbose) {
            printf("    added %ld / %ld vectors (%ld -1s)\n", nadd, n, nminus1);
        }

        this->ntotal += n;
    }

    template<typename IndexType>
    void AttrIndex<IndexType>::search_preassigned_attr(idx_t n, const float *x, idx_t k, enc_t query,
                                                       enc_t mask, const idx_t *keys, const float *coarse_dis,
                                                       float *distances, idx_t *labels, bool store_pairs,
                                                       const IVFSearchParameters *params) const {
        long nprobe = params ? params->nprobe : this->nprobe;
        long max_codes = params ? params->max_codes : this->max_codes;

        size_t nlistv = 0, ndis = 0, nheap = 0;

        using HeapForIP = CMin<float, idx_t>;
        using HeapForL2 = CMax<float, idx_t>;

        bool interrupt = false;

        int pmode = this->parallel_mode & ~this->PARALLEL_MODE_NO_HEAP_INIT;
        bool do_heap_init = !(this->parallel_mode & this->PARALLEL_MODE_NO_HEAP_INIT);

        // don't start parallel section if single query
        bool do_parallel =
                pmode == 0 ? n > 1 :
                pmode == 1 ? nprobe > 1 :
                nprobe * n > 1;

#pragma omp parallel if(do_parallel) reduction(+: nlistv, ndis, nheap)
        {
            AttrInvertedListScanner *scanner = dynamic_cast<AttrInvertedListScanner *>(this->get_InvertedListScanner(
                    store_pairs));
            ScopeDeleter1<InvertedListScanner> del(scanner);

            /*****************************************************
             * Depending on parallel_mode, there are two possible ways
             * to organize the search. Here we define local functions
             * that are in common between the two
             ******************************************************/

            // intialize + reorder a result heap

            auto init_result = [&](float *simi, idx_t *idxi) {
                if (!do_heap_init) return;
                if (this->metric_type == METRIC_INNER_PRODUCT) {
                    heap_heapify<HeapForIP>(k, simi, idxi);
                } else {
                    heap_heapify<HeapForL2>(k, simi, idxi);
                }
            };

            auto reorder_result = [&](float *simi, idx_t *idxi) {
                if (!do_heap_init) return;
                if (this->metric_type == METRIC_INNER_PRODUCT) {
                    heap_reorder<HeapForIP>(k, simi, idxi);
                } else {
                    heap_reorder<HeapForL2>(k, simi, idxi);
                }
            };

            // single list scan using the current scanner (with query
            // set porperly) and storing results in simi and idxi
            auto scan_one_list = [&](idx_t key, float coarse_dis_i,
                                     float *simi, idx_t *idxi) {

                if (key < 0) {
                    // not enough centroids for multiprobe
                    return (size_t) 0;
                }
                FAISS_THROW_IF_NOT_FMT (key < (idx_t) this->nlist,
                                        "Invalid key=%ld nlist=%ld\n",
                                        key, this->nlist);

                size_t list_size = this->invlists->list_size(key);

                // don't waste time on empty lists
                if (list_size == 0) {
                    return (size_t) 0;
                }

                scanner->set_list(key, coarse_dis_i);

                nlistv++;

                InvertedLists::ScopedCodes scodes(this->invlists, key);

                std::unique_ptr<InvertedLists::ScopedIds> sids;
                const Index::idx_t *ids = nullptr;

                if (!store_pairs) {
                    sids.reset(new InvertedLists::ScopedIds(this->invlists, key));
                    ids = sids->get();
                }

                AttrArrayInvertedLists::ScopedAttrs sattrs(dynamic_cast<AttrArrayInvertedLists *>(this->invlists), key);

                nheap += scanner->scan_codes_with_filter(list_size, sattrs.get(), scodes.get(),
                                                         ids, simi, idxi, k);

                return list_size;
            };

            /****************************************************
             * Actual loops, depending on parallel_mode
             ****************************************************/

            if (pmode == 0) {

#pragma omp for
                for (size_t i = 0; i < n; i++) {

                    if (interrupt) {
                        continue;
                    }

                    // loop over queries
                    scanner->set_query(x + i * this->d);
                    scanner->set_query_and_mask(&query, &mask);
                    float *simi = distances + i * k;
                    idx_t *idxi = labels + i * k;

                    init_result(simi, idxi);

                    long nscan = 0;

                    // loop over probes
                    for (size_t ik = 0; ik < nprobe; ik++) {

                        nscan += scan_one_list(
                                keys[i * nprobe + ik],
                                coarse_dis[i * nprobe + ik],
                                simi, idxi
                        );

                        if (max_codes && nscan >= max_codes) {
                            break;
                        }
                    }

                    ndis += nscan;
                    reorder_result(simi, idxi);

                    if (InterruptCallback::is_interrupted()) {
                        interrupt = true;
                    }

                } // parallel for
            } else if (pmode == 1) {
                std::vector<idx_t> local_idx(k);
                std::vector<float> local_dis(k);

                for (size_t i = 0; i < n; i++) {
                    scanner->set_query(x + i * this->d);
                    scanner->set_query_and_mask(&query, &mask);
                    init_result(local_dis.data(), local_idx.data());

#pragma omp for schedule(dynamic)
                    for (size_t ik = 0; ik < nprobe; ik++) {
                        ndis += scan_one_list
                                (keys[i * nprobe + ik],
                                 coarse_dis[i * nprobe + ik],
                                 local_dis.data(), local_idx.data());

                        // can't do the test on max_codes
                    }
                    // merge thread-local results

                    float *simi = distances + i * k;
                    idx_t *idxi = labels + i * k;
#pragma omp single
                    init_result(simi, idxi);

#pragma omp barrier
#pragma omp critical
                    {
                        if (this->metric_type == METRIC_INNER_PRODUCT) {
                            heap_addn<HeapForIP>
                                    (k, simi, idxi,
                                     local_dis.data(), local_idx.data(), k);
                        } else {
                            heap_addn<HeapForL2>
                                    (k, simi, idxi,
                                     local_dis.data(), local_idx.data(), k);
                        }
                    }
#pragma omp barrier
#pragma omp single
                    reorder_result(simi, idxi);
                }
            } else {
                FAISS_THROW_FMT ("parallel_mode %d not supported\n",
                                 pmode);
            }
        } // parallel section

        if (interrupt) {
            FAISS_THROW_MSG ("computation interrupted");
        }

        indexIVF_stats.nq += n;
        indexIVF_stats.nlist += nlistv;
        indexIVF_stats.ndis += ndis;
        indexIVF_stats.nheap_updates += nheap;

    }

    template<typename IndexType>
    void AttrIndex<IndexType>::search_attr(idx_t n, const float *x, idx_t k, enc_t query, enc_t mask, float *distances,
                                           idx_t *labels) const {
        std::unique_ptr<idx_t[]> idx(new idx_t[n * this->nprobe]);
        std::unique_ptr<float[]> coarse_dis(new float[n * this->nprobe]);

        double t0 = getmillisecs();
        this->quantizer->search(n, x, this->nprobe, coarse_dis.get(), idx.get());
        indexIVF_stats.quantization_time += getmillisecs() - t0;

        t0 = getmillisecs();
        this->invlists->prefetch_lists(idx.get(), n * this->nprobe);

        this->search_preassigned_attr(n, x, k, query, mask, idx.get(), coarse_dis.get(),
                                      distances, labels, false);
        indexIVF_stats.search_time += getmillisecs() - t0;
    }

    template void
    AttrIndex<IndexIVF>::search_attr(idx_t n, const float *x, idx_t k, enc_t query, enc_t mask, float *distances,
                                     idx_t *labels) const;

    template<typename IndexType>
    InvertedListScanner *AttrIndex<IndexType>::get_InvertedListScanner(bool store_pairs) const {
        return nullptr;
    }

    // IndexIVFFlat

    template<>
    AttrIndex<IndexIVFFlat>::AttrIndex(
            Index *quantizer, size_t d, size_t nlist_,
            MetricType metric) : IndexIVFFlat(quantizer, d, nlist_, metric) {
        this->invlists = new AttrArrayInvertedLists(this->nlist, this->code_size);
    }

    template void AttrIndex<IndexIVFFlat>::add_attr(idx_t n, const float *x, const enc_t *a);

    template<>
    void AttrIndex<IndexIVFFlat>::add_core_attr(idx_t n, const enc_t *attrs, const float *x, const idx_t *xids,
                                                const int64_t *precomputed_idx) {
        FAISS_THROW_IF_NOT (this->is_trained);
        assert (this->invlists);
        this->direct_map.check_can_add(xids);
        const int64_t *idx;
        ScopeDeleter<int64_t> del;

        if (precomputed_idx) {
            idx = precomputed_idx;
        } else {
            int64_t *idx0 = new int64_t[n];
            del.set(idx0);
            this->quantizer->assign(n, x, idx0);
            idx = idx0;
        }
        int64_t n_add = 0;
        for (size_t i = 0; i < n; i++) {
            idx_t id = xids ? xids[i] : this->ntotal + i;
            idx_t list_no = idx[i];
            size_t offset;

            if (list_no >= 0) {
                const float *xi = x + i * this->d;
                offset = dynamic_cast<AttrArrayInvertedLists *>(this->invlists)->add_entry_attr(
                        list_no, id, (const uint8_t *) xi, *(attrs + i));
                n_add++;
            } else {
                offset = 0;
            }
            this->direct_map.add_single_id(id, list_no, offset);
        }

        if (this->verbose) {
            printf("IndexIVFFlat::add_core: added %ld / %ld vectors\n",
                   n_add, n);
        }
        this->ntotal += n;
    }

    template<>
    void AttrIndex<IndexIVFFlat>::add_with_ids_attr(idx_t n, const float *x, const idx_t *xids, const enc_t *attrs) {
        add_core_attr(n, attrs, x, xids, nullptr);
    }

    template void
    AttrIndex<IndexIVFFlat>::search_attr(idx_t n, const float *x, idx_t k, enc_t query, enc_t mask, float *distances,
                                         idx_t *labels) const;


    template<>
    InvertedListScanner *AttrIndex<IndexIVFFlat>::get_InvertedListScanner(bool store_pairs) const {
        if (this->metric_type == METRIC_INNER_PRODUCT) {
            return new AttrIVFFlatScanner<
                    METRIC_INNER_PRODUCT, CMin<float, int64_t> >(this->d, store_pairs);
        } else if (this->metric_type == METRIC_L2) {
            return new AttrIVFFlatScanner<
                    METRIC_L2, CMax<float, int64_t> >(this->d, store_pairs);
        } else {
            FAISS_THROW_MSG("metric type not supported");
        }
        return nullptr;
    }

    // IndexIVFSQ

    template<>
    AttrIndex<IndexIVFScalarQuantizer>::AttrIndex(
            Index *quantizer, size_t d, size_t nlist, ScalarQuantizer::QuantizerType qtype, MetricType metric,
            bool encode_residual): IndexIVFScalarQuantizer(quantizer, d, nlist, qtype, metric, encode_residual) {
        this->invlists = new AttrArrayInvertedLists(this->nlist, this->code_size);
    }

    template<>
    void AttrIndex<IndexIVFScalarQuantizer>::add_with_ids_attr(idx_t n, const float *x, const idx_t *xids,
                                                               const enc_t *attrs) {
        FAISS_THROW_IF_NOT (is_trained);
        std::unique_ptr<int64_t[]> idx(new int64_t[n]);
        quantizer->assign(n, x, idx.get());
        size_t nadd = 0;
        std::unique_ptr<ScalarQuantizer::Quantizer> squant(sq.select_quantizer());

        DirectMapAdd dm_add(direct_map, n, xids);

#pragma omp parallel reduction(+: nadd)
        {
            std::vector<float> residual(d);
            std::vector<uint8_t> one_code(code_size);
            int nt = omp_get_num_threads();
            int rank = omp_get_thread_num();

            // each thread takes care of a subset of lists
            for (size_t i = 0; i < n; i++) {
                int64_t list_no = idx[i];
                if (list_no >= 0 && list_no % nt == rank) {
                    int64_t id = xids ? xids[i] : ntotal + i;

                    const float *xi = x + i * d;
                    if (by_residual) {
                        quantizer->compute_residual(xi, residual.data(), list_no);
                        xi = residual.data();
                    }

                    memset(one_code.data(), 0, code_size);
                    squant->encode_vector(xi, one_code.data());

                    size_t ofs = dynamic_cast<AttrArrayInvertedLists *>(invlists)->add_entry_attr(list_no, id,
                                                                                                  one_code.data(),
                                                                                                  attrs[i]);

                    dm_add.add(i, list_no, ofs);
                    nadd++;

                } else if (rank == 0 && list_no == -1) {
                    dm_add.add(i, -1, 0);
                }
            }
        }


        ntotal += n;
    }

    template<>
    InvertedListScanner *AttrIndex<IndexIVFScalarQuantizer>::get_InvertedListScanner(bool store_pairs) const {
        return sq.select_InvertedListScanner(metric_type, quantizer, store_pairs,
                                             by_residual);
    }

    // inv impl

    AttrArrayInvertedLists::AttrArrayInvertedLists(size_t nlist, size_t code_size) : InvertedLists(nlist, code_size) {
        ids.resize(nlist);
        codes.resize(nlist);
        attrs.resize(nlist);
    }

    size_t AttrArrayInvertedLists::list_size(size_t list_no) const {
        assert (list_no < nlist);
        return ids[list_no].size();
    }

    const uint8_t *AttrArrayInvertedLists::get_codes(size_t list_no) const {
        assert (list_no < nlist);
        return codes[list_no].data();
    }

    const idx_t *AttrArrayInvertedLists::get_ids(size_t list_no) const {
        assert (list_no < nlist);
        return ids[list_no].data();
    }

    const enc_t *AttrArrayInvertedLists::get_attrs(size_t list_no) const {
        assert (list_no < nlist);
        return attrs[list_no].data();
    }

    void AttrArrayInvertedLists::release_attrs(size_t list_no, const enc_t *attrs) const {}

    enc_t AttrArrayInvertedLists::get_single_attr(size_t list_no, size_t offset) const {
        assert (offset < list_size(list_no));
        return get_attrs(list_no)[offset];
    }

    size_t AttrArrayInvertedLists::add_entry_attr(size_t list_no, idx_t theid, const uint8_t *code, enc_t attr) {
        return add_entries_attr(list_no, 1, &theid, code, &attr);
    }

    size_t
    AttrArrayInvertedLists::add_entries(size_t list_no, size_t n_entry, const idx_t *ids_in, const uint8_t *code) {
        if (n_entry == 0) return 0;
        assert (list_no < nlist);
        size_t o = ids[list_no].size();
        ids[list_no].resize(o + n_entry);
        memcpy(&ids[list_no][o], ids_in, sizeof(ids_in[0]) * n_entry);
        codes[list_no].resize((o + n_entry) * code_size);
        memcpy(&codes[list_no][o * code_size], code, code_size * n_entry);
        return o;
    }

    size_t
    AttrArrayInvertedLists::add_entries_attr(size_t list_no, size_t n_entry, const idx_t *ids_in, const uint8_t *code,
                                             const enc_t *attrs_in) {
        if (n_entry == 0) return 0;
        assert (list_no < nlist);
        size_t o = ids[list_no].size();
        ids[list_no].resize(o + n_entry);
        memcpy(&ids[list_no][o], ids_in, sizeof(ids_in[0]) * n_entry);
        codes[list_no].resize((o + n_entry) * code_size);
        memcpy(&codes[list_no][o * code_size], code, code_size * n_entry);
        attrs[list_no].resize(o + n_entry);
        memcpy(&attrs[list_no][o], attrs_in, sizeof(enc_t) * n_entry);
        return o;
    }

    void AttrArrayInvertedLists::update_entries(size_t list_no, size_t offset, size_t n_entry, const idx_t *ids_in,
                                                const uint8_t *codes_in) {
        assert (list_no < nlist);
        assert (n_entry + offset <= ids[list_no].size());
        memcpy(&ids[list_no][offset], ids_in, sizeof(ids_in[0]) * n_entry);
        memcpy(&codes[list_no][offset * code_size], codes_in, code_size * n_entry);
    }

    void AttrArrayInvertedLists::update_entries_attr(size_t list_no, size_t offset, size_t n_entry, const idx_t *ids_in,
                                                     const uint8_t *codes_in, const enc_t *attrs_in) {
        assert (list_no < nlist);
        assert (n_entry + offset <= ids[list_no].size());
        memcpy(&ids[list_no][offset], ids_in, sizeof(ids_in[0]) * n_entry);
        memcpy(&codes[list_no][offset * code_size], codes_in, code_size * n_entry);
        memcpy(&attrs[list_no][offset], attrs_in, sizeof(enc_t) * n_entry);
    }

    void AttrArrayInvertedLists::resize(size_t list_no, size_t new_size) {
        ids[list_no].resize(new_size);
        codes[list_no].resize(new_size * code_size);
        attrs[list_no].resize(new_size);
    }

    AttrArrayInvertedLists::~AttrArrayInvertedLists() = default;

}