//
// Created by jack on 2020/7/19.
//

#ifndef FAISS_ATTRINDEX_H
#define FAISS_ATTRINDEX_H

#include <string>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/ScalarQuantizer.h>

namespace faiss {

    using idx_t = Index::idx_t;

    typedef int64_t enc_t;

    inline bool check_attr(const enc_t *query, const enc_t *mask, const enc_t *attr);

    template<typename IndexType>
    struct AttrIndex : public IndexType {

        // Index

        void add_attr(idx_t n, const float *x, const enc_t *a);

        void add_with_ids_attr(idx_t n, const float *x, const idx_t *xids, const enc_t *attrs);

        void search_attr(idx_t n, const float *x, idx_t k, enc_t q, enc_t mask,
                         float *distances, idx_t *labels) const;

        void range_search_attr(idx_t n, const float *x, float radius, enc_t q, enc_t mask,
                               RangeSearchResult *result) const;

        void assign_attr(idx_t n, const float *x, idx_t *labels, idx_t k = 1, enc_t q = 0, enc_t mask = 0) const;

        // IndexIVF

        void search_preassigned_attr(idx_t n, const float *x, idx_t k, enc_t query, enc_t mask,
                                     const idx_t *assign,
                                     const float *centroid_dis,
                                     float *distances, idx_t *labels,
                                     bool store_pairs,
                                     const IVFSearchParameters *params = nullptr
        ) const;

        InvertedListScanner *get_InvertedListScanner(bool store_pairs) const override;

        // IndexIVFFlat

        AttrIndex(Index *quantizer, size_t d, size_t nlist_, MetricType = METRIC_L2);

        void add_core_attr(idx_t n, const enc_t *attrs, const float *x, const idx_t *xids,
                           const int64_t *precomputed_idx);

        // IndexIVFSQ

        AttrIndex(Index *quantizer, size_t d, size_t nlist, ScalarQuantizer::QuantizerType qtype,
                  MetricType metric = METRIC_L2, bool encode_residual = true);

        // IndexIVFPQ

        AttrIndex();
    };

    struct AttrInvertedListScanner : InvertedListScanner {

        // new field
        const enc_t *attr;
        const enc_t *mask;

        void set_query_and_mask(const enc_t *query, const enc_t *mask_) {
            this->attr = query;
            this->mask = mask_;
        }

        virtual size_t scan_codes_with_filter(size_t n,
                                              const enc_t *attrs,
                                              const uint8_t *codes,
                                              const idx_t *ids,
                                              float *distances, idx_t *labels,
                                              size_t k) const = 0;
    };

    struct AttrArrayInvertedLists : InvertedLists {
        std::vector<std::vector<uint8_t> > codes; // binary codes, size nlist
        std::vector<std::vector<idx_t> > ids;  ///< Inverted lists for indexes
        std::vector<std::vector<enc_t> > attrs;

        AttrArrayInvertedLists(size_t nlist, size_t code_size);

        size_t list_size(size_t list_no) const override;

        const uint8_t *get_codes(size_t list_no) const override;

        const idx_t *get_ids(size_t list_no) const override;

        const enc_t *get_attrs(size_t list_no) const;

        void release_attrs(size_t list_no, const enc_t *attrs) const;

        enc_t get_single_attr(size_t list_no, size_t offset) const;

        size_t add_entry_attr(size_t list_no, idx_t theid, const uint8_t *code, enc_t attr);

        size_t add_entries(
                size_t list_no, size_t n_entry,
                const idx_t *ids, const uint8_t *code) override;

        size_t add_entries_attr(size_t list_no, size_t n_entry,
                                const idx_t *ids, const uint8_t *code, const enc_t *attrs);

        void update_entries(size_t list_no, size_t offset, size_t n_entry,
                            const idx_t *ids, const uint8_t *code) override;

        void update_entries_attr(size_t list_no, size_t offset, size_t n_entry,
                                 const idx_t *ids, const uint8_t *code, const enc_t *attrs);

        void resize(size_t list_no, size_t new_size) override;

        ~AttrArrayInvertedLists() override;

        struct ScopedAttrs {
            const AttrArrayInvertedLists *il;
            const enc_t *attrs;
            size_t list_no;

            ScopedAttrs(const AttrArrayInvertedLists *il, size_t list_no) :
                    il(il), attrs(il->get_attrs(list_no)), list_no(list_no) {}

            const enc_t *get() { return attrs; }

            enc_t operator[](size_t i) const {
                return attrs[i];
            }

            ~ScopedAttrs() {
                il->release_attrs(list_no, attrs);
            }
        };
    };
}

#endif //FAISS_ATTRINDEX_H
