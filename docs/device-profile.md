# Device profile (T83)

`cuda_available()` on this host: `False`

## n_items~1000, n_classes=30
- **cpu_only** (n_query=86, n_candidates=1910):

  | stage | seconds |
  |---|---|
  | 1_encode | 0.0037 |
  | 2_prototypes_and_freq | 0.0002 |
  | 3_description_prototype_similarity | 0.0000 |
  | 4_dense_topk | 0.0004 |
  | 5_bm25 | 0.0010 |
  | 6_scatter_knn | 0.0001 |
  | 7_ranks_margins_topn_minmax | 0.0002 |
  | 8_dataframe_construction | 0.0004 |
  | 9_fusion_predict | 0.0006 |
  | 10_calibration_thresholds | 0.0008 |
  | **sum** | **0.0073** |
  - assemble() wall = 0.0027s; comparable stage-sum / assemble() wall = 0.840

- **gpu_encoder_cpu_rest**: skipped (no CUDA device visible (torch.cuda.is_available() is False on this host); a real GPU-encoder config additionally needs a downloadable sentence-transformers model, which this offline harness does not fetch)
- **gpu_encoder_gpu_fusion**: skipped (no CUDA device visible (torch.cuda.is_available() is False on this host); a real GPU-encoder config additionally needs a downloadable sentence-transformers model, which this offline harness does not fetch)
- OOF run (n_folds=3): whole_run=0.070s, per_fold=0.023s

## n_items~10000, n_classes=30
- **cpu_only** (n_query=865, n_candidates=18287):

  | stage | seconds |
  |---|---|
  | 1_encode | 0.0358 |
  | 2_prototypes_and_freq | 0.0006 |
  | 3_description_prototype_similarity | 0.0000 |
  | 4_dense_topk | 0.0373 |
  | 5_bm25 | 0.0688 |
  | 6_scatter_knn | 0.0008 |
  | 7_ranks_margins_topn_minmax | 0.0014 |
  | 8_dataframe_construction | 0.0023 |
  | 9_fusion_predict | 0.0014 |
  | 10_calibration_thresholds | 0.0015 |
  | **sum** | **0.1500** |
  - assemble() wall = 0.1155s; comparable stage-sum / assemble() wall = 0.964

- **gpu_encoder_cpu_rest**: skipped (no CUDA device visible (torch.cuda.is_available() is False on this host); a real GPU-encoder config additionally needs a downloadable sentence-transformers model, which this offline harness does not fetch)
- **gpu_encoder_gpu_fusion**: skipped (no CUDA device visible (torch.cuda.is_available() is False on this host); a real GPU-encoder config additionally needs a downloadable sentence-transformers model, which this offline harness does not fetch)
- OOF run (n_folds=3): whole_run=0.953s, per_fold=0.318s

## n_items~100000, n_classes=30
- **cpu_only** (n_query=8665, n_candidates=200248):

  | stage | seconds |
  |---|---|
  | 1_encode | 0.3613 |
  | 2_prototypes_and_freq | 0.0091 |
  | 3_description_prototype_similarity | 0.0002 |
  | 4_dense_topk | 3.3213 |
  | 5_bm25 | 7.1450 |
  | 6_scatter_knn | 0.0090 |
  | 7_ranks_margins_topn_minmax | 0.0134 |
  | 8_dataframe_construction | 0.0238 |
  | 9_fusion_predict | 0.0116 |
  | 10_calibration_thresholds | 0.0095 |
  | **sum** | **10.9042** |
  - assemble() wall = 10.6833s; comparable stage-sum / assemble() wall = 0.985

- **gpu_encoder_cpu_rest**: skipped (no CUDA device visible (torch.cuda.is_available() is False on this host); a real GPU-encoder config additionally needs a downloadable sentence-transformers model, which this offline harness does not fetch)
- **gpu_encoder_gpu_fusion**: skipped (no CUDA device visible (torch.cuda.is_available() is False on this host); a real GPU-encoder config additionally needs a downloadable sentence-transformers model, which this offline harness does not fetch)
- OOF run (n_folds=3): whole_run=71.942s, per_fold=23.981s

## n_items~1000, n_classes=500
- **cpu_only** (n_query=400, n_candidates=13844):

  | stage | seconds |
  |---|---|
  | 1_encode | 0.0212 |
  | 2_prototypes_and_freq | 0.0040 |
  | 3_description_prototype_similarity | 0.0001 |
  | 4_dense_topk | 0.0085 |
  | 5_bm25 | 0.0097 |
  | 6_scatter_knn | 0.0013 |
  | 7_ranks_margins_topn_minmax | 0.0100 |
  | 8_dataframe_construction | 0.0019 |
  | 9_fusion_predict | 0.0012 |
  | 10_calibration_thresholds | 0.0013 |
  | **sum** | **0.0593** |
  - assemble() wall = 0.0380s; comparable stage-sum / assemble() wall = 0.935

- **gpu_encoder_cpu_rest**: skipped (no CUDA device visible (torch.cuda.is_available() is False on this host); a real GPU-encoder config additionally needs a downloadable sentence-transformers model, which this offline harness does not fetch)
- **gpu_encoder_gpu_fusion**: skipped (no CUDA device visible (torch.cuda.is_available() is False on this host); a real GPU-encoder config additionally needs a downloadable sentence-transformers model, which this offline harness does not fetch)
- OOF run (n_folds=3): whole_run=0.486s, per_fold=0.162s

## n_items~10000, n_classes=500
- **cpu_only** (n_query=913, n_candidates=29571):

  | stage | seconds |
  |---|---|
  | 1_encode | 0.0392 |
  | 2_prototypes_and_freq | 0.0060 |
  | 3_description_prototype_similarity | 0.0002 |
  | 4_dense_topk | 0.0387 |
  | 5_bm25 | 0.0427 |
  | 6_scatter_knn | 0.0029 |
  | 7_ranks_margins_topn_minmax | 0.0211 |
  | 8_dataframe_construction | 0.0036 |
  | 9_fusion_predict | 0.0021 |
  | 10_calibration_thresholds | 0.0021 |
  | **sum** | **0.1585** |
  - assemble() wall = 0.1302s; comparable stage-sum / assemble() wall = 0.884

- **gpu_encoder_cpu_rest**: skipped (no CUDA device visible (torch.cuda.is_available() is False on this host); a real GPU-encoder config additionally needs a downloadable sentence-transformers model, which this offline harness does not fetch)
- **gpu_encoder_gpu_fusion**: skipped (no CUDA device visible (torch.cuda.is_available() is False on this host); a real GPU-encoder config additionally needs a downloadable sentence-transformers model, which this offline harness does not fetch)
- OOF run (n_folds=3): whole_run=1.286s, per_fold=0.429s

## n_items~100000, n_classes=500
- **cpu_only** (n_query=8848, n_candidates=246958):

  | stage | seconds |
  |---|---|
  | 1_encode | 0.3727 |
  | 2_prototypes_and_freq | 0.0417 |
  | 3_description_prototype_similarity | 0.0024 |
  | 4_dense_topk | 3.3439 |
  | 5_bm25 | 4.1835 |
  | 6_scatter_knn | 0.0297 |
  | 7_ranks_margins_topn_minmax | 0.2131 |
  | 8_dataframe_construction | 0.0297 |
  | 9_fusion_predict | 0.0146 |
  | 10_calibration_thresholds | 0.0128 |
  | **sum** | **8.2442** |
  - assemble() wall = 7.8614s; comparable stage-sum / assemble() wall = 0.998

- **gpu_encoder_cpu_rest**: skipped (no CUDA device visible (torch.cuda.is_available() is False on this host); a real GPU-encoder config additionally needs a downloadable sentence-transformers model, which this offline harness does not fetch)
- **gpu_encoder_gpu_fusion**: skipped (no CUDA device visible (torch.cuda.is_available() is False on this host); a real GPU-encoder config additionally needs a downloadable sentence-transformers model, which this offline harness does not fetch)
- OOF run (n_folds=3): whole_run=55.717s, per_fold=18.572s

## n_items~10000, n_classes=5000
- **cpu_only** (n_query=4000, n_candidates=199251):

  | stage | seconds |
  |---|---|
  | 1_encode | 0.1887 |
  | 2_prototypes_and_freq | 0.1698 |
  | 3_description_prototype_similarity | 0.0240 |
  | 4_dense_topk | 0.6971 |
  | 5_bm25 | 0.7899 |
  | 6_scatter_knn | 0.0937 |
  | 7_ranks_margins_topn_minmax | 1.0470 |
  | 8_dataframe_construction | 0.0231 |
  | 9_fusion_predict | 0.0118 |
  | 10_calibration_thresholds | 0.0104 |
  | **sum** | **3.0554** |
  - assemble() wall = 3.1653s; comparable stage-sum / assemble() wall = 0.899

- **gpu_encoder_cpu_rest**: skipped (no CUDA device visible (torch.cuda.is_available() is False on this host); a real GPU-encoder config additionally needs a downloadable sentence-transformers model, which this offline harness does not fetch)
- **gpu_encoder_gpu_fusion**: skipped (no CUDA device visible (torch.cuda.is_available() is False on this host); a real GPU-encoder config additionally needs a downloadable sentence-transformers model, which this offline harness does not fetch)
- OOF run (n_folds=3): whole_run=29.579s, per_fold=9.860s

## n_items~100000, n_classes=5000
- **cpu_only** (n_query=9142, n_candidates=434232):

  | stage | seconds |
  |---|---|
  | 1_encode | 0.3954 |
  | 2_prototypes_and_freq | 0.3603 |
  | 3_description_prototype_similarity | 0.0542 |
  | 4_dense_topk | 3.6560 |
  | 5_bm25 | 4.2004 |
  | 6_scatter_knn | 0.3780 |
  | 7_ranks_margins_topn_minmax | 2.3554 |
  | 8_dataframe_construction | 0.0529 |
  | 9_fusion_predict | 0.0235 |
  | 10_calibration_thresholds | 0.0186 |
  | **sum** | **11.4946** |
  - assemble() wall = 11.7432s; comparable stage-sum / assemble() wall = 0.942

- **gpu_encoder_cpu_rest**: skipped (no CUDA device visible (torch.cuda.is_available() is False on this host); a real GPU-encoder config additionally needs a downloadable sentence-transformers model, which this offline harness does not fetch)
- **gpu_encoder_gpu_fusion**: skipped (no CUDA device visible (torch.cuda.is_available() is False on this host); a real GPU-encoder config additionally needs a downloadable sentence-transformers model, which this offline harness does not fetch)
- OOF run (n_folds=3): whole_run=94.740s, per_fold=31.580s
