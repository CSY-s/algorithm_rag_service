# RAG Experiment Report

- time: 2026-04-08T15:41:00.019369
- samples: 6
- algorithm_id: None

## Config Comparison

| retrieval_mode | top_k | rouge_1 | rouge_2 | rouge_l | bleu_4 | done | failed |
|---|---:|---:|---:|---:|---:|---:|---:|
| tfidf | 3 | 0.5071 | 0.4033 | 0.4686 | 0.2717 | 6 | 0 |
| tfidf | 5 | 0.5150 | 0.4011 | 0.4613 | 0.2756 | 6 | 0 |
| tfidf | 8 | 0.5331 | 0.4089 | 0.4771 | 0.2924 | 6 | 0 |
| keyword | 3 | 0.4391 | 0.3032 | 0.3614 | 0.2029 | 6 | 0 |
| keyword | 5 | 0.4150 | 0.2756 | 0.3389 | 0.1705 | 6 | 0 |
| keyword | 8 | 0.4410 | 0.2991 | 0.3785 | 0.1961 | 6 | 0 |
| vector | 3 | 0.4891 | 0.3926 | 0.4513 | 0.2553 | 5 | 1 |
| vector | 5 | 0.5144 | 0.3994 | 0.4616 | 0.2711 | 6 | 0 |
| vector | 8 | 0.5328 | 0.4099 | 0.4765 | 0.2900 | 6 | 0 |
| hybrid | 3 | 0.4950 | 0.3888 | 0.4601 | 0.2567 | 6 | 0 |
| hybrid | 5 | 0.5075 | 0.3941 | 0.4609 | 0.2636 | 6 | 0 |
| hybrid | 8 | 0.5212 | 0.3972 | 0.4615 | 0.2681 | 6 | 0 |

## Notes

- `avg_methods` is the mean of `baseline_extractive`, `rag_basic`, and `rag_plus`.
- Use the full JSON output for per-method and per-case details.