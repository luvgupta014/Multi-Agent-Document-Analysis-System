# Evaluation Results

Generated: 2026-11-27 18:55:06  
Questions: 30  
Run time: 0.7s

## Retrieval Metrics (baseline — vector search only)

| Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---|
| 0.2333 | 0.5 | 0.7667 | 0.8667 | 0.4392 |

## Ragas Metrics (LLM-judged, n=0)

> Ragas skipped: disabled via --no-ragas


## Per-Question Detail

| # | Question | MRR | R@1 | R@3 | R@5 |
|---|---|---|---|---|---|
| 1 | What has been the main driver of climate change since the 18 | 1.0 | 1.0 | 1.0 | 1.0 |
| 2 | By how much has CO2 concentration increased since pre-indust | 0.5 | 0.0 | 1.0 | 1.0 |
| 3 | What temperature limit does scientific consensus say we must | 1.0 | 1.0 | 1.0 | 1.0 |
| 4 | Name three major impacts of climate change. | 0.3333 | 0.0 | 1.0 | 1.0 |
| 5 | What individual actions help reduce climate change? | 1.0 | 1.0 | 1.0 | 1.0 |
| 6 | What international agreement coordinates global climate acti | 1.0 | 1.0 | 1.0 | 1.0 |
| 7 | How do solar panels generate electricity? | 1.0 | 1.0 | 1.0 | 1.0 |
| 8 | What is the main challenge with solar and wind energy? | 0.2 | 0.0 | 0.0 | 1.0 |
| 9 | What technology does hydroelectric power use? | 1.0 | 1.0 | 1.0 | 1.0 |
| 10 | Where is geothermal energy most viable? | 0.5 | 0.0 | 1.0 | 1.0 |
| 11 | What are three benefits of renewable energy? | 0.25 | 0.0 | 0.0 | 1.0 |
| 12 | What materials are used in biomass energy production? | 0.25 | 0.0 | 0.0 | 1.0 |
| 13 | What is the AUC-ROC of the churn prediction model in Q3? | 0.25 | 0.0 | 0.0 | 1.0 |
| 14 | What architecture does the churn model use? | 0.5 | 0.0 | 1.0 | 1.0 |
| 15 | How many customer records was the churn model trained on? | 0.5 | 0.0 | 1.0 | 1.0 |
| 16 | What is the document classifier's overall accuracy? | 0.25 | 0.0 | 0.0 | 1.0 |
| 17 | Which ticket categories have the lowest F1 score in document | 0.25 | 0.0 | 0.0 | 1.0 |
| 18 | What is the recommendation engine's click-through rate? | 0.25 | 0.0 | 0.0 | 1.0 |
| 19 | What is the total monthly cost for model serving infrastruct | 0.0 | 0.0 | 0.0 | 0.0 |
| 20 | What ELT pattern does the data pipeline follow? | 1.0 | 1.0 | 1.0 | 1.0 |
| 21 | How does the pipeline handle real-time event ingestion? | 0.5 | 0.0 | 1.0 | 1.0 |
| 22 | What are the three data storage zones? | 0.125 | 0.0 | 0.0 | 0.0 |
| 23 | What tool manages SQL transformations in the pipeline? | 0.1429 | 0.0 | 0.0 | 0.0 |
| 24 | What orchestration tool runs the pipeline DAGs? | 0.0 | 0.0 | 0.0 | 0.0 |
| 25 | How is data encrypted in transit in the pipeline? | 0.125 | 0.0 | 0.0 | 0.0 |
| 26 | What retrieval improvements are planned for the Q4 RAG pipel | 0.5 | 0.0 | 1.0 | 1.0 |
| 27 | What percentage improvement in retrieval accuracy does Q4 ex | 0.25 | 0.0 | 0.0 | 1.0 |
| 28 | What security improvements are planned for Q4? | 0.0 | 0.0 | 0.0 | 0.0 |
| 29 | How many engineers are allocated to the Q4 RAG pipeline work | 0.5 | 0.0 | 1.0 | 1.0 |
| 30 | What is the target test coverage improvement in Q4 technical | 0.0 | 0.0 | 0.0 | 0.0 |

## Notes

- Chunk IDs use format `<filename_stem>_chunk_<index>`
- Similarity metric: cosine (1 − distance)
- Embedding model: all-MiniLM-L6-v2 (384-dim)
- Phase 2 will add BM25 hybrid + cross-encoder re-ranking
