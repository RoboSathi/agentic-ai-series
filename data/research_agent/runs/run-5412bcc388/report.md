# Vector Database Selection for 10 Million Documents

**Status:** PARTIAL

## Findings

Cohere provides 10 million 768-dimensional vectors [E1]. Pinecone: 45-80ms (serverless), 20-40ms (pod-based) [E2]; Pinecone demonstrates the lowest latency, with a p95 latency of 40-50 milliseconds [E6] Pinecone supports 5,000-10,000 QPS [E7]. Pinecone pricing starts at $0.096/hr per pod (~$70/month) and can reach $200-400/month for larger deployments [E8]. Weaviate: 30-70ms (with hybrid search adding 10-20ms) [E3]; In contrast, Weaviate shows a slightly higher latency range of 50-70 milliseconds [E9] Milvus query latency p95 is 25-50ms depending on index type and configuration [E4]; Zilliz Cloud (Milvus) costs $200-400 [E5]. If dataset does not fit in memory and speed is not critical, SQLite can be used as storage [E10]; otherwise Weaviate, Qdrant and Elasticsearch are good options [E11].

## Details

Pinecone demonstrates the lowest latency, with a p95 latency of 40-50 milliseconds [E6]; Pinecone supports 5,000-10,000 QPS [E7] Its cost starts around $70/month for 10M vectors and can rise to $200-400/month for larger usage [E8]. Weaviate provides hybrid search capabilities but exhibits higher latency (30-70ms base, plus 10-20ms for hybrid) [E3][E9]. Milvus/Zilliz Cloud delivers latency in the 25-50ms range with cost estimated $200-400 [E4][E5]. When memory constraints or speed are not priorities, SQLite can serve as a simple storage fallback [E10]; otherwise Weaviate, Qdrant, and Elasticsearch remain viable alternatives [E11].

## Known unknowns

- None declared during scoping.

## Coverage & Limits

- Every must-pass criterion was met.
- 2 source(s) rejected — The model's answer could not be used. e.g. https://www.firecrawl.dev/blog/best-vector-databases, https://eastondev.com/blog/en/posts/ai/20260427-rag-vector-database-selection/
- Budget: Q1: fetches exhausted.
- Budget: Q2: fetches exhausted.

## Citations

- [E1] https://zvec.org/en/docs/db/benchmarks — retrieved 2026-10-02T10:52:57.666274+00:00 — page sha256 075df39e69649854… — “Cohere 10M: 10 million 768-dimensional vectors”
- [E2] https://reintech.io/blog/vector-database-comparison-2026-pinecone-weaviate-milvus-qdrant-chroma — retrieved 2026-10-02T10:52:55.366961+00:00 — page sha256 3476b58240dd5804… — “Pinecone: 45-80ms (serverless), 20-40ms (pod-based)”
- [E6] https://dasroot.net/posts/2026/03/vector-databases-comparison-pinecone-weaviate-milvus/ — retrieved 2026-10-02T10:53:29.427055+00:00 — page sha256 c720d162530fa953… — “Pinecone demonstrates the lowest latency, with a p95 latency of 40-50 milliseconds, making it particularly well-suited for real-time applications where responsiveness is crucial.”
- [E7] https://dasroot.net/posts/2026/03/vector-databases-comparison-pinecone-weaviate-milvus/ — retrieved 2026-10-02T10:53:29.427055+00:00 — page sha256 c720d162530fa953… — “Pinecone supports 5,000-10,000 QPS, making it efficient for moderate-scale applications.”
- [E8] https://dasroot.net/posts/2026/03/vector-databases-comparison-pinecone-weaviate-milvus/ — retrieved 2026-10-02T10:53:29.427055+00:00 — page sha256 c720d162530fa953… — “Pricing for Pinecone in 2026 starts at $0.096/hr per pod, which equates to approximately $70/month for 10M vectors. For larger deployments, costs can range from $200–$400/month, depending on the number of pods and usage.”
- [E3] https://reintech.io/blog/vector-database-comparison-2026-pinecone-weaviate-milvus-qdrant-chroma — retrieved 2026-10-02T10:52:55.366961+00:00 — page sha256 3476b58240dd5804… — “Weaviate: 30-70ms (with hybrid search adding 10-20ms)”
- [E9] https://dasroot.net/posts/2026/03/vector-databases-comparison-pinecone-weaviate-milvus/ — retrieved 2026-10-02T10:53:29.427055+00:00 — page sha256 c720d162530fa953… — “In contrast, Weaviate shows a slightly higher latency range of 50-70 milliseconds, which may still be acceptable for many applications but could introduce minor delays in highly interactive systems.”
- [E4] https://reintech.io/blog/vector-database-comparison-2026-pinecone-weaviate-milvus-qdrant-chroma — retrieved 2026-10-02T10:52:55.366961+00:00 — page sha256 3476b58240dd5804… — “Milvus: 25-50ms (depends on index type and configuration)”
- [E5] https://reintech.io/blog/vector-database-comparison-2026-pinecone-weaviate-milvus-qdrant-chroma — retrieved 2026-10-02T10:52:55.366961+00:00 — page sha256 3476b58240dd5804… — “Zilliz Cloud (Milvus): $200-400”
- [E10] https://jina.ai/news/benchmark-vector-search-databases-with-one-million-data — retrieved 2026-10-02T10:56:06.476764+00:00 — page sha256 bbade8c29ddbb424… — “Finally, if your dataset does not fit in memory, and you do not care much about the speed of nearest neighbor search, you can use SQLite as storage.”
- [E11] https://jina.ai/news/benchmark-vector-search-databases-with-one-million-data — retrieved 2026-10-02T10:56:06.476764+00:00 — page sha256 bbade8c29ddbb424… — “Otherwise, Weaviate, Qdrant and Elasticsearch are good options.”
