```mermaid
graph LR
A[Master Agent] --> B[Query Planner]
B --> C[Document Agent 1]
B --> D[Document Agent 2]
B --> E[Document Agent 3]
C --> F[Reranking Module]
D --> F
E --> F
F --> G[Final Answer]
```
