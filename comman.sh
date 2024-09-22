docker run --rm \
  --name document_processing_app \
  --network my_network \
  -v $(pwd)/docs:/app/docs \
  -v $(pwd)/logs:/app/logs \
  -e OPENAI_API_KEY=OPENAI_API_KEY\
  -e COLLECTION_NAME=llama_index_docs \
  -e EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2 \
  -e OPENAI_MODEL=text-davinci-003 \
  -e QDRANT_HOST=qdrant \
  -e QDRANT_PORT=6333 \
  -e VECTOR_SIZE=384 \
  -e VECTOR_DISTANCE=COSINE\
  document_processing_app \
  python main.py --query "V-Probing" --docs_path "app/docs/PHYSICSOFLLM.pdf"
