# main.py
import argparse
import traceback
from qdrant_client import QdrantClient, models
from transformers import AutoTokenizer, AutoModel
from config.config import (
    COLLECTION_NAME,
    OPENAI_API_KEY,
    EMBEDDING_MODEL_NAME,
    OPENAI_MODEL,
)
from utils.logger import logger
from utils.document_loader import load_documents
from index.vector_store import (
    initialize_vector_store,
    add_documents_to_collection,
    connect_qdrant,
    create_collection_if_not_exists
)
from agents.document_agents import DocumentAgent
from agents.master_agent import MasterAgent
from query.reranker import RerankModule, RerankingOptimizer
from query.query_planner import QueryPlanner
import dspy
from dspy.retrieve.qdrant_rm import QdrantRM

def parse_arguments():
    parser = argparse.ArgumentParser(description="Document Processing Application")
    parser.add_argument(
        "--docs_path",
        type=str,
        required=True,
        help="Path to the documents (PDF) to be processed."
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query string to process."
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    docs_path = args.docs_path
    query = args.query

    logger.info("Starting the document processing application.")
    try:
        # Initialize Qdrant client
        qdrant_client = connect_qdrant()
    
        # Check or create collection
        create_collection_if_not_exists(qdrant_client, COLLECTION_NAME)
    
        # Initialize tokenizer and model for encoding queries
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
        logger.info("Initialized tokenizer and embedding model.")
    
        # Initialize vector store
        vector_store = initialize_vector_store()
    
        # Load documents
        logger.info(f"Loading documents from {docs_path}")
        documents = load_documents(docs_path)
        if not documents:
            logger.error("No documents found. Exiting.")
            exit()
        logger.info(f"Loaded {len(documents)} documents from: {[doc.metadata.get('source', 'Unknown') for doc in documents]}")
    
        # Add documents to Qdrant
        add_documents_to_collection(documents, qdrant_client, COLLECTION_NAME, vector_store)

        # Configure dspy settings
        openai_api_key = OPENAI_API_KEY
        openai = dspy.OpenAI(model=OPENAI_MODEL, api_key=openai_api_key)
        qdrant_retriever_model = QdrantRM(COLLECTION_NAME, qdrant_client, k=10)
        dspy.settings.configure(lm=openai, rm=qdrant_retriever_model)
    
        # Initialize DocumentAgents
        document_agents = {
            str(idx): DocumentAgent(
                document_id=str(idx),
                content=doc.text,
                qdrant_client=qdrant_client,
                collection_name=COLLECTION_NAME,
                tokenizer=tokenizer,
                model=model
            )
            for idx, doc in enumerate(documents)
        }
        logger.info(f"Created {len(document_agents)} document agents.")
    
        logger.info("Configured dspy settings with OpenAI and Qdrant retriever.")
    
        # Initialize Reranker and Optimizer
        reranker = RerankModule()
    
        # Initialize Query Planner
        query_planner = QueryPlanner()
    
        # Initialize Master Agent
        master_agent = MasterAgent(document_agents, reranker, query_planner)
    
        # Process the query
        logger.info(f"Processing query: {query}")
        response = master_agent.process_query(query)
        print(response)
        logger.info(f"Response: {response}")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
