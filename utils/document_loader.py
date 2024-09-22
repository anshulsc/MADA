
from utils.logger import logger
from unstructured.partition.auto import partition
from llama_index.core import Document

def load_documents(file_path):
    logger.info(f"Loading documents from {file_path}")
    try:
        elements = partition(filename=file_path)
        docs = [Document(text=str(el), metadata={"source": file_path}) for el in elements]
        print(docs[0].text)
        logger.info(f"Loaded {len(docs)} documents from {file_path}")
        return docs
    except Exception as e:
        logger.error(f"Failed to load documents from {file_path}: {e}")
        return []
