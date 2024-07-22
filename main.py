import logging
import re
import numpy as np
import dspy
from pathlib import Path 
import dotenv
dotenv.load_dotenv()

from llama_index.readers.file import UnstructuredReader
from llama_index.core import Document 
from llama_index.core import VectorStoreIndex


import os

from qdrant_client import QdrantClient,models

from dspy.retrieve.qdrant_rm import QdrantRM
from dspy.teleprompt import BootstrapFewShotWithRandomSearch


from unstructured.partition.auto import partition

from transformers import AutoTokenizer, AutoModel
import traceback
import torch


# Initialize tokenizer and model for encoding queries
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def encode_query(query):
      inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
      outputs = model(**inputs)

      return outputs.last_hidden_state.shape

# Set up logging to file with UTF-8 encoding to handle Unicode characters
import logging

qdrant_client = QdrantClient(path="data/qdrantssZ.db")

COLLECTION_NAME = "llama_index_docS"

# Check if the collection exists or needs to be created
try:
    collection_info = qdrant_client.get_collection(COLLECTION_NAME)
    logging.info(f"Collection '{COLLECTION_NAME}' already exists with configuration: {collection_info.config}")
except Exception as e:
    if "not found" in str(e):
        logging.info(f"Collection '{COLLECTION_NAME}' does not exist. Attempting to create without specifying vector size.")
        qdrant_client.create_collection(collection_name=COLLECTION_NAME, vectors_config= models.VectorParams(size=100, distance=models.Distance.COSINE))
        logging.info(f"Created collection '{COLLECTION_NAME}' with automatic vector sizing.")
    else:
        logging.error(f"An error occurred while accessing collection info: {str(e)}")

openai_api_key = os.getenv("OPENAI_API_KEY")
openai = dspy.OpenAI(model="gpt-4o-mini", api_key=openai_api_key)


qdrant_retriever_model = QdrantRM(COLLECTION_NAME, qdrant_client, k = 10)

dspy.settings.configure(lm=openai, rm=qdrant_retriever_model)

print("Ready to go!")


def load_documents(file_path):
    logging.info(f"Loading documents from {file_path}")
    elements = partition(filename=file_path)
    docs = [Document(text=str(el), metadata={"source": file_path}) for el in elements]
    logging.info(f"Loaded {len(docs)} documents from {file_path}")
    return docs



import uuid
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core.data_structs.data_structs import IndexDict

vector_store = VectorStoreIndex(
    embed_model=OpenAIEmbedding(api_base="https://api.openai.com/v1/",
        api_key=openai_api_key),
    store_nodes_override=True,
    index_struct=IndexDict(),
)



def add_documents_to_collection(documents, qdrant_client, collection_name, vector_store):
  
    nodes = [TextNode(text=doc.text) for doc in documents] 
    embedded_nodes = vector_store._get_node_with_embedding(nodes)

    points = []
    for node in embedded_nodes:
        if hasattr(node, "embedding") and node.embedding is not None:
            embedding = node.embedding
            if isinstance(embedding, list):
                vector = embedding

            else:
                vector = embedding.tolist()
            
            point = {"id": str(uuid.uuid4()), 
                     "payload": {"document": node.text},
                     "vector": vector,
                     }
            points.append(point)

        
        try:
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
            logging.info(f"Added {len(points)} documents to collection '{collection_name}'")
        except Exception as e:
            logging.error(f"Failed to add documents to collection '{collection_name}': {str(e)}")


# Example usage
# documents = load_documents("/Users/anshulsingh/Downloads/docs/CV24.pdf")
# qdrant_client = QdrantClient(host="localhost", port=6333)
# collection_name = "testing"
# add_documents_to_collection(documents, qdrant_client, collection_name, vector_store)




class RerankingSignature(dspy.Signature):
    document_id = dspy.InputField(desc="ID of the document")
    initial_score = dspy.InputField(desc="Initial score of the document")
    query = dspy.InputField(desc="User query for contextual relevance")
    features = dspy.InputField(desc="Features extracted for reranking")
    rereanked_score = dspy.OutputField(desc="Recomputed score after reranking")


class RerankModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=10)

    def forward(self, document_id, query, intial_score):
        context = self.retrieve(query).passages
        print(f"Initial Score Type: {type(intial_score)}")
        reranked_score = intial_score + len(context)
        return reranked_score
    
import numpy as np

def calculate_score(predicted_relevance, true_relevance, k=10):

    if len(predicted_relevance) == 0 or len(true_relevance) == 0:
        return 0.0
    
    sorted_indices = np.argsort(predicted_relevance)[::-1]


    dcg = 0.0 
    for i in range(min(k, len(sorted_indices))):
        idx = sorted_indices[i]
        rel = true_relevance[idx]
        dcg += (2 ** rel - 1) / np.log2(i + 2)
        

    ideal_relevance = sorted(true_relevance, reverse=True)
    idcg = 0.0
    for i in range(min(k, len(ideal_relevance))):
        rel = ideal_relevance[i]
        idcg += (2 ** rel - 1) / np.log2(i + 2)

    
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return ndcg


class RerankingOptimizer(dspy.Module):
    def __init__(self, rerank_module):
        super().__init__()
        self.rerank_module = rerank_module
        self.lm = dspy.settings.lm
        self.teleprompter = BootstrapFewShotWithRandomSearch(
            metric=self.custom_metric,
            teacher_settings={'lm': self.lm},  # Use the explicitly passed LM
            max_bootstrapped_demos=2,  # Reduce the number of bootstrapped demos
            max_labeled_demos=8,  # Reduce the number of labeled demos
            num_candidate_programs=4,  # Reduce the number of candidate programs
            num_threads=4
        )
    
    def custom_metric(self, predictions, labels, extra_arg=None):
        logging.debug(f"custom_metric called with predictions: {predictions}, labels: {labels}")
        if len(predictions) == 0 or len(labels) == 0:
            logging.warning("No predictions or labels provided for custom_metric")
            return 0
        
        predicted_scores = []
        true_scores = []

        for pred in predictions:
            try:
                score = float(pred.split('reranked_score:')[1].split()[0])
                predicted_scores.append(score)
            except:
                logging.warning(f"Error extractiing predicted score from: {pred}")
                pass

        for label in labels:
            try:
                score = float(label.split('reranked_score:')[1].split()[0])
                true_scores.append(score)
            except (IndexError, ValueError):
                logging.warning(f"Error extracting true score from: {label}")
                pass

        if len(predicted_scores) == 0 or len(true_scores) == 0:
            logging.warning("No valid scores extracted for custom_metric")
            return 0
        
        if len(predicted_scores) != len(true_scores):
            logging.warning("Predicted and true scores have different lengths")
            return 0
        
        logging.debug(f"Predicted scores: {predicted_scores}")
        logging.debug(f"True scores: {true_scores}")

        squared_errors = [(pred_score - true_score) ** 2 for pred_score, true_score in zip(predicted_scores, true_scores)]

        if len(squared_errors) == 0:
            logging.warning("No squared errors calculated for custom_metric")
            return 0
        
        logging.debug(f"Squared errors: {squared_errors}")

        mse = np.mean(squared_errors)
        logging.debug(f"MSE: {mse}")
        
        return mse
    

    def optimize_reranking(self, document_ids, initial_scores, query):
        logging.debug(f"optimize_reranking called with document_ids: {document_ids}, initial_scores: {initial_scores}, query: {query}")
        if len(document_ids) == 0 or len(initial_scores) == 0:
            logging.error("Empty training set.")
            return None

        def trainset_generator():
            logging.debug("trainset_generator called")
            for i, (doc_id, score) in enumerate(zip(document_ids, initial_scores)):
                logging.debug(f"Generating example {i+1}/{len(document_ids)}")
                logging.debug(f"Document ID: {doc_id}")
                logging.debug(f"Initial Score: {score}")
                logging.debug(f"Query: {query}")
                example = dspy.Example(
                    document_id=doc_id,
                    initial_score=score,
                    query=query
                ).with_inputs("document_id", "initial_score", "query")
                logging.debug(f"Generated example: {example}")
                yield example

        try:
            print("Starting optimization...")
            optimized_program = self.teleprompter.compile(
                student=self.rerank_module,
                trainset=trainset_generator()
            )
            print("Optimization completed.")
            return optimized_program
        except ZeroDivisionError as e:
            logging.error(f"Division by zero error during optimization: {str(e)}")
            # Add additional debugging or error handling code here
            return None
        except Exception as e:
            logging.error(f"Failed to optimize reranking: {str(e)}")
            # Add additional debugging or error handling code here
            return None


class QueryPlanningSignature(dspy.Signature):
    query = dspy.InputField(desc="User query")
    agent_ids = dspy.InputField(desc="Available agent IDs")
    historical_data = dspy.InputField(desc="Historical performance data")
    selected_agents = dspy.OutputField(desc="Agents selected for the query")


class QueryPlanner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.process_query = dspy.ChainOfThought(QueryPlanningSignature)

    def forward(self, query, agent_ids, historical_data=None):
        context = f"Query: {query}\nAgents: {agent_ids}\nHistorical Data: {historical_data if historical_data else 'No historical data'}"
        prediction = self.process_query(query=query, agent_ids=agent_ids, historical_data=historical_data)
        return prediction.selected_agents if hasattr(prediction, 'selected_agents') else []
    



class DocumentAgent(dspy.Module):
    def __init__(self, document_id, content, qdrant_client, collection_name):
        super().__init__()
        self.document_id = document_id
        self.content = content
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.lm = dspy.settings.lm  # Assuming Claude is configured globally


    def request(self, prompt):

        try:
            response = self.lm(prompt)
            
            if isinstance(response, str):
                return response
            elif isinstance(response, list):
                return " ".join(response)
            elif isinstance(response, dict):
                
                if 'response' in response:
                    return response['response']
                else:
                    logging.warning(f" response key not found in response: {response}")

            else:
                logging.warning(f"Unexpected response format: {response}")

        except Exception as e:
            logging.error(f"Failed to generate response: {str(e)}")
        
        return None
    
    def encode_query(self, query):
        inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)

        return outputs.last_hidden_state.mean(dim=1).detach().numpy()
    
    def fetch_updated_data(self, query):
        """ Fetches updated or additional data relevant to the query from Qdrant. """

        try:
            batch_results = self.qdrant_client.query_batch(
                self.collection_name,
                query_texts = [query],
                limit=3
            )
            logging.debug(f"Batch results: {batch_results}")
            additional_data = " ".join([result.payload['document'] for batch in batch_results for result in batch['results']])
        except Exception as e:
            logging.error(f"Failed to fetch additional data: {str(e)}")
            additional_data = ""

        return additional_data
    

    def evaluate(self, query):
        """Evaluates the query by fetching data based on the query context and returns a score. """

        if "update" in query.lower():
            updated_content = self.fetch_updated_data(query)
            content_to_use = f"{self.content}\n{updated_content}"
        else:
            content_to_use = self.content

        logging.debug(f"Content to use: {content_to_use}")

        prompt = f"Evaluate the following content based on the query: {query}\nContent: {content_to_use}"   
        logging.debug(f"Prompt: {prompt}")

        try:
            response = self.request(prompt)
            logging.debug(f"Raw API response: {response}")

            if isinstance(response, str):
                if "does not directly answer" in response.lower() or "not relevant" in response.lower():
                    score = 0.0
                elif "provides some information" in response.lower() or "partially relevant" in response.lower():
                    score = 0.5
                else:
                    score = 1.0

        except Exception as e:
            logging.error(f"Error during Anthroptic API request: {str(e)}")
            score = 0.0

        logging.debug(f"Score: {score}")
        return score
    

    def answer_query(self, query):
        """ Use the evaluate method to process the query and fetch the final answer from the LM"""

        sub_queries = self.break_down_query(query)
        
        sub_answers = []
        cited_documents = []

        for sub_query in sub_queries:
            score = self.evaluate(sub_query)
            logging.debug(f"Score for sub-query '{sub_query}': {score}")

            if score > 0:

                relevant_parts  = self.extract_answer(sub_query)

                sub_answer = self.generate_answer(sub_query, relevant_parts)
                sub_answers.append(sub_answer)

                cited_documents.append(self.document_id)

        combined_answer = " ".join(sub_answers)

        refined_answer = self.refine_answer(query, combined_answer)

        cited_docs_str = ", ".join(f"Document {doc_id}" for doc_id in cited_documents)

        final_answer = f"{refined_answer}\n\nCited Documents: {cited_docs_str}"

        return final_answer
    

    def break_down_query(self, query):
        """ Breaks down the query into sub-queries for evaluation. """

        sub_queries = []

        sub_queries = re.split(r"\b(and|or|additionally)\b", query, flags=re.IGNORECASE)
        sub_queries = [q.strip() for q in sub_queries if q.strip()]
        
        return sub_queries
    

    def generate_answer(self, query, relevant_parts):
        """ Generates an answer using the language model based on the query and relevant parts """
        prompt = f"Query: {query}\nRelevant information: {' '.join(relevant_parts)}\nAnswer:"
        response = self.request(prompt)
        
        if response:
            return response.strip()
        else:
            return "I don't have enough information to answer this query."
        
    def refine_answer(self, query, answer):
        """ Refines the generated answer using the language model """
        prompt = f"Query: {query}\nGenerated answer: {answer}\nRefined answer:"
        response = self.request(prompt)
        
        if response:
            return response.strip()
        else:
            return answer
        

    def extract_answer(self, query):
        """ Extracts the relevant information from the document content to construct an answer """
        # Preprocess the query and content
        processed_query = self.preprocess_text(query)
        processed_content = self.preprocess_text(self.content)

        # Perform relevance scoring or information extraction techniques
        # to identify the most relevant parts of the content
        relevant_parts = self.find_relevant_parts(processed_query, processed_content)

        # Construct the answer based on the relevant parts
        answer = self.construct_answer(relevant_parts)

        return answer
    
    def preprocess_text(self, text):
        """ Preprocesses the text by lowercasing, removing punctuation, etc. """
        # Implement text preprocessing steps here
        processed_text = text.lower()
        # Add more preprocessing steps as needed
        return processed_text

    def find_relevant_parts(self, query, content):
        """ Finds the most relevant parts of the content based on the query """
        # Convert the content into sentences
        sentences = self.split_into_sentences(content)
        
        # Calculate the similarity between the query and each sentence
        similarities = []
        for sentence in sentences:
            similarity = self.calculate_similarity(query, sentence)
            similarities.append(similarity)
        
        # Sort the sentences based on their similarity scores
        sorted_sentences = [x for _, x in sorted(zip(similarities, sentences), reverse=True)]
        
        # Return the top N most relevant sentences
        top_n = 3  # Adjust the number of relevant sentences to return
        relevant_parts = sorted_sentences[:top_n]
        
        return relevant_parts

    def split_into_sentences(self, text):
        """ Splits the text into sentences """
        # You can use a library like NLTK or spaCy for more accurate sentence splitting
        # For simplicity, we'll use a basic approach here
        sentences = text.split(". ")
        return sentences
    

    def calculate_similarity(self, query, sentence):
        """ Calculates the similarity between the query and a sentence """
        # You can use more advanced similarity metrics like cosine similarity or TF-IDF
        # For simplicity, we'll use the Jaccard similarity here
        query_words = set(query.split())
        sentence_words = set(sentence.split())
        intersection = query_words.intersection(sentence_words)
        union = query_words.union(sentence_words)
        similarity = len(intersection) / len(union)
        return similarity

    def construct_answer(self, relevant_parts):
        """ Constructs the answer based on the relevant parts """
        # Join the relevant parts into a coherent answer
        answer = " ".join(relevant_parts)
        
        # Perform any necessary post-processing or formatting
        answer = answer.capitalize()
        
        return answer


class MasterAgent(dspy.Module):
    def __init__(self, document_agents, reranker, query_planner):
        super().__init__()
        self.document_agents = document_agents
        self.reranker = reranker
        self.query_planner = query_planner

    def process_query(self, query):
        # Use the query planner to determine which agents to involve in the query process
        selected_agents = self.query_planner.forward(query, ", ".join(list(self.document_agents.keys())))
        
        # Print the selected agents
        selected_agents_str = ", ".join([f"Document {agent_id}" for agent_id in selected_agents])
        logging.info(f"Selected agents for query '{query}': {selected_agents_str}")

        # Evaluate the query using the selected agents, generating initial scores
        initial_scores = {agent_id: agent.evaluate(query) for agent_id, agent in self.document_agents.items() if agent_id in selected_agents}

        # Rerank the results based on the initial scores
        results = {doc_id: self.reranker.forward(doc_id, query, score) for doc_id, score in initial_scores.items()}

        # Handle cases where no valid results are found
        if not results:
            return "No documents found."

        # Identify the top document based on the reranked scores and get the final answer
        top_doc_id = max(results, key=results.get)
        final_answer = self.document_agents[top_doc_id].answer_query(query)
        
        return final_answer
    


if __name__ == "__main__":
    logging.basicConfig(filename='app.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
    logging.info("Starting the document processing application.")

    try:
        file_path = "/Users/anshulsingh/Downloads/docs/CV24.pdf"
        documents = load_documents(file_path)
       

        
        if not documents:
            logging.error("No documents found. Exiting.")
            exit()

        logging.info(f"Loaded documents: {[doc.metadata['source'] for doc in documents]}")
        add_documents_to_collection(documents, qdrant_client, COLLECTION_NAME, vector_store)

        # Update DocumentAgent initialization to include qdrant_client and COLLECTION_NAME
        document_agents = {str(idx): DocumentAgent(document_id=idx, content=doc.text, qdrant_client=qdrant_client, collection_name=COLLECTION_NAME) for idx, doc in enumerate(documents)}
        logging.info(f"Created {len(document_agents)} document agents.")

        reranker = RerankModule()
        optimizer = RerankingOptimizer(reranker)
        query_planner = QueryPlanner()
        master_agent = MasterAgent(document_agents, reranker, query_planner)

        query = "Tell me about my project"
        logging.info(f"Processing query: {query}")
        
        response = master_agent.process_query(query) 
        print(response) # Directly process the query without optimization
        logging.info(f"Response: {response}")

    except Exception as e:
        logging.error(f"An error occurred during application execution: {str(e)}")
        logging.error(traceback.format_exc())  # Provides a stack trace