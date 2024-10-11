# agents/document_agent.py
import re
from dspy import Module, settings
import torch
import numpy as np
from utils.logger import logger

class DocumentAgent(Module):
    def __init__(self, document_id, content, qdrant_client, collection_name, tokenizer, model):
        super().__init__()
        self.document_id = document_id
        self.content = content
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.tokenizer = tokenizer
        self.model = model
        self.lm = settings.lm
         # Language model set in settings

    def request(self, prompt):
        try:
            response = self.lm(prompt)
            if isinstance(response, str):
                return response
            elif isinstance(response, list):
                return " ".join(response)
            elif isinstance(response, dict):
                return response.get('response', None)
            else:
                logger.warning(f"Unexpected response format: {response}")
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
        return None

    def encode_query(self, query):
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    def fetch_updated_data(self, query):
        """ Fetches updated or additional data relevant to the query from Qdrant. """
        try:
            batch_results = self.qdrant_client.query_batch(
                self.collection_name,
                query_texts=[query],
                limit=3
            )
            additional_data = " ".join(
                [result.payload['document'] for batch in batch_results for result in batch['results']]
            )
            logger.debug(f"Fetched additional data: {additional_data}")
        except Exception as e:
            logger.error(f"Failed to fetch additional data: {e}")
            additional_data = ""
        return additional_data

    def evaluate(self, query):
        """Evaluates the query by fetching data based on the query context and returns a score."""
        if "update" in query.lower():
            updated_content = self.fetch_updated_data(query)
            content_to_use = f"{self.content}\n{updated_content}"
        else:
            content_to_use = self.content

        logger.debug(f"Content to use: {content_to_use}")
        prompt = f"Evaluate the following content based on the query: {query}\nContent: {content_to_use}"   
        logger.debug(f"Prompt: {prompt}")

        try:
            response = self.request(prompt)
            logger.debug(f"Raw API response: {response}")
            if isinstance(response, str):
                if "does not directly answer" in response.lower() or "not relevant" in response.lower():
                    score = 0.0
                elif "provides some information" in response.lower() or "partially relevant" in response.lower():
                    score = 0.5
                else:
                    score = 1.0
            else:
                score = 0.0
        except Exception as e:
            logger.error(f"Error during API request: {e}")
            score = 0.0

        logger.debug(f"Score: {score}")
        return score

    def answer_query(self, query):
        """ Use the evaluate method to process the query and fetch the final answer from the LM"""
        sub_queries = self.break_down_query(query)
        sub_answers = []
        cited_documents = []

        for sub_query in sub_queries:
            score = self.evaluate(sub_query)
            logger.debug(f"Score for sub-query '{sub_query}': {score}")
            if score > 0:
                relevant_parts = self.extract_answer(sub_query)
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
        sub_queries = re.split(r"\b(and|or|additionally)\b", query, flags=re.IGNORECASE)
        return [q.strip() for q in sub_queries if q.strip()]

    def generate_answer(self, query, relevant_parts):
        """ Generates an answer using the language model based on the query and relevant parts """
        prompt = f"Query: {query}\nRelevant information: {' '.join(relevant_parts)}\nAnswer:"
        response = self.request(prompt)
        return response.strip() if response else "I don't have enough information to answer this query."

    def refine_answer(self, query, answer):
        """ Refines the generated answer using the language model """
        prompt = f"Query: {query}\nGenerated answer: {answer}\nRefined answer:"
        response = self.request(prompt)
        return response.strip() if response else answer

    def extract_answer(self, query):
        """ Extracts the relevant information from the document content to construct an answer """
        processed_query = self.preprocess_text(query)
        processed_content = self.preprocess_text(self.content)
        relevant_parts = self.find_relevant_parts(processed_query, processed_content)
        return self.construct_answer(relevant_parts)

    def preprocess_text(self, text):
        """ Preprocesses the text by lowercasing, removing punctuation, etc. """
        return text.lower()

    def find_relevant_parts(self, query, content):
        """ Finds the most relevant parts of the content based on the query """
        sentences = self.split_into_sentences(content)
        similarities = [self.calculate_similarity(query, sentence) for sentence in sentences]
        sorted_sentences = [x for _, x in sorted(zip(similarities, sentences), reverse=True)]
        return sorted_sentences[:3]

    def split_into_sentences(self, text):
        """ Splits the text into sentences """
        return text.split(". ")

    def calculate_similarity(self, query, sentence):
        """ Calculates the similarity between the query and a sentence """
        query_words = set(query.split())
        sentence_words = set(sentence.split())
        intersection = query_words.intersection(sentence_words)
        union = query_words.union(sentence_words)
        return len(intersection) / len(union) if union else 0

    def construct_answer(self, relevant_parts):
        """ Constructs the answer based on the relevant parts """
        answer = " ".join(relevant_parts).capitalize()
        return answer
