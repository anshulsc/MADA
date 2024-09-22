# modules/reranker.py
import numpy as np
from dspy.retrieve.qdrant_rm import QdrantRM
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy import Module, Signature, InputField, OutputField, settings
from utils.logger import logger
import dspy as dspy

class RerankingSignature(Signature):
    document_id = InputField(desc="ID of the document")
    initial_score = InputField(desc="Initial score of the document")
    query = InputField(desc="User query for contextual relevance")
    features = InputField(desc="Features extracted for reranking")
    reranked_score = OutputField(desc="Recomputed score after reranking")

class RerankModule(Module):
    def __init__(self):
        super().__init__()
        self.retrieve = settings.rm  # Retrieve module set in settings

    def forward(self, document_id, query, initial_score):
        context = self.retrieve(query)
        print(f"Context: {context}")
        logger.debug(f"Initial Score Type: {type(initial_score)}")
        reranked_score = initial_score + len(context)
        logger.debug(f"Reranked Score for Document {document_id}: {reranked_score}")
        return reranked_score

def calculate_score(predicted_relevance, true_relevance, k=10):
    if not predicted_relevance or not true_relevance:
        return 0.0
    
    sorted_indices = np.argsort(predicted_relevance)[::-1]
    dcg = sum((2 ** true_relevance[idx] - 1) / np.log2(i + 2) for i, idx in enumerate(sorted_indices[:k]))
    
    ideal_relevance = sorted(true_relevance, reverse=True)
    idcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_relevance[:k]))
    
    return dcg / idcg if idcg > 0 else 0.0

class RerankingOptimizer(Module):
    def __init__(self, rerank_module):
        super().__init__()
        self.rerank_module = rerank_module
        self.lm = settings.lm  # Language model set in settings
        self.teleprompter = BootstrapFewShotWithRandomSearch(
            metric=self.custom_metric,
            teacher_settings={'lm': self.lm},
            max_bootstrapped_demos=2,
            max_labeled_demos=8,
            num_candidate_programs=4,
            num_threads=4
        )
    
    def custom_metric(self, predictions, labels, extra_arg=None):
        logger.debug(f"custom_metric called with predictions: {predictions}, labels: {labels}")
        if not predictions or not labels:
            logger.warning("No predictions or labels provided for custom_metric")
            return 0
        
        try:
            predicted_scores = [float(pred.split('reranked_score:')[1].split()[0]) for pred in predictions]
            true_scores = [float(label.split('reranked_score:')[1].split()[0]) for label in labels]
        except (IndexError, ValueError) as e:
            logger.warning(f"Error extracting scores: {e}")
            return 0
        
        if len(predicted_scores) != len(true_scores):
            logger.warning("Predicted and true scores have different lengths")
            return 0
        
        mse = np.mean([(p - t) ** 2 for p, t in zip(predicted_scores, true_scores)])
        logger.debug(f"MSE: {mse}")
        return mse
    
    def optimize_reranking(self, document_ids, initial_scores, query):
        logger.debug(f"optimize_reranking called with document_ids: {document_ids}, initial_scores: {initial_scores}, query: {query}")
        if not document_ids or not initial_scores:
            logger.error("Empty training set.")
            return None

        def trainset_generator():
            for doc_id, score in zip(document_ids, initial_scores):
                example = dspy.Example(
                    document_id=doc_id,
                    initial_score=score,
                    query=query
                ).with_inputs("document_id", "initial_score", "query")
                yield example

        try:
            logger.info("Starting optimization...")
            optimized_program = self.teleprompter.compile(
                student=self.rerank_module,
                trainset=trainset_generator()
            )
            logger.info("Optimization completed.")
            return optimized_program
        except ZeroDivisionError as e:
            logger.error(f"Division by zero error during optimization: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to optimize reranking: {e}")
            return None
