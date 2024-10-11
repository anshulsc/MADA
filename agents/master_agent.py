# agents/master_agent.py
from dspy import Module
from query.query_planner import QueryPlanner
from query.reranker import RerankModule
from utils.logger import logger

class  MasterAgent(Module):
    def __init__(self, document_agents, reranker, query_planner):
        super().__init__()
        self.document_agents = document_agents
        self.reranker = reranker
        self.query_planner = query_planner

    def process_query(self, query):
        agent_ids = list(self.document_agents.keys())
        selected_agents = self.query_planner.forward(
            query=query,
            agent_ids=", ".join(agent_ids)
        )

        logger.info(f"Selected agents for query '{query}': {selected_agents}")

        initial_scores = {
            agent_id: agent.evaluate(query)
            for agent_id, agent in self.document_agents.items()
            if agent_id in selected_agents
        }

        logger.debug(f"Initial Scores: {initial_scores}")

        results = {
            doc_id: self.reranker.forward(doc_id, query, score)
            for doc_id, score in initial_scores.items()
        }

        logger.debug(f"Reranked Results: {results}")

        if not results:
            logger.warning("No documents found after reranking.")
            return "No documents found."

        top_doc_id = max(results, key=results.get)
        logger.info(f"Top Document ID: {top_doc_id}")

        final_answer = self.document_agents[top_doc_id].answer_query(query)
        return final_answer
