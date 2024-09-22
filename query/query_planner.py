# modules/query_planner.py
from dspy import Module, Signature, InputField, OutputField, ChainOfThought
from utils.logger import logger

class QueryPlanningSignature(Signature):
    query = InputField(desc="User query")
    agent_ids = InputField(desc="Available agent IDs")
    historical_data = InputField(desc="Historical performance data")
    selected_agents = OutputField(desc="Agents selected for the query")

class QueryPlanner(Module):
    def __init__(self):
        super().__init__()
        self.process_query = ChainOfThought(QueryPlanningSignature)
    
    def forward(self, query, agent_ids, historical_data=None):
        context = f"Query: {query}\nAgents: {agent_ids}\nHistorical Data: {historical_data if historical_data else 'No historical data'}"
        prediction = self.process_query(query=query, agent_ids=agent_ids, historical_data=historical_data)
        selected_agents = prediction.selected_agents if hasattr(prediction, 'selected_agents') else []
        logger.info(f"Selected agents for query '{query}': {selected_agents}")
        return selected_agents
