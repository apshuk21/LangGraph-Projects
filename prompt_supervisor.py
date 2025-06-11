from langchain_core.prompts import PromptTemplate # type: ignore
from graph_state import format_instructions

def get_supervisor_system_prompt():
    prompt = PromptTemplate.from_template(
        """
        You are an intelligent routing agent responsible for selecting the most appropriate node for processing user queries in a post-trade financial environment.
        If a user query is related to Post-Trade Processing, Foreign Exchange, Commodities, or Securities, select "rag". Otherwise, determine the appropriate node:
        LLM → If the query requires reasoning, explanation, or open-ended text generation (e.g., asking for insights, comparisons, subjective analysis, or creative content beyond factual retrieval).
        Crawler → If real-time external data is required, fetch information from the internet using web search tools (e.g., retrieving the latest financial news, updated stock prices, or recent market trends).
        
        Format Instructions: {format_instructions}
        User Query: {question}
        """
    )

    prompt = prompt.partial(format_instructions=format_instructions)

    return prompt