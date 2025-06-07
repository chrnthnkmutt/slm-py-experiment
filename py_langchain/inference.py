from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, SystemMessage

# Initialize the LLM with your local Ollama server
llm = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required but unused
    model="qwenCPU:latest",
    temperature=0
)

# Create a simple tool for the agent
def search_tool(query: str) -> str:
    """A simple search tool that provides information about the 2020 World Series."""
    if "2020" in query.lower() and "world series" in query.lower():
        if "where" in query.lower() or "played" in query.lower():
            return "The 2020 World Series was played at Globe Life Field in Arlington, Texas (neutral site due to COVID-19)."
        else:
            return "The Los Angeles Dodgers won the 2020 World Series, defeating the Tampa Bay Rays 4 games to 2."
    return "I don't have information about that query."

# Create the tool
tools = [
    Tool(
        name="search",
        description="Search for information about sports events",
        func=search_tool
    )
]

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can search for information when needed."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Create the agent
agent = create_openai_functions_agent(llm, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run the agent with your question
response = agent_executor.invoke({
    "input": "Where was the 2020 World Series played?",
    "chat_history": []
})

print(response["output"])
