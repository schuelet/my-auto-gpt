from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

from langchain.agents.load_tools import get_all_tool_names
from langchain.agents import load_tools, initialize_agent, AgentType

get_all_tool_names()

llm = OpenAI()

tools = load_tools(["wikipedia","llm-math"], llm=llm)

agent = initialize_agent(tools=tools, llm=llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

result = agent.run("in what year was the python 3 released. Multiply that year with 3")

# check if the answer is correct
from langchain.chains.constitutional_ai.base import ConstitutionalChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple

ethical_principle = ConstitutionalPrinciple(
    name="Math Checker",
    critique_request="The model should give the mathematically correct answer",
    revision_request="Rewrite the model's output to be both factual and mathematically correct",
)

constitutional_chain = ConstitutionalChain.from_llm(
    chain=agent,
    constitutional_principles=[ethical_principle],
    llm=llm,
    verbose=True,
)

constitutional_chain.run(question="in what year was the python 3 released. Multiply that year with 3")