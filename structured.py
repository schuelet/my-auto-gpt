from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import pandas as pd
import json

hat_model = ChatOpenAI(temperature=0, max_tokens=1000)

# How you would like your reponse structured. This is basically a fancy prompt template
response_schemas = [
    ResponseSchema(name="input_industry", description="This is the input_industry from the user"),
    ResponseSchema(name="standarized_industry", description="This is the industry you feel is most closely matched to the users input"),
    ResponseSchema(name="match_score",  description="A score 0-100 of how close you think the match is between user input and your match")
]

# How you would like to parse your output
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)


# See the prompt template you created for formatting
format_instructions = output_parser.get_format_instructions()
print (output_parser.get_format_instructions())


template = """
You will be given a series of industry names from a user.
Find the best corresponding match on the list of standardized names.
The closest match will be the one with the closest semantic meaning. Not just string similarity.

{format_instructions}

Wrap your final output with closed and open brackets (a list of json objects)

input_industry INPUT:
{user_industries}

STANDARDIZED INDUSTRIES:
{standardized_industries}

YOUR RESPONSE:
"""

prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template(template)  
    ],
    input_variables=["user_industries", "standardized_industries"],
    partial_variables={"format_instructions": format_instructions}
)

# Get your standardized names. You can swap this out with whatever list you want!

standardized_industries = ", ".join([])
standardized_industries

# Your user input

user_input = "air LineZ, airline, aviation, planes that fly, farming, bread, wifi networks, twitter media agency"

_input = prompt.format_prompt(user_industries=user_input, standardized_industries=standardized_industries)


print (f"There are {len(_input.messages)} message(s)")
print (f"Type: {type(_input.messages[0])}")
print ("---------------------------")
print (_input.messages[0].content)

output = chat_model(_input.to_messages())


if "```json" in output.content:
    json_string = output.content.split("```json")[1].strip()
else:
    json_string = output.content

print(output.content)