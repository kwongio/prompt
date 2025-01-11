import json
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from openai import Client
from pydantic import BaseModel

from prompt_template import prompt_template_langchain

load_dotenv()

client = Client(
    api_key=os.getenv("API_KEY")
)


class Output(BaseModel):
    summary: str

output_parser = PydanticOutputParser(pydantic_object=Output)
prompt_maker = PromptTemplate(
    template=prompt_template_langchain,
    input_variables=["review"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)

model = ChatOpenAI(
    temperature=0.0,
    openai_api_key=os.getenv("API_KEY"),
    model_name="gpt-4o-mini",
)

chain = (prompt_maker | model | output_parser)


def inference_langchain(reviews):
    reviews = "\n".join(reviews)
    output = chain.invoke({"reviews": reviews})
    return output.summary


if __name__ == '__main__':
    print(inference_langchain(
        [
            "보는 내내 시간 가는줄 모르고 정말 재밌게 봤습니다~",
            "정말 쓰레기같은 영화... 다신 안본다",
            "이런 영화가 있을 수가... 정말 재미없었어요"
        ]
    ))