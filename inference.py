from dotenv import load_dotenv
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from openai import Client

from prompt_template import prompt_template, prompt_template_langchain

load_dotenv()

client = Client(
    api_key=os.getenv("API_KEY")
)


def inference_all(reviews):
    reviews = "\n".join([f"{review['id']}. {review['document']}" for review in reviews])
    prompt = prompt_template.format(reviews=reviews)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    output = response.choices[0].message.content
    return output


prompt = PromptTemplate(
    template=prompt_template_langchain,
    input_variables=["reviews"],
)

model = ChatOpenAI(
    temperature=0.0,
    openai_api_key=os.getenv("API_KEY"),
    model_name="gpt-4o-mini"
)

output_parser =  StrOutputParser()
chain = (prompt | model | output_parser)

def inference_all_langchain(reviews):
    reviews = "\n".join([f"{review['id']}. {review['document']}" for review in reviews])
    output = chain.invoke({"reviews": reviews})
    return output

if __name__ == '__main__':
    print(inference_all_langchain([
        {"id": 1, "document": "이 영화가 정말 재밌어요"},
        {"id": 2, "document": "이 영화가 정말 재미없어요"},
        {"id": 3, "document": "이 영화가 정말 재밌어요"},
        {"id": 4, "document": "이 영화가 정말 재미없어요"},
        {"id": 5, "document": "이 영화가 정말 재밌어요"}
    ]))
