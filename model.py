import os
import base64
import uuid
import io
import re
from unstructured.partition.pdf import partition_pdf
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
# from langchain_openai import OpenAIEmbeddings
from IPython.display import HTML, display
from PIL import Image
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from fastapi import FastAPI,Request, Form, Response, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")



database_path = 'chroma_db/'



def load_model(model_name):
  if model_name=="gemini-pro":
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key= GOOGLE_API_KEY)
  else:
    llm=ChatGoogleGenerativeAI(model="gemini-pro-vision" , google_api_key= GOOGLE_API_KEY)

  return llm

#create load vector store
def load_vector_store(directory, embedding_model):
    vectorstore = Chroma(
        embedding_function=embedding_model,
        persist_directory=directory
    )
    return vectorstore


# text_model = load_model("gemini-pro")
text_model =  ChatOpenAI(
    model="gpt-4o",
    temperature=0
)


embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

from langchain_openai import OpenAIEmbeddings
# embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vector_db = load_vector_store(database_path , embedding_model)


app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)


# prompt_template = """You are a tutor assistant. You aims to provide personalized instruction, guided problem-solving, and adaptive teaching to cater to each student's unique needs and learning pace.
# Answer the question based only on the following context, which can include text, images and tables.
# You help everyone by answering questions, and improve your answers from previous answers in History.
# Answer in the same language the question was asked.
# Answer in a way that is easy to understand.
# Do not say "Based on the information you provided, ..." or "I think the answer is...". Just answer the question directly in detail.

# History: {chat_history}

# Context: {context}

# Question: {question}


# Don't answer if you are not sure and decline to answer and say "Sorry, I don't have much information about it."
# Just return the helpful answer in as much as detailed possible.
# Answer:
# """


topics = "Programming and Electronics"
age = "15"
learning_style = "Visual"
mentored_notes = "Shows good understanding of basic concepts but struggles with applying them in new contexts."
example_scenario = " Imagine you're navigating through a maze and you have a sensor that can detect walls around you. How would you use this information to move forward without hitting any walls?"
desired_format = "Visual explanation with a simple diagram.Don't use images"
appropriate_tone = "Encouraging and friendly."



prompt_template = """

[Context]
Curriculum: RoboticGen Academy, Notes Content: {context}.

[Task]
Task: {question}.

"""




history_summarize_prompt_template = """You are an assistant tasked with summarizing text for retrieval.
Summarize the human question and AI answer in a concise manner.It should be a brief summary of the conversation.

human question: {human_question}
AI answer: {ai_answer}

Summary:
"""

history_summarize_chain = LLMChain(llm=text_model,
                                  prompt=PromptTemplate.from_template(history_summarize_prompt_template))

qa_chain = LLMChain(llm=text_model,
                    prompt=PromptTemplate.from_template(prompt_template))


@app.get("/")
def root():
    return {"Hello": "World"}


chat_summary = []

def get_string_from_chat_summary(chat_summary):
    chat_string = ""
    for chat in chat_summary:
        chat_string += chat + '\n'
    return chat_string



@app.post("/get_answer")
async def get_answer(question: str = Form(...)):
    relevant_docs = vector_db.similarity_search(question)
    chat_history = get_string_from_chat_summary(chat_summary)

    context = ""
    relevant_images = []
    for d in relevant_docs:
        if d.metadata['type'] == 'text':
            context += '[text]' + d.metadata['original_content']
        elif d.metadata['type'] == 'table':
            context += '[table]' + d.metadata['original_content']
        elif d.metadata['type'] == 'image':
            context += '[image]' + d.page_content
            relevant_images.append(d.metadata['original_content'])
    result = qa_chain.run({'context': context, 'question': question , 'chat_history': chat_history, 'topics': topics, 'age': age, 'learning_style': learning_style, 'mentored_notes': mentored_notes, 'example_scenario': example_scenario, 'desired_format': desired_format, 'appropriate_tone': appropriate_tone})
    new_chat_summary = history_summarize_chain.run({'human_question': question, 'ai_answer': result})
    chat_summary.append(new_chat_summary)
    return JSONResponse({"relevant_images": relevant_images[0], "result": result , "chat_history": chat_summary})

    
    

    return JSONResponse({"relevant_images": relevant_images[0], "result": result , "chat_history": chat_history})


@app.post("/get_answer/gpt")
async def getAnswerGpt(question : str):
    relevant_docs = vector_db.similarity_search(question)
    context = ""
    relevant_images = []
    for d in relevant_docs:
        if d.metadata['type'] == 'text':
            context += '[text]' + d.metadata['original_content']
        elif d.metadata['type'] == 'table':
            context += '[table]' + d.metadata['original_content']
        elif d.metadata['type'] == 'image':
            context += '[image]' + d.page_content
            relevant_images.append(d.metadata['original_content'])
    result = qa_chain.run({'context': context, 'question': question })
    return result


