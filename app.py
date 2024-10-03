from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse


import os
import uuid
import base64
from IPython import display
from unstructured.partition.pdf import partition_pdf
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

import streamlit as st
from htmlTemplates import css, bot_template, user_template


from IPython import display

import boto3
import botocore

from dotenv import load_dotenv
load_dotenv()


from io import BytesIO




from fastapi.middleware.cors import CORSMiddleware
from typing import List
import tempfile




#get functions main.py


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict the allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # This allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # This allows all headers
)


os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

text_model =  ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    max_tokens=1500,
)




def extract_images_and_tables_from_unstructured_pdf(pdf,output_path):
    # Load the PDF file
    raw_pdf_elements=partition_pdf(
    filename= f"{os.getenv('SAVE_PDF_DIR')}/copy_{pdf.filename}",
                    
    strategy="hi_res",                                
    extract_images_in_pdf=True,                      
    extract_image_block_types=["Image", "Table"],          
    extract_image_block_to_payload=False,                  
    extract_image_block_output_dir=output_path, 
    )
    
    img=[]
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Image" in str(type(element)):
            img.append(str(element))

    tab=[]
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tab.append(str(element))

    print(img , "images")
    print(tab , "tables")
    return img,tab


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')
    


def upload_to_s3(file_name, bucket, object_name=None):
    if object_name is None:
        object_name = file_name

    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except Exception as e:
        print(e)
        return False
   
    return True

def download_from_s3(bucket, object_name, file_name):
    s3 = boto3.client('s3')
    try:
        s3.download_file(bucket, object_name, file_name)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise



def summarize_image(encoded_image, vision_model):
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval. It should be included the most important information in the image. """

    msg = [
        HumanMessage(content=[
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                },
            },
        ])
    ]
    response = vision_model.invoke(msg)
    return response.content


def get_summary_of_images(image_elements,output_path, vision_model):
    image_summaries = []
    img_base64_list = []
    img_path_list = []
    for i in os.listdir(output_path):
        if i.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(output_path, i)
            encoded_image = encode_image(image_path)
            img_base64_list.append(encoded_image)
            summary = summarize_image(encoded_image, vision_model)
            image_summaries.append(summary)

        image_path = os.path.join(output_path, i)
        unique_filename = uuid.uuid4().hex + i
        upload_to_s3(image_path, os.getenv("BUCKET_NAME"), unique_filename)

        #remove the image from the local directory
        os.remove(image_path)
     
      
        #add unique id to image path
        img_path_list.append("https://obotutor.s3.eu-north-1.amazonaws.com/" + unique_filename)

    print(image_summaries , "image summaries")
    print(img_base64_list , "image base64 list")
    print(img_path_list , "image path")
    return image_summaries,img_base64_list , img_path_list


def get_summary_of_tables(table_elements,text_model):
    table_summaries = []

    # Prompt
    prompt_text = """You are an assistant tasked with summarizing tables for retrieval. \
    These summaries will be embedded and used to retrieve the raw table elements. \
    Give a concise summary of the table that is well optimized for retrieval. Table:{element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    summarize_chain = {"element": lambda x: x} | prompt | text_model | StrOutputParser()

    table_summaries = summarize_chain.batch(table_elements, {"max_concurrency": 5})

    return table_summaries

#create documents for tables summary
def create_documents_tables(table_elements,table_summaries,documents,retrieve_contents):
    i = str(uuid.uuid4())
    for e, s in zip(table_elements, table_summaries):
        doc = Document(
            page_content = s,
            metadata = {
                'id': i,
                'type': 'table',
                'original_content': e
            }
        )
        retrieve_contents.append((i, e))
        documents.append(doc)

def create_documents_images(img_base64_list,image_summaries,documents,retrieve_contents, img_path_list):
    i = str(uuid.uuid4())
    for e, s , p in zip(img_base64_list, image_summaries , img_path_list):
        doc = Document(
            page_content = s,
            metadata = {
                'id': i,
                'type': 'image',
                'original_content': p,
                'base64': e
            }
        )
        retrieve_contents.append((i, p))
        documents.append(doc)


def get_pdf_text(pdf):
    text = ""

    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
    


def create_documents_text(text_list,documents):
    i = str(uuid.uuid4())
    for s in text_list:
        doc = Document(
            page_content = s,
            metadata = {
                'id': i,
                'type': 'text',
                
            }
        )
        # print(doc)
        documents.append(doc)

#create vector store
def create_vector_store(documents, embedding_model, dbpath):
    vectorstore = Chroma.from_documents(documents=documents, embedding=embedding_model , persist_directory=dbpath)
    return vectorstore


def load_vector_store(directory, embedding_model):
    vectorstore = Chroma(
        embedding_function=embedding_model,
        persist_directory=directory
    )
    return vectorstore

#create chain
def create_chain(text_model):
    prompt_template = """
    You are a tutor assistant. You aims to provide personalized instruction, guided problem-solving, and adaptive teaching to cater to each student's unique needs and learning pace.
    Answer the question based only on the following context, which can include text, images and tables:
    {context}
    Question: {question}
    Don't answer if you have no context and decline to answer and say "Sorry, I don't have much information about it."
    Just return the helpful answer in as much as detailed possible.
    Answer:
    """
    qa_chain = LLMChain(llm=text_model,
                        prompt=PromptTemplate.from_template(prompt_template))
    
    return qa_chain



def retrieve_content(query,chain,vectorstore):
    relevant_docs = vectorstore.similarity_search(query)
    print(relevant_docs)
    context = ""
    relevant_images = []
    for d in relevant_docs:
        if d.metadata['type'] == 'text':
            context += '[text]' + d.page_content
        elif d.metadata['type'] == 'table':
            context += '[table]' + d.page_content
        elif d.metadata['type'] == 'image':
            context += '[image]' + d.page_content
            relevant_images.append(d.metadata['original_content'])
    result = chain.run({'context': context, 'question': query})
    print(relevant_images , "relevant images")
    return result, relevant_images






@app.post("/model/upload/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    file_path = os.getenv("SAVE_PDF_DIR")
    os.makedirs(file_path, exist_ok=True)

    uploaded_files_info = []

    try:
        for file in files:
            new_file_path = os.path.join(file_path, f"copy_{file.filename}")

            # Read the file content
            pdf_content = await file.read()



            with open(new_file_path, "wb") as f:
                f.write(pdf_content)

            
            documents = []
            retrieve_contents = []
            vectorstore = None

            text = get_pdf_text( f"{os.getenv('SAVE_PDF_DIR')}/copy_{file.filename}")
            print("completed reading pdf")
            text_chunks = get_text_chunks(text)
            print("completed splitting text")
            create_documents_text(text_chunks,documents)
            print("completed creating documents")
            
            
            img_elements, table_elements = extract_images_and_tables_from_unstructured_pdf(file,os.getenv("OUTPUT_DIR"))
            print("completed extracting images and tables")
            vision_model = text_model
            print("completed vision model")
            if img_elements:
                image_summaries,img_base64_list, img_path_list = get_summary_of_images(img_elements,os.getenv("OUTPUT_DIR"),vision_model)
                print("completed image summaries")
                create_documents_images(img_base64_list,image_summaries,documents,retrieve_contents, img_path_list)
                print("completed creating documents images")
                
            # if table_elements:
            #     table_summaries = get_summary_of_tables(table_elements,text_model)
            #     create_documents_tables(table_elements,table_summaries,documents,retrieve_contents)
            
                
            embedding_model =  OpenAIEmbeddings(model="text-embedding-3-small")
            print("completed embedding model")
            vectorstore = create_vector_store(documents, embedding_model, os.getenv("VECTOR_DB_DIR"))
            print("completed vector store")
            uploaded_files_info.append(
              file.filename
                
            )
            print("completed uploading files info")
          


        return {"message": "Files uploaded successfully", "files": uploaded_files_info}

    except Exception as e:
        return {"message": f"An error occurred: {e}" , "files": uploaded_files_info}

  

# @app.post("/process_pdf/")
# async def process_pdf(filename: str):
#     pdf_path = f"/tmp/{filename}"
    
#     # Extract text and other contents
#     pdf_reader = PdfReader(pdf_path)
#     text = "".join([page.extract_text() for page in pdf_reader.pages])
    
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     text_chunks = text_splitter.split_text(text)
    
#     documents = [Document(page_content=chunk) for chunk in text_chunks]
    
#     embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
#     vectorstore = Chroma.from_documents(documents=documents, embedding=embedding_model)
    
#     return JSONResponse({"status": "Processed successfully"})

# @app.post("/query/")
# async def query_document(question: str):
#     # Vector store retrieval logic
#     query_result = vectorstore.similarity_search(question)
#     chain = create_chain(text_model)
#     result = chain.run({'context': query_result, 'question': question})
    
#     return {"result": result}

# Helper functions like create_chain and others can be added as needed.
