from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse


import os
import uuid
import base64
from IPython import display
from unstructured.partition.pdf import partition_pdf
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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


# Global System Configuration for OboTutor
OBOTUTOR_SYSTEM_CONFIG = {
    "name": "OboTutor",
    "role": "AI Tutor Assistant",
    "core_mission": "Help students learn effectively and achieve academic success",
    "personality_traits": [
        "Patient and encouraging",
        "Knowledgeable and adaptive", 
        "Supportive and professional",
        "Clear and educational"
    ],
    "key_capabilities": [
        "Document analysis (PDFs, images, tables)",
        "Step-by-step explanations",
        "Homework and project assistance", 
        "Exam preparation support",
        "Adaptive teaching methods"
    ],
    "teaching_principles": [
        "Break down complex topics into manageable parts",
        "Use examples and analogies for clarity",
        "Encourage active learning and critical thinking",
        "Provide context and real-world connections",
        "Maintain supportive tone throughout"
    ]
}




from fastapi.middleware.cors import CORSMiddleware
from typing import List
import tempfile
import sqlite3
from datetime import datetime




#get functions main.py


app = FastAPI()




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict the allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # This allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # This allows all headers
)


os.environ['GOOGLE_API_KEY']=os.getenv("GOOGLE_API_KEY")

text_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5,
    max_tokens=1500,
    google_api_key=os.getenv("GOOGLE_API_KEY")
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
        img_path_list.append("https://obotutor-2.s3.eu-north-1.amazonaws.com/" + unique_filename)

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
    prompt_template = """You are OboTutor, an AI tutor that helps students learn effectively. 

PERSONALITY:
- Be conversational and natural - no formal greetings like "Hello there!"
- Jump straight into answering questions without mentioning "documents" or "context"
- Be encouraging but not overly enthusiastic
- Keep explanations clear and easy to follow

RESPONSE STYLE:
- Answer questions directly without preambles
- Don't mention "based on the documents" or "from the context provided"
- Use "you" and "your" to make it personal
- Include examples and analogies when they help explain concepts
- Break down complex ideas into simple steps

GUIDELINES:
- Only use information from the provided context
- If you can't answer from the context, say: "I don't have enough information about that topic. Could you share more details or upload relevant materials?"
- Ask follow-up questions when they would help the student learn
- Suggest practical applications when relevant

CONTEXT: {context}

QUESTION: {question}

Answer the question naturally and conversationally:"""
    
    prompt = PromptTemplate.from_template(prompt_template)
    qa_chain = prompt | text_model | StrOutputParser()
    
    return qa_chain

def create_conversation_chain(text_model):
    """Create a chain for general conversation"""
    conversation_template = """You are OboTutor, a friendly AI tutor who helps students with their learning.

PERSONALITY:
- Be casual and conversational - no formal greetings 
- Jump right into helping without lengthy introductions
- Be encouraging and supportive
- Keep things simple and clear

CONVERSATION STYLE:
- Respond naturally like you're chatting with a student
- When students greet you, respond warmly and ask how you can help
- For questions, give direct helpful answers
- If students seem stuck, offer encouragement and different approaches
- Ask follow-up questions to better understand their needs

WHAT YOU CAN HELP WITH:
- Analyzing documents and materials they upload
- Explaining concepts step by step
- Helping with homework and projects
- Exam preparation
- Study strategies

TONE:
- Friendly and approachable
- Patient and understanding
- Positive and motivating
- Clear and helpful

Student: {question}

Response:"""
    prompt = PromptTemplate.from_template(conversation_template)
    conversation_chain = prompt | text_model | StrOutputParser()
    
    return conversation_chain

def is_conversational_input(user_input):
    """Detect if the input is conversational rather than document-related"""
    conversational_patterns = [
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
        'how are you', 'whats up', "what's up", 'thanks', 'thank you', 'bye', 
        'goodbye', 'see you', 'nice to meet you', 'how do you do',
        'can you help me', 'help', 'who are you', 'what can you do',
        'what are you', 'introduce yourself'
    ]
    
    user_lower = user_input.lower().strip()
    
    # Check for exact matches or if the input starts with conversational patterns
    for pattern in conversational_patterns:
        if user_lower == pattern or user_lower.startswith(pattern):
            return True
    
    # Check if it's a very short input (likely conversational)
    if len(user_input.split()) <= 3 and any(word in user_lower for word in ['hi', 'hello', 'hey', 'thanks', 'help']):
        return True
        
    return False

def should_search_documents(user_input):
    """Determine if the input warrants searching documents"""
    document_keywords = [
        'document', 'pdf', 'content', 'about', 'explain', 'describe', 'summary',
        'what is', 'tell me about', 'information', 'details', 'specification',
        'requirement', 'feature', 'technical', 'system', 'project', 'proposal'
    ]
    
    user_lower = user_input.lower()
    return any(keyword in user_lower for keyword in document_keywords)



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
    result = chain.invoke({'context': context, 'question': query})
    print(relevant_images , "relevant images")
    return result, relevant_images



def save_document_to_db(filename, file_path):
    """Save document information to database"""
    conn = sqlite3.connect("./obotutor.db")
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO documents (filename, file_path, processed)
        VALUES (?, ?, ?)
    ''', (filename, file_path, True))
    
    document_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return document_id

def save_chat_to_db(question, answer, document_id=None):
    """Save chat interaction to database"""
    conn = sqlite3.connect("./obotutor.db")
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO chat_history (question, answer, document_id)
        VALUES (?, ?, ?)
    ''', (question, answer, document_id))
    
    conn.commit()
    conn.close()

@app.post("/model/upload/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    file_path = os.getenv("SAVE_PDF_DIR")
    os.makedirs(file_path, exist_ok=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.getenv("OUTPUT_DIR"), exist_ok=True)
    os.makedirs(os.getenv("VECTOR_DB_DIR"), exist_ok=True)

    uploaded_files_info = []

    try:
        for file in files:
            new_file_path = os.path.join(file_path, f"copy_{file.filename}")

            # Read the file content
            pdf_content = await file.read()

            with open(new_file_path, "wb") as f:
                f.write(pdf_content)

            # Save to database
            document_id = save_document_to_db(file.filename, new_file_path)
            
            documents = []
            retrieve_contents = []
            vectorstore = None

            text = get_pdf_text(new_file_path)
            print("completed reading pdf")
            text_chunks = get_text_chunks(text)
            print("completed splitting text")
            create_documents_text(text_chunks,documents)
            print("completed creating documents")
            
            # Comment out image processing if you don't have S3 bucket
            # img_elements, table_elements = extract_images_and_tables_from_unstructured_pdf(file,os.getenv("OUTPUT_DIR"))
            # print("completed extracting images and tables")
            # vision_model = text_model
            # print("completed vision model")
            # if img_elements:
            #     image_summaries,img_base64_list, img_path_list = get_summary_of_images(img_elements,os.getenv("OUTPUT_DIR"),vision_model)
            #     print("completed image summaries")
            #     create_documents_images(img_base64_list,image_summaries,documents,retrieve_contents, img_path_list)
            #     print("completed creating documents images")
                
            # Use HuggingFace embeddings instead of Gemini to avoid quota issues
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            print("completed embedding model")
            vectorstore = create_vector_store(documents, embedding_model, os.getenv("VECTOR_DB_DIR"))
            print("completed vector store")
            uploaded_files_info.append(file.filename)
            print("completed uploading files info")

        return {"message": "Files uploaded successfully", "files": uploaded_files_info}

    except Exception as e:
        return {"message": f"An error occurred: {e}", "files": uploaded_files_info}

@app.post("/model/query/")
async def query_document(request: dict):
    try:
        # Extract parameters from request
        question = request.get("question", "")
        chat_history = request.get("chat_history", "")
        chatbox_id = request.get("chatbox_id")
        
        if not question:
            return {"error": "Question is required"}
        
        # If we have chat history, include it in the context
        enhanced_question = question
        if chat_history:
            enhanced_question = f"""Previous conversation:
{chat_history}

Current question: {question}

Please consider the conversation context when answering. If the question refers to something mentioned earlier (like "the project", "it", "this"), use that context to provide a relevant answer."""
        # Check if it's a conversational input
        if is_conversational_input(enhanced_question):
            conversation_chain = create_conversation_chain(text_model)
            result = conversation_chain.invoke({'question': enhanced_question})
            
            # Save chat to database
            save_chat_to_db(question, result)
            
            return {
                "result": result,
                "relevant_images": [],
                "type": "conversation",
                "context_used": len(chat_history) > 0
            }
        
        # Use HuggingFace embeddings instead of Gemini to avoid quota issues
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Check if it's document-specific or should search documents
        if should_search_documents(enhanced_question):
            vectorstore = load_vector_store(os.getenv("VECTOR_DB_DIR"), embedding_model)
            
            # Check if vectorstore has documents
            if vectorstore._collection.count() > 0:
                chain = create_chain(text_model)
                result, relevant_images = retrieve_content(enhanced_question, chain, vectorstore)
                
                # Save chat to database
                save_chat_to_db(question, result)
                
                return {
                    "result": result,
                    "relevant_images": relevant_images,
                    "type": "document-based",
                    "context_used": len(chat_history) > 0
                }
            else:
                # No documents available
                conversation_chain = create_conversation_chain(text_model)
                result = conversation_chain.invoke({'question': enhanced_question}) + "\n\nTo get document-specific answers, please upload some PDFs first."
                
                save_chat_to_db(question, result)
                
                return {
                    "result": result,
                    "relevant_images": [],
                    "type": "conversation",
                    "context_used": len(chat_history) > 0
                }
        else:
            # General conversational response
            conversation_chain = create_conversation_chain(text_model)
            result = conversation_chain.invoke({'question': enhanced_question})
            
            save_chat_to_db(question, result)
            
            return {
                "result": result,
                "relevant_images": [],
                "type": "conversation",
                "context_used": len(chat_history) > 0
            }
            
    except Exception as e:
        return {"error": f"An error occurred: {e}"}
