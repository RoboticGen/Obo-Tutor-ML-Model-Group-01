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
from langchain.chains import LLMChain
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



os.environ['GOOGLE_API_KEY']=os.getenv("GOOGLE_API_KEY")

text_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.5,
    max_tokens=1500,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)



def extract_images_and_tables_from_unstructured_pdf(pdf,output_path):

    save_directory = os.getenv("SAVE_PDF_DIR")
    os.makedirs(save_directory, exist_ok=True)

    if pdf is not None:
    # Define a new file name to save the uploaded PDF
        new_file_path = os.path.join(save_directory, f"copy_{pdf.name}")

        # Write the content of the uploaded PDF to a new PDF file
        with open(new_file_path, "wb") as f:
            f.write(pdf.getbuffer()) 




    # Load the PDF file
    raw_pdf_elements=partition_pdf(
    filename= f"{os.getenv('SAVE_PDF_DIR')}/copy_{pdf.name}",
                    
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

def create_conversation_chain(text_model):
    """Create a chain for general conversation"""
    conversation_template = """
    You are a friendly and helpful tutor assistant. You are designed to help students with their learning and provide educational support.
    
    Respond to the user's message in a friendly, encouraging, and educational manner. 
    If they greet you, greet them back and offer to help with their studies or document questions.
    If they ask general questions, provide helpful educational responses.
    
    User message: {question}
    
    Response:
    """
    conversation_chain = LLMChain(llm=text_model,
                                prompt=PromptTemplate.from_template(conversation_template))
    
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
    result = chain.run({'context': context, 'question': query})
    print(relevant_images , "relevant images")
    return result, relevant_images
    

def handle_userinput(user_question, chain, vectorstore, text_model):
    """Handle user input with both conversational and document-based responses"""
    
    # Check if it's a conversational input
    if is_conversational_input(user_question):
        # Handle with conversation chain
        conversation_chain = create_conversation_chain(text_model)
        result = conversation_chain.run({'question': user_question})
        
        st.write(bot_template.replace("{{MSG}}", result), unsafe_allow_html=True)
        
        # Suggest document-related help if vectorstore has content
        if vectorstore and vectorstore._collection.count() > 0:
            suggestion = "\n\n💡 You can also ask me questions about the documents you've uploaded!"
            st.write(bot_template.replace("{{MSG}}", suggestion), unsafe_allow_html=True)
        
        return
    
    # Check if we should search documents and if we have the necessary components
    if should_search_documents(user_question) and vectorstore and chain and vectorstore._collection.count() > 0:
        # Handle with document search
        result, relevant_images = retrieve_content(user_question, chain, vectorstore)
        
        st.write(bot_template.replace("{{MSG}}", result), unsafe_allow_html=True)
        
        # Display the relevant images using web URLs
        for img in relevant_images:
            display.Image(url=img)
    else:
        # If no documents available or not document-related, use conversational response
        conversation_chain = create_conversation_chain(text_model)
        
        if not vectorstore or vectorstore._collection.count() == 0:
            # No documents uploaded yet
            result = conversation_chain.run({'question': user_question}) + "\n\n📚 To get document-specific answers, please upload some PDFs using the sidebar and click 'Process'."
        else:
            # General response
            result = conversation_chain.run({'question': user_question})
        
        st.write(bot_template.replace("{{MSG}}", result), unsafe_allow_html=True)



def main():
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    
    documents = []
    retrieve_contents = []
    vectorstore = None
    
    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True , type=["pdf"])
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing"):
                    
                    for pdf_doc in pdf_docs:
                        st.write(f"Processing {pdf_doc.name}...")

                        text = get_pdf_text(pdf_doc)
                        print(f"Extracted text length: {len(text)}")
                        
                        text_chunks = get_text_chunks(text)
                        print(f"Created {len(text_chunks)} text chunks")
                        
                        create_documents_text(text_chunks,documents)
                        print(f"Total documents so far: {len(documents)}")
                        
                        # Comment out image processing for now to focus on text
                        # img_elements, table_elements = extract_images_and_tables_from_unstructured_pdf(pdf_doc,os.getenv("OUTPUT_DIR"))
                        # vision_model = text_model
                        # if img_elements:
                        #     image_summaries,img_base64_list, img_path_list = get_summary_of_images(img_elements,os.getenv("OUTPUT_DIR"),vision_model)
                        #     create_documents_images(img_base64_list,image_summaries,documents,retrieve_contents, img_path_list)
                            
                        # if table_elements:
                        #     table_summaries = get_summary_of_tables(table_elements,text_model)
                        #     create_documents_tables(table_elements,table_summaries,documents,retrieve_contents)
                    
                    if documents:
                        st.write(f"Creating vector store with {len(documents)} documents...")
                        vectorstore = create_vector_store(documents, embedding_model, os.getenv("VECTOR_DB_DIR"))
                        st.success("Documents processed successfully")
                    else:
                        st.error("No documents were created from the PDFs")
            else:
                st.warning("Please upload at least one PDF file")

    # Try to load existing vector store if no new processing happened
    if not vectorstore:
        try:
            vectorstore = load_vector_store(os.getenv("VECTOR_DB_DIR"), embedding_model)
            # Test if vector store has documents
            if vectorstore._collection.count() > 0:
                st.info(f"Loaded existing vector store with {vectorstore._collection.count()} documents")
            else:
                st.warning("Vector store is empty. Please upload and process some PDFs first.")
        except Exception as e:
            st.warning("No existing vector store found. Please upload and process some PDFs first.")
            print(f"Error loading vector store: {e}")

    # Handle user questions regardless of vectorstore status
    if user_question:
        if vectorstore and vectorstore._collection.count() > 0:
            chain = create_chain(text_model)
            handle_userinput(user_question, chain, vectorstore, text_model)
        else:
            # No documents available, but still handle conversational input
            handle_userinput(user_question, None, None, text_model)

                        

if __name__ == "__main__":
    main()

