import os
import uuid
import base64
from IPython import display
from unstructured.partition.pdf import partition_pdf
# from langchain.chat_models import ChatOpenAI
# from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from IPython import display

from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")



def load_model(model_name):
  if model_name=="gemini-pro":
    llm = ChatGoogleGenerativeAI(model="gemini-pro" , google_api_key= GOOGLE_API_KEY)
  else:
    llm=ChatGoogleGenerativeAI(model="gemini-pro-vision" ,  google_api_key= GOOGLE_API_KEY)

  return llm



def extract_images_and_tables_from_unstructured_pdf(pdf_path,output_path):
    # Load the PDF file
    raw_pdf_elements=partition_pdf(
    filename=pdf_path,                  
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

    return img,tab


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def summarize_image(encoded_image, vision_model):
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval."""

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
    for i in os.listdir(output_path):
        if i.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(output_path, i)
            encoded_image = encode_image(image_path)
            img_base64_list.append(encoded_image)
            summary = summarize_image(encoded_image, vision_model)
            image_summaries.append(summary)

    print(image_summaries , "image summaries")
    print(img_base64_list , "image base64 list")
    return image_summaries,img_base64_list

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

def create_documents_images(img_base64_list,image_summaries,documents,retrieve_contents):
    i = str(uuid.uuid4())
    for e, s in zip(img_base64_list, image_summaries):
        doc = Document(
            page_content = s,
            metadata = {
                'id': i,
                'type': 'image',
                'original_content': e
            }
        )
        retrieve_contents.append((i, e))
        documents.append(doc)




def get_documents_text(path_name):
        
    loader = PyPDFLoader(path_name)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    return docs


def create_documents_text(text_documents,documents):
    i = str(uuid.uuid4())
    for s in text_documents:
        doc = Document(
            page_content = s.page_content,
            metadata = {
                'id': i,
                'type': 'text',
                'original_content': s.metadata['source']
            }
        )
        # print(doc)
        documents.append(doc)



#create vector store
def create_vector_store(documents, embedding_model, dbpath):
    vectorstore = Chroma.from_documents(documents=documents, embedding=embedding_model , persist_directory=dbpath)
    return vectorstore



#create load vector store
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
            context += '[table]' + d.metadata['original_content']
        elif d.metadata['type'] == 'image':
            context += '[image]' + d.page_content
            relevant_images.append(d.metadata['original_content'])
    result = chain.run({'context': context, 'question': query})
    print(relevant_images , "relevant images")
    return result, relevant_images
    


if __name__ == '__main__':
    pathname1 = "text_arduino.pdf"
    pathname2 = "intro_arduino.pdf"
    text_model = load_model("gemini-pro")
    vision_model = load_model("gemini-pro-vision")
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    fpath="./extracted_data3/"
    database_path = 'docs/chroma/'


    #check model
    # print(text_model.invoke("please come up with the best funny line.").content)

    query1 = "What is arduino?"
    query3 = "What is devops?"
    query4 = "Explain Relationship between DevOps features and software quality"
    query5 = "Can you explain about arduino board?"


    #load db and ask question
    vector_store = load_vector_store(database_path , embedding_model)
    chain = create_chain(text_model)
    result, relevant_images = retrieve_content(query4,chain,vector_store)
    print(result)

    
    


    
    # call text pdf documents
    # text_documents = get_documents_text(pathname1)
    # documents = []
    # retrieve_contents = []
    # create_documents_text(text_documents,documents)
    # vector_store = create_vector_store(documents, embedding_model,database_path)
    # chain = create_chain(text_model)
    # result, relevant_images = retrieve_content(query1,chain,vector_store)
    # print(result)
    # print(relevant_images)

    # call image and table pdf documents
    # image_elements, table_elements = extract_images_and_tables_from_unstructured_pdf(pathname2,fpath)
    # image_summaries, images_list  = get_summary_of_images(image_elements,fpath,vision_model)
    # table_summaries = get_summary_of_tables(table_elements,text_model)
    # documents = []
    # retrieve_contents = []
    # create_documents_images(images_list,image_summaries,documents,retrieve_contents)
    # create_documents_tables(table_elements,table_summaries,documents,retrieve_contents)
    # vector_store = create_vector_store(documents, embedding_model,database_path)
    # chain = create_chain(text_model)
    # result, relevant_images = retrieve_content(query4,chain,vector_store)
    # print(result)











    





    
