from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader   # load pdf files
from langchain.text_splitter import RecursiveCharacterTextSplitter              # create chunks of data
from langchain_huggingface import HuggingFaceEmbeddings                         # create embeddings
from langchain_community.vectorstores import FAISS                              # create vector database

# step 1: load the data (raw PDF files)
DATA_PATH = "data/"   # path where all the pdf files are stored

def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_files(data=DATA_PATH)
print("length of pdf pages :", len(documents))

# step 2: split the data into chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)
print("length of text chunks :", len(text_chunks))

# step 3: create embeddings of the chunks
def create_embeddings():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model 

embedding_model = create_embeddings()   # ✅ fixed function call

# step 4 : store embeddings in FAISS vector database
DB_FAISS_PATH = "vectorstore/db_faiss"

db = FAISS.from_documents(text_chunks, embedding_model)   # ✅ fixed typo
db.save_local(DB_FAISS_PATH)
