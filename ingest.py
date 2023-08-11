from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS

DATA_PATH =  'data/'
DB_FAISS_PATH = 'vectorstores/db_faiss'

# Create vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob='ADA.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    texts = text_splitter.split_documents(documents) # documents will be splitted into text

    # Used to sentence transformer
    embeddings = HuggingFaceBgeEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs = {'device' : 'cpu'})

    db = FAISS.from_documents(texts,embeddings)
    db.save_local(DB_FAISS_PATH)
    print("Vector Database Created")

if __name__ == '__main__':
    create_vector_db()