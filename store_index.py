from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE

extracted_data = load_pdf_file(data='Data/')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE)

index_name = "medical-bot"

if not pc.has_index(index_name):
    pc.create_index(
        metric="cosine",
        name=index_name,
        vector_type="dense",
        dimension=384,
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
   
    )

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name = index_name,
    embedding=embeddings
)