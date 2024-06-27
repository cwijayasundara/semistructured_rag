import warnings
import os
from dotenv import load_dotenv
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone, ServerlessSpec
from DLAIUtil import Utils
warnings.filterwarnings('ignore')
_ = load_dotenv()

import os

api_key = os.environ["PINECONE_API_KEY"]

utils = Utils()
INDEX_NAME = utils.create_dlai_index_name('dl-ai')

pinecone = Pinecone(api_key=api_key)

if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
    pinecone.delete_index(INDEX_NAME)
pinecone.create_index(
    INDEX_NAME,
    dimension=512,
    metric="dotproduct",
    spec=ServerlessSpec(cloud='aws', region='us-west-2')
)
index = pinecone.Index(INDEX_NAME)
print(f"Index {INDEX_NAME} created successfully")