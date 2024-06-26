import warnings
from dotenv import load_dotenv
import os
from langchain_community.retrievers import (
    PineconeHybridSearchRetriever,
)
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from pinecone_text.sparse import BM25Encoder

warnings.filterwarnings('ignore')
_ = load_dotenv()

api_key = os.environ["PINECONE_API_KEY"]

index_name = "langchain-pinecone-hybrid-search"

# initialize Pinecone client
pc = Pinecone(api_key=api_key)

# create the index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # dimensionality of dense model
        metric="cosine",  # sparse values supported only for cosine
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

embeddings = OpenAIEmbeddings()

# or from pinecone_text.sparse import SpladeEncoder if you wish to work with SPLADE

bm25_encoder = BM25Encoder().default()

corpus = ["foo", "bar", "world", "hello"]

# fit tf-idf values on your corpus
bm25_encoder.fit(corpus)

# store the values to a json file
bm25_encoder.dump("bm25_values.json")

# load to your BM25Encoder object
bm25_encoder = BM25Encoder().load("bm25_values.json")

retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings,
    sparse_encoder=bm25_encoder,
    index=index
)

result = retriever.invoke("foo")
print(result)
