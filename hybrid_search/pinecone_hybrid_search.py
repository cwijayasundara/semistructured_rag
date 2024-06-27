import warnings
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone
import ssl

from langchain_community.retrievers import PineconeHybridSearchRetriever

warnings.filterwarnings('ignore')
_ = load_dotenv()

pc = Pinecone(api_key="0c310e8b-447b-49e8-bc84-1d36a7478798")
index = pc.Index("hybrid-search-index")

print(index)

embeddings = OpenAIEmbeddings()

# or from pinecone_text.sparse import SpladeEncoder if you wish to work with SPLADE
# use default tf-idf values
ssl._create_default_https_context = ssl._create_unverified_context
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

retriever.add_texts(["foo", "bar", "world", "hello"])

result = retriever.invoke("foo")
print(result)