import warnings
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from unstructured.partition.html import partition_html
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from unstructured.chunking.title import chunk_by_title
from langchain_community.vectorstores.utils import filter_complex_metadata

warnings.filterwarnings('ignore')
_ = load_dotenv()

filename = "../docs/nvidia_financial_results_q1_fiscal_2025.html"

html_elements = partition_html(filename=filename,
                               max_characters=4096)

elements = chunk_by_title(html_elements)


documents = []
for element in elements:
    metadata = element.metadata.to_dict()
    del metadata["languages"]
    metadata["source"] = metadata["filename"]
    documents.append(Document(page_content=element.text, metadata=metadata))

# Filter out elements with complex metadata that are not useful for the vector store
documents = filter_complex_metadata(documents)

embeddings = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(documents, embeddings)

query = "Whats the basic net income per share for the three months ended April 28, 2024?"

retriever = vectorstore.as_retriever()

result = retriever.invoke(query, k=4)

print(result)

template = """Answer the question based only on the following context, which can include text and tables:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(temperature=0,
                   model="gpt-4o")

# RAG pipeline
chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
)

query = "Whats the basic net income per share for the three months ended April 28, 2024?"

response = chain.invoke(query)
print(response)
