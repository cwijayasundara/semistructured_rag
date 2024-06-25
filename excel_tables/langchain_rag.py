import warnings
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub

warnings.filterwarnings('ignore')
_ = load_dotenv()

loader = UnstructuredExcelLoader("../docs/nvidia_quarterly_revenue_trend_by_market.xlsx",
                                 mode="elements",
                                 infer_table_structure=True,
                                 chunking_strategy="by_title",
                                 max_characters=4096,
                                 new_after_n_chars=3800,
                                 combine_text_under_n_chars=2000)

docs = loader.load()

# export the documents to a markdown file
for doc in docs:
    with open(f"../docs/{doc.metadata['page_name']}.md", "w") as f:
        content_metadata = doc.page_content + "\n\n" + str(doc.metadata)
        f.write(content_metadata)

documents = filter_complex_metadata(docs)

vectorstore = Chroma.from_documents(documents=docs,
                                    embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

llm = ChatOpenAI(model_name="gpt-4o",
                 temperature=0)

prompt = hub.pull("rlm/rag-prompt")


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Chain
rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

# Question
response = rag_chain.invoke("What is the total revenue in Q1 FY25?")
print(response)
