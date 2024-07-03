from unstructured.partition.html import partition_html
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

filename = "nested_table.html"

html_elements = partition_html(filename=filename,
                               chunking_strategy="by_title",
                               max_characters=4096,
                               new_after_n_chars=3800,
                               combine_text_under_n_chars=2000)

documents = []
for element in html_elements:
    metadata = element.metadata.to_dict()
    del metadata["languages"]
    del metadata["orig_elements"]
    metadata["source"] = metadata["filename"]
    documents.append(Document(page_content=element.text, metadata=metadata))

# print the content of the table
# for doc in documents:
#     print(doc.page_content, doc.metadata)


embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

query = "When was Berlin Academy of Sciences founded?"

retriever = vectorstore.as_retriever()
result = retriever.invoke(query, k=4)
print(result)