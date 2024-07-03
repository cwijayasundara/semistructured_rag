from unstructured.partition.html import partition_html

input_html = "nested_table.html"

# Using partition_html to ingest HTML content
document_elements = partition_html(filename=input_html,
                                   chunking_strategy="by_title",
                                   max_characters=4096,
                                   new_after_n_chars=4096,
                                   overlap=256,
                                   combine_text_under_n_chars=2000
                                   )

print(len(document_elements))

for element in document_elements:
    print(element.text)
    print(element.metadata)
    print("\n")
