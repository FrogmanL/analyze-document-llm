from langchain.text_splitter import RecursiveCharacterTextSplitter

# split document into manageable chunks
def split_doc_by_char(documents, chunkSize, chunkOverlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunkSize, chunk_overlap=chunkOverlap)
    split_text = text_splitter.split_documents(documents)
    return split_text

# Truncate doc to remove footer data
def truncate_doc(split_text, endPage):
    text_splitter_no_last_page = [x for x in split_text if x.metadata['page'] < endPage]
    return text_splitter_no_last_page