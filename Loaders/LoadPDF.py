from langchain_community.document_loaders.pdf import PyPDFLoader
import time
import Loaders.TextParser as TextParser

# Function to load and split the PDF into chunks
def load_pdf(file_path):
    
    print(f"Loading PDF ...")
    start = time.time()

    loader = PyPDFLoader(file_path)
    # Load the PDF content
    documents = loader.load()
    
    # Split documents into manageable chunks for processing
    #It seems that higher chunk size is processed faster,
    #but lower chunk size produces better granularity is analysis results.
    docs = TextParser.split_doc_by_char(documents, 4000, 100)
    docs = TextParser.truncate_doc(docs, 21)
    
    end = time.time()
    print(f"Loaded PDF in {(end-start):.2f} seconds")

    return docs