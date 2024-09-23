from langchain_community.document_loaders.text import TextLoader
import time
import Loaders.TextParser as TextParser

# Function to load and split the text into chunks
def load_text(file_path):
    
    print(f"Loading Text ...")
    start = time.time()

    loader = TextLoader(file_path)
    # Load the PDF content
    documents = loader.load()
    
    # Split documents into manageable chunks for processing
    #It seems that higher chunk size is processed faster,
    #but lower chunk size produces better granularity is analysis results.
    docs = TextParser.split_doc_by_char(documents, 4000, 100)
    
    end = time.time()
    print(f"Loaded Text in {(end-start):.2f} seconds")

    return docs