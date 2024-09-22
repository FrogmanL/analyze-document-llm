import os
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from textwrap import fill
import time


# Set your OpenAI API key (replace 'your-api-key' with your actual API key)
# os.environ["OPENAI_API_KEY"] = "your-api-key"

# Function to load and split the text into chunks
def load_web_page(url):
    
    print(f"Loading Web Page ...")
    start = time.time()

    loader = WebBaseLoader(url)
    # Load the PDF content
    documents = loader.load()
    
    # Split documents into manageable chunks for processing
    #It seems that higher chunk size is processed faster,
    #but lower chunk size produces better granularity is analysis results.
    docs = split_doc_by_char(documents, 4000, 100)
    
    end = time.time()
    print(f"Loaded Text in {(end-start):.2f} seconds")

    return docs

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
    docs = split_doc_by_char(documents, 4000, 100)
    
    end = time.time()
    print(f"Loaded Text in {(end-start):.2f} seconds")

    return docs

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
    docs = split_doc_by_char(documents, 4000, 100)
    docs = truncate_doc(docs, 21)
    
    end = time.time()
    print(f"Loaded PDF in {(end-start):.2f} seconds")

    return docs

# split document into manageable chunks
def split_doc_by_char(documents, chunkSize, chunkOverlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunkSize, chunk_overlap=chunkOverlap)
    split_text = text_splitter.split_documents(documents)
    return split_text

# Truncate doc to remove footer data
def truncate_doc(split_text, endPage):
    text_splitter_no_last_page = [x for x in split_text if x.metadata['page'] < endPage]
    return text_splitter_no_last_page

def define_llm():
    return ChatOpenAI(
        temperature=0.1,
        model_name="tinyllama:latest",
        api_key="ollama",
        #base_url="http://192.168.1.137:11434/v1",
        base_url="http://localhost:11434/v1",
    )

# Summarization function that chains map and reduce methods
def map_reduce_summarization(docs):
    
    llm = define_llm()
    
    # Step 1: Use the 'map' approach to create a summary for each chunk
    print(f"Mapping ...")
    start = time.time()
    map_chain = load_summarize_chain(llm, chain_type="map_reduce")
    intermediate_summaries = map_chain.invoke(docs)
    end = time.time()
    print(f"Mapped in {(end-start):.2f} seconds")


    # Step 2: Use the 'reduce' approach to further summarize the intermediate summaries
    print(f"Reducing ...")
    start = time.time()
    reduce_chain = load_summarize_chain(llm, chain_type="map_reduce")
    final_summary = reduce_chain.invoke(intermediate_summaries)
    end = time.time()
    print(f"Reduced in {(end-start):.2f} seconds")

    return final_summary

# Summarization function that chains stuff method
def stuff_summarization(docs):
    
    llm = define_llm()
    
    print(f"Stuffing ...")
    start = time.time()
    map_chain = load_summarize_chain(llm, chain_type="stuff")
    final_summary = map_chain.invoke(docs)
    end = time.time()
    print(f"Stuffed in {(end-start):.2f} seconds")

    return final_summary

# Main function to run the summarization
def main(path):
    print(f"Summarizing ...")
    start = time.time()
    
    # Load the PDF file
    #docs = load_pdf(path)
    #docs = load_text(path)
    docs = load_web_page(path)

    # Apply the stuff summarization
    #final_summary = stuff_summarization(docs)

    # Apply the map/reduce summarization
    final_summary = map_reduce_summarization(docs)

    end = time.time()
    print(f"Summarized in {(end-start):.2f} seconds")

    # Output the final summary
    with open("output.txt", "w", errors="ignore") as f:
        f.write(final_summary["output_text"])    

    print("\n\n=== Final Summary ===")
    print(fill(final_summary["output_text"]))

if __name__ == "__main__":
    # Replace 'your-pdf-file.pdf' with your actual PDF file path
    #path = './Inputs/' + 'document.pdf'
    #path = './Inputs/' + 'text.txt'
    path = 'https://en.wikipedia.org/wiki/Athletics_at_the_1904_Summer_Olympics_%E2%80%93_Men%27s_marathon'
    
    main(path)
