import os
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
import time

# Set your OpenAI API key (replace 'your-api-key' with your actual API key)
# os.environ["OPENAI_API_KEY"] = "your-api-key"

# Function to load and split the PDF into chunks
def load_pdf(file_path):
    
    print(f"Loading PDF ...")
    start = time.time()

    loader = PyPDFLoader(file_path)
    # Load the PDF content
    documents = loader.load()
    
    # Split documents into manageable chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    end = time.time()
    print(f"Loaded PDF in {(end-start):.2f} seconds")

    return docs

# Summarization function that chains map, reduce, and stuff methods
from langchain_core.output_parsers import StrOutputParser
def chained_summarization(docs):
    
    llm = ChatOpenAI(
        temperature=0.1,
        model_name="tinyllama:latest",
        api_key="ollama",
        #base_url="http://192.168.1.137:11434/v1",
        base_url="http://localhost:11434/v1",
    )
    
    prompt_template = """Write a long summary of the following document. 
    Only include information that is part of the document. 
    Do not include your own opinion or analysis.

    Use this content:
    "{context}"
    Summary:"""
    prompt = PromptTemplate.from_template(prompt_template)

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

# Main function to run the chained summarization
def main(pdf_file):
    print(f"Summarizing ...")
    start = time.time()
    # Load the PDF file
    docs = load_pdf(pdf_file)
    
    # Apply the chained summarization
    final_summary = chained_summarization(docs[:-3])

    end = time.time()
    print(f"Summarized in {(end-start):.2f} seconds")

    # Output the final summary
    from textwrap import fill
    print("\n\n=== Final Summary ===")
    print(fill(final_summary["output_text"]))

if __name__ == "__main__":
    # Replace 'your-pdf-file.pdf' with your actual PDF file path
    pdf_file_path = 'document.pdf'
    main(pdf_file_path)
