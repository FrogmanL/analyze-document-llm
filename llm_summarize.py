from textwrap import fill
import time
import Loaders
import AI.TextSummarization as TextSummarization
import Loaders.LoadWebPage

# Main function to run the summarization
def main(path):
    print(f"Summarizing ...")
    start = time.time()
    
    # Load the PDF file
    #docs = load_pdf(path)
    #docs = load_text(path)
    docs = Loaders.LoadWebPage.load_web_page(path)

    # Apply the stuff summarization
    #final_summary = stuff_summarization(docs)

    # Apply the map/reduce summarization
    final_summary = TextSummarization.stuff_summarization(docs)

    end = time.time()
    print(f"Summarized in {(end-start):.2f} seconds")

    # Output the final summary
    with open("./Outputs/output.txt", "w", errors="ignore") as f:
        f.write(final_summary["output_text"])    

    print("\n\n=== Final Summary ===")
    print(fill(final_summary["output_text"]))

if __name__ == "__main__":
    # Replace 'your-pdf-file.pdf' with your actual PDF file path
    #path = './Test/' + 'document.pdf'
    #path = './Test/' + 'text.txt'
    path = 'https://en.wikipedia.org/wiki/Athletics_at_the_1904_Summer_Olympics_%E2%80%93_Men%27s_marathon'
    
    main(path)
