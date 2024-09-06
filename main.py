from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import textwrap
import os
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
from dotenv import load_dotenv

load_dotenv()

def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# Ensure Google Palm LLM authentication
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is not set in environment variables")

llm = GooglePalm(google_api_key=google_api_key, temperature=0.1)

instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"
model = genai.GenerativeModel('gemini-1.5-flash')

def create_vector_db():
    loader = CSVLoader(file_path='Data_FAQ1.csv', source_column="prompt")
    data = loader.load()

    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)

def get_qa_chain(query):
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)
    retriever = vectordb.as_retriever(score_threshold=0.7)
    context = retriever.get_relevant_documents(query)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {query}"""

    prompt = PromptTemplate(
        input_variables=["context", "query"],
        template=prompt_template
    ).format(context=context, query=query)
    
    response = chat.send_message(prompt)
    print(response.text)

if __name__ == "__main__":
    # Uncomment this if you need to create the vector database initially
    create_vector_db()
    chat = model.start_chat(history=[])
    
    while True:
        query = input("Enter your query (or type 'quit' to stop or type 'history' for history): ")
        if query.lower() == "quit":
            break
        elif query.lower() == "history":
            for message in chat.history:
                display(Markdown(f'**{message.role}**: {message.parts[0].text}'))
            break
        get_qa_chain(query)
