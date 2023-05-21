from dotenv import load_dotenv
from os.path import exists
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import json


def write_json(dic, file):
    with open(file, 'w') as f:
        json.dump(dic, f)


def read_json(file):
    with open(file, 'r') as f:
        return json.load(f)


def convert_to_chunks(pdf_a):

    pdf_reader = PdfReader(pdf_a)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    return chunks


def create_indexes(chunks, path_to_local_embeddings=None):

    embeddings = OpenAIEmbeddings()

    if (path_to_local_embeddings is not None) and exists(path_to_local_embeddings):
        knowledge_base = FAISS.load_local("./bindexes.ind", embeddings)
    else:
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        knowledge_base.save_local("./bindexes.ind")

    return knowledge_base


def talk_to_llm(text, indexe, chain):
    docs = ""
    try:
        docs = indexe.similarity_search(text)
    except:
        pass
    question = chain.run(input_documents=docs, question=text)

    return question


def main():
    load_dotenv()
    st.set_page_config(page_title="ElmTech")
    st.header("ELMTECH")
    c1, c2 = st.columns(2)
    c1.text("Name: Student1")
    c2.text("Subject: Electricity-101")
    pdf = "./electric2.pdf"
    st.header("Question")
    print("Your pdf path: {}".format(pdf))

    j = read_json("qa.json")

    llm = OpenAI(temperature=0)

    # extract the text
    if pdf is None:
        raise Exception

    # Chunks
    chunks = convert_to_chunks(pdf)

    # create embeddings
    print("Creating Embeddings...")
    knowledge_base = create_indexes(chunks, "./bindexes.ind")

    chain = load_qa_chain(llm, chain_type="stuff")

    # j["question_round"]
    # Generate Question
    question_gen = "Generate a random question form the electricity document"
    question = talk_to_llm(question_gen, knowledge_base, chain)

    # Show user input
    st.write("- {}".format(question))
    user_answer = st.text_input("Your Answer: ")

    st.subheader("Result")
    # Check if answer exist
    if user_answer:
        answer = "Answer with yes only or no and explination and page number if the answer is wronge, is ""{}"" an answer for the question ""{}""?".format(
            question, user_answer)
        response2 = talk_to_llm(answer, knowledge_base, chain)
        st.write(response2)


if __name__ == '__main__':
    main()
