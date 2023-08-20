import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

import pickle
import os

# Sidebar contents
with st.sidebar:
    st.title('Chat With PDF 📄')
    st.markdown('''
    ## About
    This is a PDF chat application, built using 
    [Streamlit](https://streamlit.io/) ,
    [LangChain](https://python.langchain.com/) ,
    [OpenAI](https://platform.openai.com/docs/models) LLM model.
                
 
    ''')
    add_vertical_space(5)
    st.write('''
    #### Resources:
    [Prompt Engineer](https://youtube.com/@engineerprompt)   
    ''')


def main():
    st.header('Chat with pdf')
    pdf = st.file_uploader("Upload your PDF here",type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        # st.write(pdf_reader)

        # st.write(pdf_reader.pages[20].extract_text())

        text = ""
        
        for page in pdf_reader.pages:
            text = text+page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )

        chunks = text_splitter.split_text(text=text)

        #embeddings
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
                # st.write('Embeddings Loaded from the Disk')
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
                # st.write('Embeddings computation completed')
        
        # accepts for inputs 
        query = st.text_input(f"Ask questions from your document \n {pdf.name}")
        # st.write(query)

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI(temperature=0,)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            reponse = chain.run(input_documents=docs, question=query)
            st.write(reponse)
            # st.write(docs)

        
        # st.write(chunks)




if __name__ == '__main__':
    main()