import os
import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from langchain_community.vectorstores import FAISS 
from langchain_core.prompts import PromptTemplate

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFacePipeline.from_model_id(
        model_id=huggingface_repo_id,
        task="text2text-generation",
        pipeline_kwargs={"max_new_tokens": 512, "temperature": 0.5}
    )
    return llm

def main():
    st.title("🚀 JAI SHREE RAM BOT!")

    if "messages" not in st.session_state:
        st.session_state.messages = []    # store conversation history

    for message in st.session_state.messages:  
        st.chat_message(message["role"]).write(message["content"])  

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})  

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you don't know the answer, just say that you don't know.
        Don't provide anything out of the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        HUGGINGFACE_REPO_ID = "google/flan-t5-small"  # small model for testing
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load vectorstore.")
                return
         
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )
      
            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            source_documents = response["source_documents"]

            result_to_show = result + "\n\n**Sources:** " + str([doc.metadata for doc in source_documents])
            
            st.chat_message("assistant").markdown(result_to_show)
            st.session_state.messages.append({"role": "assistant", "content": result_to_show})

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.stop()

if __name__ == "__main__":
    main()
