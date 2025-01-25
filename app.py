#All of the libraries
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


load_dotenv()
 
#functions
def get_vectorstore_from_urls(urls):
    #get the text in document form
    loader_multiple_pages = WebBaseLoader(urls)
    document = loader_multiple_pages.load()

    #split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    #create a vectorstore from the chunks
    vector_store = FAISS.from_documents(document_chunks, OpenAIEmbeddings())
    return vector_store 

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"), 
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a seaarch query to look up in order to get information relevant to the conversation.")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
        
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


#AI response
def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store) 
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({ 
        "chat_history": st.session_state.chat_history,
        "input": user_query
        })
    
    return response['answer']
   
#app config
st.set_page_config(page_title="Blindern IB AI", page_icon="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSwSKwMRFTf_05ETkNe_zE6Bd91Xh6C-ty8uQ&s")
st.title("Blindern IB AI")

#sidebar
with st.sidebar:
    st.header("My personal project")
    st.write("So, this is my personal project. Its an RAG LLM, which has specific access to the webpages that are on the schools webpage. Its purpose is to make it easier to answer questions for potential IB applicants for the school, since the the schools website can be annoying to navigate sometimes. And its just simpler.")

    st.write("NB! Please reload the website after you have asked a question on a topic, and you're moving onto a different topic. There is a limitation where it (the AI) does not take in more information than one link, so it will answer incorrectly of you don't. (This doesn't alwasy happen, but it's a precaution you could take to avoid getting the wrong info.")

    st.write("Also, sometimes the AI wil be stubborn and just refuse to listen. In that case relaoding will also help, and most likely also provide a correct answer.")
    website_urls = (
    ["https://blindern.vgs.no/ib/ib-information/are-you-interested-in-ib/",
    "https://blindern.vgs.no/ib/ib-information/our-policies/",
    "https://blindern.vgs.no/ib/diploma-programme/information/",
    "https://blindern.vgs.no/ib/diploma-programme/dp-subjects-offer/",
    "https://blindern.vgs.no/ib/middle-years-programme/curriculum-overview/",
    "https://blindern.vgs.no/ib/middle-years-programme/information-for-applicants/",
    "https://blindern.vgs.no/ib/middle-years-programme/information/",
    "https://blindern.vgs.no/ib/ib-news/school-regulations/",
    "https://blindern.vgs.no/ib/ib-news/statements-from-former-ib-students/",
    "https://blindern.vgs.no/ib/ib-news/konkurransen-unge-forskere/",
    "https://www.ibo.org/programmes/middle-years-programme/"]
    )

 


#session state
if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
    AIMessage(content="Hello i am an AI chatbot and an expert on the IB system in Blindern VGS. How may i help you?")

]    
        
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore_from_urls(website_urls)


#user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    response = get_response(user_query)
    
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

#conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
