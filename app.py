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
from fuzzywuzzy import process, fuzz

# Predefined Q&A dictionary
predefined_answers = {
    "How do I apply?": "MYP4- complete the digital application on our website before April 1st. \n\nMYP5- complete the digital application on our website before March 1st. In addition, you must complete the vigo application and MYP5 Blindern must be your first choice.",
    "Do I submit grades": "Yes, you will submit grades for the past two years. The grades can be uploaded when you complete the school application.",
    "Who can write a reference letter?": "You can ask a teacher at your school.",
    "What will determine if I get accepted?": "Admission to the program is based on the academic record of the student.",
    "Can you tell me about the make-up of the class?": "We have a very diverse class. We have applicants from Norwegian public and private schools and from abroad.",
    "Do I have to apply to MYP5, if I am currently in MYP4": "Once you are accepted into our program, you will be given priority. This means that you will not compete with outside applicants. You must be passing your classes to move on in the program.",
    "I am currently taking Spanish/German at my current school. Can I continue with the same language?": "We offer French on two levels (beginner and intermediate). French is a mandatory subject. It will not be possible to replace French with another language (exception- if French is your mother tongue language).",
    "When will I find out if I am accepted?": "We should have all decisions out by the first week of June.",
    "If I am not accepted into MYP4, can I apply to MYP5 the following year?": "Admission can be quite competitive. If you are not offered a place, we encourage you to apply the following year. We have 64 places in MYP5 and only 28 in MYP4.",
    "Can I set up a meeting with the coordinators to find out more about the program?": "Yes, you can contact Aldo Mercado for MYP and Morten Døviken for DP. You are also welcome to visit the school during Open Day at the end of January.",
    "What is the Middle Years Programme?": "The MYP is a challenging framework that encourages students to make practical connections between their studies and the real world. The MYP is inclusive by design; students of all interests and academic abilities can benefit from their participation.",
    "What is the Diploma Programme?": "The DP is a two-year program that aims to develop students who have excellent breadth and depth of knowledge – students who flourish physically, intellectually, emotionally and ethically.",
    "What is the IB?": "The International Baccalaureate® (IB) offers high quality programmes of international education to a worldwide community of schools. There are more than 1.4 million IB students in over 150 countries.",
    "Where do I submit my application for MYP4 or MYP5?": "MYP4- complete the digital application on our website before April 1st. \n\nMYP5- complete the digital application on our website before March 1st. In addition, you must complete the vigo application and MYP5 Blindern must be your first choice.",
    "What is the procedure for applying to MYP4/MYP5?": "MYP4- complete the digital application on our website before April 1st. \n\nMYP5- complete the digital application on our website before March 1st. In addition, you must complete the vigo application and MYP5 Blindern must be your first choice.",
    "Can you guide me through the application process?": "MYP4- complete the digital application on our website before April 1st. \n\nMYP5- complete the digital application on our website before March 1st. In addition, you must complete the vigo application and MYP5 Blindern must be your first choice.",
    "Which steps do I follow to complete my digital application?": "MYP4- complete the digital application on our website before April 1st. \n\nMYP5- complete the digital application on our website before March 1st. In addition, you must complete the vigo application and MYP5 Blindern must be your first choice.",
    "When and how should I apply for the IB MYP program?": "MYP4- complete the digital application on our website before April 1st. \n\nMYP5- complete the digital application on our website before March 1st. In addition, you must complete the vigo application and MYP5 Blindern must be your first choice.",
    "Must I send my school report?": "Yes, you will submit grades for the past two years. The grades can be uploaded when you complete the school application.",
    "Is it necessary to submit my latest school grades?": "Yes, you will submit grades for the past two years. The grades can be uploaded when you complete the school application.",
    "How should I send my academic records?": "Yes, you will submit grades for the past two years. The grades can be uploaded when you complete the school application.",
    "Will my past grades impact my application?": "Yes, you will submit grades for the past two years. The grades can be uploaded when you complete the school application.",
    "Can my teacher write my recommendation letter?": "You can ask a teacher at your school.",
    "Is a recommendation from my school required?": "You can ask a teacher at your school.",
    "What are the requirements for acceptance?": "Admission to the program is based on the academic record of the student.",
    "What qualifications do I need to get in?": "Admission to the program is based on the academic record of the student.",
    "Will my previous academic performance be reviewed?": "Admission to the program is based on the academic record of the student.",
    "How competitive is admission to the program?": "Admission can be quite competitive. If you are not offered a place, we encourage you to apply the following year. We have 64 places in MYP5 and only 28 in MYP4. Admission to the program is based on the academic record of the student.",
    "How soon will I receive my acceptance letter?": "We should have all decisions out by the first week of June.",
    "What is the deadline for acceptance notifications?": "We should have all decisions out by the first week of June.",
    "When are final admission results released?": "We should have all decisions out by the first week of June.",
    "Can I try again for admission in MYP5 if I miss the spot for MYP4?": "Admission can be quite competitive. If you are not offered a place, we encourage you to apply the following year. We have 64 places in MYP5 and only 28 in MYP4.",
    }


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


def get_response(user_input):
    # Convert input to lowercase for case-insensitive matching
    user_input = user_input.lower().strip()

    # Check for an exact match first
    for question in predefined_answers.keys():
        if user_input == question.lower():
            return predefined_answers[question]

    # Use fuzzy matching to find a close match
    best_match, score = process.extractOne(user_input, predefined_answers.keys())

    # If the similarity is high (ensuring relevance), return the predefined answer
    if score > 93:  # 93 is strict enough to prevent mistakes
        return predefined_answers[best_match]

    # No predefined match found -> use RAG (LLM)
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({ 
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']
#app config
st.set_page_config(page_title="Blindern IB AI", page_icon="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSwSKwMRFTf_05ETkNe_zE6Bd91Xh6C-ty8uQ&s")
st.title("Blindern IB AI")

#sidebar
with st.sidebar:
    st.header("My personal project")
    st.write("So, this is my personal project. Its an RAG LLM, which has specific access to the webpages that are on the schools webpage (related to IB). Its purpose is to make it easier to answer questions for potential IB applicants for the school, since the the schools website can be annoying to navigate sometimes. And its just simpler.")

    
 

    st.write("NB! Please reload the website after you have asked a question on a topic, and you're moving onto a different topic. There is a limitation where it (the AI) does not take in more information than one link, so it will answer incorrectly of you don't. (This doesn't always happen, but it's a precaution you could take to avoid getting the wrong info.)")

    
 
    st.write("Also, sometimes the AI wil be stubborn and just refuse to listen. In that case reloading will also help, and most likely also provide a correct answer.")

    
 
    st.write("Finally, some answers may be wrong or formatted weird, in that case maybe try again or just don't your choice ig.")
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
    "https://blindern.vgs.no/ib/ib-news/konkurransen-unge-forskere/",]
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
