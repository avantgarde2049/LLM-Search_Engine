from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper #need these wrapper fns to run the above tools
import streamlit as st
import os
from langchain_classic.agents import initialize_agent,AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama 


#Arxiv and wikipedia tools
wiki_wrap=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250) #wrapper fetches the summary results we want to filter out
wiki=WikipediaQueryRun(api_wrapper=wiki_wrap)

arxiv_wrap=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrap)

search= DuckDuckGoSearchRun(name="Search")
# -----------------------------
# Streamlit UI
# -----------------------------

st.title("Langchain - Chat with Search")
"""
In this example,we're using `StreamLitCallbackHandler` to display the thoughts and actions of an agent in an interactive Steamlit app.

"""

#Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Ollama API Key:",type="password")
# -----------------------------
# Session state
# -----------------------------

if "messages" not in st.session_state: #creating session state
    st.session_state["messages"]=[    ##message and roles saved in session state
        {"role":"assistant","content":"Hi! I'm a chatbot.How may I help you today."}
    ] 
# Display previous messages
for msg in st.session_state.messages:  
    st.chat_message(msg["role"]).write(msg['content']) #all sessions will be recorded in session state in this key value pair
# -----------------------------
# User input
# -----------------------------

if prompt:=st.chat_input(placeholder="What is Machine Learning"): #this prompt by default will have this specific placeholder
    st.session_state.messages.append({"role":"user","content":prompt}) # the next input by user would be the prompt and appended in session state in next write method
    st.chat_message(msg["role"]).write(prompt)
# -----------------------------
    # LLM
    # -----------------------------

    if api_key:
        llm = ChatOllama(
        model="mistral",
        ollama_api_key=api_key,
        streaming=True)
# -----------------------------
        # Tools
        # -----------------------------
    tools=[wiki,arxiv,search]
# -----------------------------
        # Agent
        # -----------------------------

    #Convert the tools into an agent to invoke this agent
    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors=True)
# -----------------------------
        # Assistant response
        # -----------------------------
#so when assistant is giving any reponse
    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False) #st.container contains multiple argumenst of the assistant actioning and thinking
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant","content":response}) # the output by assistant as response
        st.write(response)
