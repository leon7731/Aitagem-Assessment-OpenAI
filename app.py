__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import streamlit as st
import pandas as pd
from io import StringIO

from llm_model import LLM_Model
from Config.Config import settings
import os

# # Set Local environment variables
# os.environ["OPENAI_API_KEY"] = settings.openai_api_key
# os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
# os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Set Streamlit environment variables
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["langchain_api_key"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"

st.title("Aitagem Assessment (OpenAI)")

# Uploading files
uploaded_file = st.file_uploader("Choose your Student Exam Result file", type="csv")
if uploaded_file is not None:
    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    file_content = stringio.getvalue()  # Extracting string content from StringIO
    
    # Show DataFrame
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)
    
    # Convert DataFrame to text format
    text_output = ""
    for index, row in dataframe.iterrows():
        for column in dataframe.columns:
            text_output += f"{column}: {row[column]}\n"
        text_output += "\n"  # Add a newline to separate rows
    
    # # Display the text output
    # st.text(text_output)
    
    
    
    # Load the score grade data
    score_grade_df = pd.read_csv("Data/score_grade.csv") 
    
    # Load the subject weight values
    subject_weight_values_df = pd.read_csv("Data/subject_weight_values.csv")
    
    
    
    # Initialize LLM_Model with the student report file content
    llm_model = LLM_Model(student_report_file=text_output, 
                          score_grade = score_grade_df,
                          subject_weight_values = subject_weight_values_df,
                          persist_directory="./university_courses_vector_db")
    
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Enter your question:"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get response
        response_content = llm_model.query(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response_content)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_content})
else:
    st.write("Please upload the student report file first.")
