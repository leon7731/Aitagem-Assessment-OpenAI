# import os
# from groq import Groq
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# # Config Folder
# from Config.Config import settings

# # os.environ["OPENAI_API_KEY"] = settings.openai_api_key
# os.environ["GROQ_API_KEY"] = settings.groq_api_key
# os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
# os.environ["LANGCHAIN_TRACING_V2"] = "true"


# class LLM_Langchain:
#     def __init__(self, file_content, llm_model_name, system_prompt):
#         self.file_content = file_content
#         self.llm_model_name = llm_model_name
#         self.system_prompt= system_prompt
 
        
#     def generate_response(self):
        
#         # LLM Model
#         llm_model = ChatGroq(temperature=0, 
#                              model_name=self.llm_model_name)
        
#         # Create Prompt Template
#         system = self.system_prompt
#         human = "{text}"
#         prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
        
        
#         # Output Parser
#         output_parser = StrOutputParser()
        
#         # Create Chain
#         chain = prompt | llm_model | output_parser
        
#         # Invoke Chain
#         output = chain.invoke({"text": "Evaluate the call quality of the following call: \n\n" + self.file_content})
        
#         return output
     