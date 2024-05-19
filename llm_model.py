from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# Open AI Model: "gpt-3.5-turbo", 'gpt-4o'
class LLM_Model:
    def __init__(self, student_report_file, 
                 score_grade, 
                 subject_weight_values,
                 persist_directory, 
                 model_name="gpt-3.5-turbo"):
        
        # Load the student report file
        self.student_report_document = {
            "title": "Student Report File",
            "content": student_report_file
        }
        
        # Load the score grade
        self.score_grade = score_grade
        
        # Load the subject weight values
        self.subject_weight_values = subject_weight_values
        
        # Load the vector store from disk
        self.db = Chroma(
            embedding_function=OpenAIEmbeddings(),
            persist_directory=persist_directory
        )

        # Initialize ChatOpenAI
        self.llm = ChatOpenAI(model=model_name, temperature=0)

        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_template("""
    
        The are the STPM Rules you need to follow before you answer the question:
        - STPM has 3 semesters, and each semester has 4 subjects and for the entire 3 semesters have 1 coursework for each subject.
                                 
        These are the Rules you need to follow before you answer the question:
        - Any university course information should only be answered based on only the information provided in the student report file. Any additional information should not be considered.
        - All answers should only be based on the {student_report_document} student report file, {score_grade} score grade file, and {subject_weight_values} subject weight values file before providing any answers.
        
        With the {student_report_document} student report file, {score_grade} score grade file, and {subject_weight_values} subject weight values file information, you can now answer the following questions:
        1. Extract Basic Information about the Course based on the Top 5 the Universities:
           i) University Name
           ii) Course Title
           iii) Course URL
           iv) Duration of the Course
           v) Entry Requirements
           vi) Course Fees
           
        2. Analyze Admission Criteria:
           - What are the key admission criteria for this course?
           - How competitive is the admission process for this course?
           - Are there specific grades or test scores required?
           
        
        3. Scholarship and Financial Aid:
            - Are there any scholarships or financial aid available for this course?
            - What are the eligibility criteria for scholarships or financial aid?
            - How can a student apply for scholarships or financial aid?
            
           
        4. University Ranking and Reputation:
           - How is the university ranked globally and within the country?
           - What is the reputation of the university and this specific course?
           
        5. Recommendations for Students:
           - Based on the University information, {student_report_document} student report file, {score_grade} score grade file, and {subject_weight_values} subject weight values file information suggest what steps can a student take to improve their chances of admission? The explanation should be concise and descriptive with how you arrived at the recommendation with justification.
           - Are there particular skills or experiences that the university values?
           - What are the common characteristics of successful applicants?
           
   
        <context>
        {context}
        </context>
        Question:
        {input}
        """)

        # Create Stuff Document Chain and Retrieval Chain
        self.document_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.retriever = self.db.as_retriever()
        self.retrieval_chain = create_retrieval_chain(self.retriever, self.document_chain)

    def query(self, input_text):
        response = self.retrieval_chain.invoke({
            "input": input_text,
            "context": "Provide detailed and useful information based on the student report.",
            "student_report_document": self.student_report_document,
            "score_grade": self.score_grade,
            "subject_weight_values": self.subject_weight_values
        })
        return response['answer']
