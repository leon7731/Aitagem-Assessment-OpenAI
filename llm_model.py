from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

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
        
        These are the Rules you need to follow before you answer the question:
        - Any university course information should only be answered based on only the information provided in the student report file. Any additional information should not be considered.
        - All answers should only be based on the {student_report_document} student report file, {score_grade} score grade file, and {subject_weight_values} subject weight values file before providing any answers.
        - All answer outputs should only be followed based on the criteria and format provided.
        
                                       
                                                       
        1. Determine the Subject Grade Value (NGMP):
        - To calculate the STPM pointer, the weighted value of each subject is different. Below is an example of how to calculate the values for different subjects.
        - The STPM has 4 subjects and 3 semesters overall. The student's CGPA is calculated based on the subjects taken.
        Example:
        General Studies NGMP Calculation Method:
        - Weighted Value (%) X Subject Grade Pointer ÷ 100 = new value

        Subject: General Studies
        - Grade: A, General Studies 1 : 4.0, Weight: 29%, Mark: 1.16
        - Grade: B, General Studies 2 : 3.0, Weight: 22%, Mark: 0.66
        - Grade: A, General Studies 3 : 4.0, Weight: 29%, Mark: 1.16
        - Grade: A, General Studies Coursework 4 (KK) : 4.0, Weight: 20%, Mark: 0.8
        Total: 3.78
        - Grade: A-, NGMP: 3.67

        Subject: Accounts
        - Grade: A, Accounts 1 : 4.0, Weight: 28%, Mark: 1.12
        - Grade: A, Accounts 2 : 4.0, Weight: 28%, Mark: 1.12
        - Grade: A-, Accounts 3 : 3.67, Weight: 20%, Mark: 0.73
        - Grade: A, Accounts Coursework 4 (KK) : 4.0, Weight: 20%, Mark: 0.8
        Total: 3.69
        - Grade: A-, NGMP: 3.67

        Subject: Economics
        - Grade: A-, Economics 1 : 3.67, Weight: 26.67%, Mark: 0.98
        - Grade: A-, Economics 2 : 3.67, Weight: 26.67%, Mark: 0.98
        - Grade: B+, Economics 3 : 3.33, Weight: 26.67%, Mark: 0.89
        - Grade: A, Economics Coursework 4 (KK) : 4.0, Weight: 20%, Mark: 0.8
        Total: 3.65
        - Grade: B+, NGMP: 3.33

        Subject: History
        - Grade: A-, History 1 : 3.67, Weight: 29%, Mark: 1.06
        - Grade: B, History 2 : 3.00, Weight: 22%, Mark: 0.66
        - Grade: A, History 3 : 4.00, Weight: 29%, Mark: 1.16
        - Grade: A, History Coursework 4 (KK) : 4.0, Weight: 20%, Mark: 0.8
        Total: 3.68
        - Grade: A-, NGMP: 3.67

        2. Calculation of CGPA:
        - The student's CGPA is calculated based on the four best subjects taken by the candidate only.
        - Formula: Total NGMP ÷ 4 = CGPA

        Example:
        - General Studies: 3.67
        - Accounts: 3.67
        - Economics: 3.33
        - History: 3.67
        Total NGMP: 3.67 + 3.67 + 3.33 + 3.67 = 14.34
        CGPA: 14.34 ÷ 4 = 3.585
        
        3. verify the student's CGPA. If the student's CGPA is more than 4.0 then recalculate the CGPA again.
        
        4. The output after the calculation of the CGPA should follow the format below:
            Example Output:
            - General Studies: 3.67
            - Accounts: 3.67
            - Economics: 3.33
            - History: 3.67
            Total NGMP: 3.67 + 3.67 + 3.33 + 3.67 = 14.34
            CGPA: 14.34 ÷ 4 = 3.585

            Therefore, your current CGPA is 3.585.

        
        With the above information, you can now answer the following questions:
        1. Extract Basic Information:
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
           
        3. Recommendations for Students:
           - Based on the extracted information, what steps can a student take to improve their chances of admission?
           - Are there particular skills or experiences that the university values?
           - What are the common characteristics of successful applicants?
           
        4. University Ranking and Reputation:
           - How is the university ranked globally and within the country?
           - What is the reputation of the university and this specific course?
           
        5. Personalization:
           - Tailor the recommendations based on the student's current academic CGPA performance and extracurricular activities.
           - Provide a realistic action plan for the student to follow.
        
            
            
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
