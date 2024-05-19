# Config Folder
from Config.Config import settings

import os

# Data Loader
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter

## Vector Embedding And Vector Store
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

os.environ["OPENAI_API_KEY"] = settings.openai_api_key


# # Universiti Sains Malaysia (USM) Info
# usm_info = [
#     "https://unienrol.com/u/universiti-sains-malaysia-usm"
# ]

# # Universiti Sains Malaysia (USM) Undergrad Programs

# usm_accounting_finance_undergrad_program = [
#     "https://unienrol.com/c/universiti-sains-malaysia-usm-degree-in-accounting-429bac"
# ]


# usm_computer_undergrad_program = [
#     "https://unienrol.com/c/universiti-sains-malaysia-usm-degree-in-computer-science-429bbf"
# ]


# usm_engineering_undergrad_program = [
#     "https://unienrol.com/c/universiti-sains-malaysia-usm-degree-in-manufacturing-engineering-429bc2",
#     "https://unienrol.com/c/universiti-sains-malaysia-usm-degree-in-mechatronics-engineering-429bc4",
#     "https://unienrol.com/c/universiti-sains-malaysia-usm-degree-in-mechanical-engineering-429bc1",
#     "https://unienrol.com/c/universiti-sains-malaysia-usm-degree-in-chemical-engineering-429bc6",
#     "https://unienrol.com/c/universiti-sains-malaysia-usm-degree-in-electronic-engineering-429bc5",
#     "https://unienrol.com/c/universiti-sains-malaysia-usm-degree-in-electrical-engineering-429bc3",
#     "https://unienrol.com/c/universiti-sains-malaysia-usm-degree-in-civil-engineering-429bc8",
#     "https://unienrol.com/c/universiti-sains-malaysia-usm-degree-in-aeronautical-engineering-429bc7",
# ]


# usm_medicine_undergrad_program = [
#     "https://unienrol.com/c/universiti-sains-malaysia-usm-degree-in-medical-technology-429bcc",
#     "https://unienrol.com/c/universiti-sains-malaysia-usm-degree-in-sports-science-429bcf",
#     "https://unienrol.com/c/universiti-sains-malaysia-usm-degree-in-medical-technology-429bb8",
#     "https://unienrol.com/c/universiti-sains-malaysia-usm-degree-in-forensic-science-429bc9",
#     "https://unienrol.com/c/universiti-sains-malaysia-usm-degree-in-biomedical-sciences-429bcb",
#     "https://unienrol.com/c/universiti-sains-malaysia-usm-degree-in-nutrition-429bce",
#     "https://unienrol.com/c/universiti-sains-malaysia-usm-degree-in-pharmacy-429bc0",
#     "https://unienrol.com/c/universiti-sains-malaysia-usm-degree-in-health-studies-429bcd",
#     "https://unienrol.com/c/universiti-sains-malaysia-usm-degree-in-medical-technology-429bca",
#     "https://unienrol.com/c/universiti-sains-malaysia-usm-degree-in-medicine-323d52",
#     "https://unienrol.com/c/universiti-sains-malaysia-usm-degree-in-medicine-429bd1",
#     "https://unienrol.com/c/universiti-sains-malaysia-usm-degree-in-dentistry-0ed6b2",
#     "https://unienrol.com/c/universiti-sains-malaysia-usm-degree-in-dentistry-429bd2",
#     "",
    
# ]


# Asia Pacific University of Technology & Innovation (APU) Info
apu_info = [
    "https://unienrol.com/u/asia-pacific-university-of-technology-&-innovation-apu"
]

# Asia Pacific University of Technology & Innovation (APU) Undergrad Programs
apu_scholarships = [
    "https://unienrol.com/scholarships/details/21db62/apu-mdec-digital-ninja-scholarships",
    "https://unienrol.com/scholarships/details/12ec8d/apu-merit-award-degree",
    "https://unienrol.com/scholarships/details/1161a8/apu-100-merit-award",
    "https://unienrol.com/scholarships/details/4671f2/apu-east-malaysia-community-rebate",
    "https://unienrol.com/scholarships/details/0b4ca2/apu-sportsman-scholarships",
]


apu_accounting_finance_undergrad_program = [
    "https://unienrol.com/c/asia-pacific-university-of-technology-innovation-apu-degree-in-banking-and-finance-341b76",
    "https://unienrol.com/c/asia-pacific-university-of-technology-innovation-apu-degree-in-banking-and-finance-5d38af",
    "https://unienrol.com/c/asia-pacific-university-of-technology-innovation-apu-degree-in-accounting-and-finance-674a43",
    "https://unienrol.com/c/asia-pacific-university-of-technology-innovation-apu-degree-in-banking-and-finance-72d04e",
    "https://unienrol.com/c/asia-pacific-university-of-technology-innovation-apu-degree-in-accounting-and-finance-341b72",
    "https://unienrol.com/c/asia-pacific-university-of-technology-innovation-apu-degree-in-accounting-and-finance-72d045",
    "https://unienrol.com/c/asia-pacific-university-of-technology-innovation-apu-degree-in-accounting-and-finance-72d044",
    "https://unienrol.com/c/asia-pacific-university-of-technology-innovation-apu-degree-in-accounting-and-finance-72d043",

]


apu_computer_undergrad_program = [
    "https://unienrol.com/c/asia-pacific-university-of-technology-innovation-apu-degree-in-computer-science-341b75",
    "https://unienrol.com/c/asia-pacific-university-of-technology-innovation-apu-degree-in-information-technology-it-5d38a6",
    "https://unienrol.com/c/asia-pacific-university-of-technology-innovation-apu-degree-in-information-technology-it-341b74",
    "https://unienrol.com/c/asia-pacific-university-of-technology-innovation-apu-degree-in-information-technology-it-5d38a3",
    "https://unienrol.com/c/asia-pacific-university-of-technology-innovation-apu-degree-in-information-technology-it-5d38a2",
    "https://unienrol.com/c/asia-pacific-university-of-technology-innovation-apu-degree-in-computer-science-5d38a4",
    "https://unienrol.com/c/asia-pacific-university-of-technology-innovation-apu-degree-in-computer-science-5d38a5",
    "https://unienrol.com/c/asia-pacific-university-of-technology-innovation-apu-degree-in-software-engineering-72d05d",
    "https://unienrol.com/c/asia-pacific-university-of-technology-innovation-apu-degree-in-information-technology-it-72d059",
    "https://unienrol.com/c/asia-pacific-university-of-technology-innovation-apu-degree-in-information-technology-it-72d058",
    "https://unienrol.com/c/asia-pacific-university-of-technology-innovation-apu-degree-in-information-technology-it-72d057",
    "https://unienrol.com/c/asia-pacific-university-of-technology-innovation-apu-degree-in-information-technology-it-72d056",
    "https://unienrol.com/c/asia-pacific-university-of-technology-innovation-apu-degree-in-information-technology-it-72d055",
    "https://unienrol.com/c/asia-pacific-university-of-technology-innovation-apu-degree-in-information-technology-it-72d054",
    "https://unienrol.com/c/asia-pacific-university-of-technology-innovation-apu-degree-in-computer-science-72d053",
]


apu_engineering_undergrad_program = [
    "https://unienrol.com/c/asia-pacific-university-of-technology-innovation-apu-degree-in-mechanical-engineering-341b73",
    "https://unienrol.com/c/asia-pacific-university-of-technology-innovation-apu-degree-in-computer-systems-engineering-72d042",
    "https://unienrol.com/c/asia-pacific-university-of-technology-innovation-apu-degree-in-electronic-engineering-2cc5b0",
    "https://unienrol.com/c/asia-pacific-university-of-technology-innovation-apu-degree-in-petroleum-engineering-2cc5b1",
    "https://unienrol.com/c/asia-pacific-university-of-technology-innovation-apu-degree-in-mechatronics-engineering-2cc5af",
    "https://unienrol.com/c/asia-pacific-university-of-technology-innovation-apu-degree-in-electrical-electronic-engineering-2cc5ae",
]




# Taylor Info
taylor_info = [
    "https://unienrol.com/u/taylors-university"
]

# Taylor Info Undergrad Programs
taylor_scholarships = [
    "https://unienrol.com/scholarships/details/67fa92/taylor-s-university-alumni-bursary-undergraduate-programme",
    "https://unienrol.com/scholarships/details/20ecb2/taylor-s-education-group-teg-progression-bursary-acca-full-time-advanced-diploma-degree",
    "https://unienrol.com/scholarships/details/50c772/taylor-s-university-uec-prestige-award",
    "https://unienrol.com/scholarships/details/01b013/taylor-s-uwe-50-dual-award-scholarship",
    "https://unienrol.com/scholarships/details/54fb62/taylor-s-university-excellence-award-degree-pharmacy",
    "https://unienrol.com/scholarships/details/19c162/taylor-s-excellence-award-a-levels-3-subjects",
    "https://unienrol.com/scholarships/details/43b473/taylor-s-college-excellence-award-sace-international",
    "https://unienrol.com/scholarships/details/075d42/taylor-s-college-young-accountant-scholarship",
    "https://unienrol.com/scholarships/details/22ab52/taylor-s-excellence-award-acca",
    "https://unienrol.com/scholarships/details/138f22/taylor-s-college-acca-scholarship",
    "https://unienrol.com/scholarships/details/050802/taylor-s-university-community-scholarship",
    "https://unienrol.com/scholarships/details/3a1ee7/taylor-s-university-talent-scholarship-adp-diploma-advanced-diploma-degree",
    "https://unienrol.com/scholarships/details/74ab16/taylor-s-university-sports-scholarship-adp-diploma-advanced-diploma-degree",
    "https://unienrol.com/scholarships/details/3a1c93/taylor-s-university-merit-scholarship-adp-diploma-degree-advanced-diploma",
    "https://unienrol.com/scholarships/details/69a25e/taylor-s-university-excellence-award-all-degree-except-pharmacy",
    "https://unienrol.com/scholarships/details/77c713/taylor-s-excellence-award-a-levels-4subjects-adp",
    "https://unienrol.com/scholarships/details/69a252/taylor-s-college-excellence-award-foundation",
    "https://unienrol.com/scholarships/details/4c5164/taylor-s-college-talent-scholarship",
    "https://unienrol.com/scholarships/details/4e7e82/taylor-s-college-sports-scholarship",
    "https://unienrol.com/scholarships/details/36fba2/taylor-s-college-merit-scholarship",
]


taylor_accounting_finance_undergrad_program = [
    "https://unienrol.com/c/taylors-university-degree-in-accounting-and-finance-2ccc4d",
    "https://unienrol.com/c/taylors-university-degree-in-accounting-55364a",
    "https://unienrol.com/c/taylors-university-degree-in-accounting-200d42",
    "https://unienrol.com/c/taylors-university-degree-in-economics-and-finance-2ccc7b",
    "https://unienrol.com/c/taylors-university-degree-in-economics-and-finance-010b17",
    "https://unienrol.com/c/taylors-university-degree-in-banking-and-finance-2ccc64",
    "https://unienrol.com/c/taylors-university-degree-in-banking-and-finance-010b16",
    "https://unienrol.com/c/taylors-university-degree-in-professional-accounting-07eaa2", 
]


taylor_computer_undergrad_program = [
    "https://unienrol.com/c/taylors-university-degree-in-software-engineering-2ccd8e",
    "https://unienrol.com/c/taylors-university-degree-in-information-technology-it-010b1c",
    "https://unienrol.com/c/taylors-university-degree-in-computer-science-2ccd68",  
]


taylor_engineering_undergrad_program = [
    "https://unienrol.com/c/taylors-university-degree-in-electrical-electronic-engineering-19ced3",
    "https://unienrol.com/c/taylors-university-degree-in-mechanical-engineering-2ccd54",
    "https://unienrol.com/c/taylors-university-degree-in-electrical-electronic-engineering-2ccd43",
    "https://unienrol.com/c/taylors-university-degree-in-chemical-engineering-2ccd32"
]


taylor_medicine_undergrad_program = [
    "https://unienrol.com/c/taylors-university-degree-in-pharmaceutical-science-653ce2",
    "https://unienrol.com/c/taylors-university-degree-in-pharmacy-2ccdb4",
    "https://unienrol.com/c/taylors-university-degree-in-medicine-2cce0d",
    "https://unienrol.com/c/taylors-university-degree-in-biomedical-sciences-2ccdf0",    
]

# Combine all the URLs
URLS = []

URLS.extend(apu_info)
URLS.extend(apu_scholarships)
URLS.extend(apu_accounting_finance_undergrad_program)
URLS.extend(apu_computer_undergrad_program)
URLS.extend(apu_engineering_undergrad_program)

URLS.extend(taylor_info)
URLS.extend(taylor_scholarships)
URLS.extend(taylor_accounting_finance_undergrad_program)
URLS.extend(taylor_computer_undergrad_program)
URLS.extend(taylor_engineering_undergrad_program)
URLS.extend(taylor_medicine_undergrad_program)


# Load Data from the Web
loader = WebBaseLoader(URLS)
data = loader.load()


# Split the Data into Chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
websites_data = text_splitter.split_documents(data)


# Chroma save to disk
db = Chroma.from_documents(
    websites_data,
    OpenAIEmbeddings(),
    persist_directory="./university_courses_vector_db")