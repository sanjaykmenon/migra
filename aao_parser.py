from datetime import date as dt, datetime
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator
import argparse
import fitz 
import instructor 
import nltk
import spacy
from typing import List
import lancedb
from lancedb.pydantic import LanceModel, vector
import pyarrow as pa
import json
import pandas as pd
from dotenv import load_dotenv
import os
import supabase
import uuid
import glob

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GPT_MODEL = os.getenv("GPT_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)


# Initialize the instructor client
instructor_client = instructor.from_openai(OpenAI())

#openai client
openai_client = OpenAI()

# Function to extract text from a PDF
def extract_text(pdf_path: str) -> str:
    """
    Extracts text from the PDF and returns it as a single string.
    """
    doc = fitz.open(pdf_path)
    full_text = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(full_text)

# Pydantic model for the initial summary
class InitialSummary(BaseModel):
    """
    This is an initial summary which should be long (15 - 20 sentences, ~400 words)
    yet highly non-specific, containing little information beyond the entities marked as missing.
    Use overly verbose languages and fillers (Eg. This article discusses) to reach ~400 words.
    """
    summary: str = Field(
        ...,
        description="This is a summary of the article provided which is descriptive and verbose. It should be roughly 400 words in length",
    )
    @field_validator('summary')
    def min_length(cls, summary: str) -> str:
        """
        Validates that the summary is at least 200 words long.
        """
        tokens = nltk.word_tokenize(summary)
        num_tokens = len(tokens)
        if num_tokens < 50:
            raise ValueError("The current summary is too short. Please make sure that you generate a new summary that is around 50 words long.")
        return summary
    
# class RewrittenSummary(BaseModel):
#     """

#     This is a new, denser summary of identical length which covers every entity
#     and detail from the previous summary plus the Missing Entities.

#     Guidelines
#     - Make every word count : Rewrite the previous summary to improve flow and make space for additional entities
#     - Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
#     - The new summary should be highly dense and concise yet self-contained, eg., easily understood without the Article.
#     - Make space with fusion, compression, and removal of uninformative phrases like "the article discusses"
#     - Missing entities can appear anywhere in the new summary

#     An Entity is a real-world object that's assigned a name - for example, a person, country a product or a book title.
#     """
#     summary: str = Field(
#         ...,
#         description="This is a new, denser summary of identical length which covers every entity and detail from the previous summary plus the Missing Entities. It should have a similar length ( ~ 300 words ) as the previous summary and should be easily understood without the Article",
#     )
#     absent: List[str] = Field(
#         ...,
#         default_factory=list,
#         description="this is a list of Entities found absent from the new summary that were present in the previous summary",
#     )
#     missing: List[str] = Field(
#         default_factory=list,
#         description="This is a list of 2-4 informative Entities from the Article that are missing from the new summary which should be included in the next generated summary.",
#     )

#     @field_validator("summary")
#     def min_length(cls, v: str) -> str:
#         """
#         Validates that the summary is at least 100 words long.
#         """
#         tokens = nltk.word_tokenize(v)
#         num_tokens = len(tokens)
#         if num_tokens < 50:
#             raise ValueError("The current summary is too short. Please make sure that you generate a new summary that is around 300 words long.")
#         return v

#     @field_validator("summary")
#     def min_entity_density(cls, v: str) -> str:
#         """
#         Validates that the summary has a sufficient entity density.
#         """
#         tokens = nltk.word_tokenize(v)
#         num_tokens = len(tokens)
#         doc = nlp(v)
#         num_entities = len(doc.ents)
#         density = num_entities / num_tokens
#         if density < 0.02:
#             raise ValueError(f"The summary of {v} has too few entities. Please regenerate a new summary with more new entities added to it. Remember that new entities can be added at any point of the summary.")
#         return v


class DocumentInfo(BaseModel):
    title: str = Field(..., description="The title of the document with specific details")
    beneficiary_details: List[str] = Field(..., description="extract details of beneficiary such as where they are from, what did they do and anything else that can be used to uniquely identify them")
    beneficiary_status: str = Field(..., description="extract details of the type of visa / status")
    key_reasons: List[str] = Field(..., description="extract detailed reasons explaining the evidence that was presented for the petition and corresponding reasons why the petition was denied or approved or any other decision was made")
    summary: List[str] = Field(..., description="add details of entites, people, locations and any other specific detail")
    date_of_application: dt = Field(..., description="extract date of the document")
    summary_embedding: List[float] = Field(..., description="OpenAI embedding of the summary")
    footnotes: List[str] = Field(..., description="extract all footnotes mentioned in the text.")
    cfr_code: List[str] = Field(..., description="Extract all the sections of the U.S. Code and the Code of Federal Regulations mentioned in the text. ")
    full_text: str = Field(..., description="The full text of the document")

    summary: str = None
    full_text: str = None

    def set_full_text(self, full_text: str):
        self.full_text = full_text
    
    def set_summary(self, summary: str):
        self.summary = summary

    def set_summary_embedding(self, summary_embedding: List[float]):
        self.summary_embedding = summary_embedding

# Load the Spacy model
nlp = spacy.load("en_core_web_sm")

def summarize_article(article: str, beneficiary_details: List[str], key_reasons: List[str], summary_steps: int = 1):
    summary_chain = []

    # Incorporate beneficiary details and key reasons into the article
    enhanced_article = f"{article}\n\nBeneficiary Details: {' '.join(beneficiary_details)}\n\nKey Reasons: {' '.join(key_reasons)}"

    #generating initial summary
    summary: InitialSummary = instructor_client.chat.completions.create(
        model=GPT_MODEL,
        response_model=InitialSummary,
        messages=[
            {
                "role": "system",
                "content": "You are an expert in immigration law. Write a concise summary of the following document that includes information such as the beneficiary details, relevant evidence presented and the corresponding reasons why the petition was denied or approved or any other decision was made. The summary should be compact but still cover all details from the original document, including new entities to ensure it is rich in information."
            },
            {
                "role":"user", "content": f"Here is the Article: {article}, Beneficiary Details: {beneficiary_details}, Key Reasons: {key_reasons}"},
            {
                "role":"user",
                "content": "The generated summary should be about 450 words",
            },
        ],
        max_retries=2,
    )
    summary_chain.append(summary.summary)
    print(f"Initial Summary:\n{summary.summary}\n")
    return summary_chain[-1]

# def summarize_article(article: str, summary_steps: int = 1):
#     summary_chain = []

#     #generating initial summary
#     summary: InitialSummary = instructor_client.chat.completions.create(
#         model=GPT_MODEL,
#         response_model=InitialSummary,
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are an expert in immigration law. Write a concise summary of the following document that includes key details such as relevant laws, legal terms, people, locations, and institutions. The summary should be compact but still cover all details from the original document, including new entities to ensure it is rich in information and meets our entity density requirement."
#             },
#             {
#                 "role":"user", "content": f"Here is the Article: {article}"},
#             {
#                 "role":"user",
#                 "content": "The generated summary should be about 300 words",
#             },
#         ],
#         max_retries=2,
#     )
#     summary_chain.append(summary.summary)
#     print(f"Initial Summary:\n{summary.summary}\n")
#     return summary_chain[-1]


def get_structured_output(text: str):
    """
    Uses the patched OpenAI client to get structured output as JSON.
    """
    response = instructor_client.chat.completions.create(
        model="gpt-4o-2024-05-13",
        response_model=DocumentInfo,  
        messages=[
            {"role": "user", "content": f"As an immigration attorney, your task is to extract all relevant entities from this document. Your goal is to provide as much detail with specificity about this document as you can. \n{text}"}
        ]
    )
    return response


def generate_openai_embedding(text: str):
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def main(pdf_folder: str):
    #get list of all PDF files in the folder
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))

    for pdf_file in pdf_files:

        text = extract_text(pdf_file)
        document_info = get_structured_output(text)
        document_summary = summarize_article(text, document_info.beneficiary_details, document_info.key_reasons)
        document_info.set_summary(document_summary)
        document_info.set_full_text(text)
        embedding = generate_openai_embedding(document_summary)
        document_info.summary_embedding = embedding
        #convert to a dictiionary
        document_info_dict = document_info.model_dump()

        document_info_dict['id'] = str(uuid.uuid4())
        document_info_dict['created_at'] = datetime.now().isoformat()
        #document_info_dict['date_of_application']  = datetime.strptime(document_info_dict['date_of_application'], '%Y-%m-%d')
        document_info_dict['date_of_application'] = document_info_dict['date_of_application'].isoformat() #do we need 2 steps?
        #document_info_dict['beneficiary_details'] = [document_info_dict['beneficiary_details']]


        try:
            db_table_write = supabase.table("eb1_case_data").insert(document_info_dict).execute()

        except ValueError as e:
            print(f"Error inserting data: {e}")

            print("-----------------")
            print("Beneficiary Details")
            print(document_info_dict.beneficiary_details)
            print("-----------------")
            print("Key Reasons")
            print(document_info_dict.key_reasons)
            print("-----------------")
            print("Summary")
            print(document_info_dict.summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and process PDF text into structured JSON data.")
    parser.add_argument('pdf_folder', type=str, help='Path to the PDF folder')
    args = parser.parse_args()
    main(args.pdf_folder)