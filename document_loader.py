import bs4
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

from langchain_community.document_loaders import JSONLoader
import json
from pathlib import Path
from pprint import pprint

# loader = JSONLoader(
#     file_path='./data_sources/json_source/books.json',
#     jq_schema='.',
#     text_content=False)
#
# data = loader.load()
# pprint(data)

# load csv
csv_loader = CSVLoader(file_path='./data_sources/csv/books.csv')
csv_docs = csv_loader.load()

# load pdfs
pdf_loader = PyPDFDirectoryLoader("data_sources/pdf_source/")
pdf_docs = pdf_loader.load()

# Create text splitter with smaller chunk size
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)

# Split documents into smaller chunks
split_csv_docs = text_splitter.split_documents(csv_docs)
split_pdf_docs = text_splitter.split_documents(pdf_docs)

# Combine all documents
split_docs = split_csv_docs + split_pdf_docs

# Export split_docs for use in other modules
__all__ = ['split_docs']

# load text on web
web_loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
web_docs = web_loader.load()

# split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(pdf_docs) + text_splitter.split_documents(web_docs) + csv_docs
