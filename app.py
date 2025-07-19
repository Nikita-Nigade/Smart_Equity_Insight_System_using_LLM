import os
import streamlit as st
import time
import base64
import datetime
# from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# st.set_page_config(page_title="MarketMentor AI", layout="wide")
# Load environment variables
# load_dotenv()
api_key = st.secrets["OPENAI_API_KEY"]
# Initialize session state variables
if 'history' not in st.session_state:
    st.session_state.history = []

if 'documents' not in st.session_state:
    st.session_state.documents = []

# Add new session state variables to maintain UI state across tabs
if 'current_answer' not in st.session_state:
    st.session_state.current_answer = None

if 'current_sources' not in st.session_state:
    st.session_state.current_sources = None

if 'current_question' not in st.session_state:
    st.session_state.current_question = None

if 'is_uncertain' not in st.session_state:
    st.session_state.is_uncertain = False

if 'summaries_generated' not in st.session_state:
    st.session_state.summaries_generated = False

if 'article_summaries' not in st.session_state:
    st.session_state.article_summaries = []

if 'all_summaries_text' not in st.session_state:
    st.session_state.all_summaries_text = ""

# Store complete Q&A history in a single string for export
if 'qa_export_content' not in st.session_state:
    st.session_state.qa_export_content = ""

st.title("Information Retrieval System")
st.sidebar.title("Enter News Article URLs")

# Take 2 URLs from the user
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(2)]

process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()

# Use GPT-4o Mini
llm = ChatOpenAI(temperature=0.7, max_tokens=500, model_name="gpt-4o-mini-2024-07-18")

# Define summarization prompt
summarization_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    Please provide a comprehensive summary of the following article:

    {text}

    Focus on the key points, main arguments, and important details.
    The summary should be well-structured and capture the essence of the article.
    """
)

# Define improved QA prompt that handles uncertainty better
# This prompt will now directly use the raw extracted article texts
qa_prompt = PromptTemplate(
    input_variables=["question", "article_texts"],
    template="""
    You are an information retrieval assistant. Your task is to answer user questions or provide summaries based on the provided article texts.

    Article Texts:
    {article_texts}

    ---

    User Query: {question}

    Instructions:
    1. If the user's query is explicitly asking for a summary (e.g., "summarize this", "give me a summary", "what is this article about"), provide a concise summary of the main points from the "Article Texts" above.
    2. If the user's query is a question seeking specific information, answer it directly using only the information present in the "Article Texts".
    3. If you cannot find the answer to the question, or if you don't have enough information to provide a good summary or answer, clearly state: "I don't have enough information to answer this question accurately from the provided articles" or "I cannot provide a comprehensive summary based on the provided articles" and explain why.
    4. Do not make up information.
    5. At the end of your answer, include a line like: *Confidence Score (0-100):* [Your confidence score here, e.g., 85] based on how relevant and complete the provided context was to answer the query.
    """
)

# Function to summarize text
def summarize_document(text):
    chain = LLMChain(llm=llm, prompt=summarization_prompt)
    result = chain.run(text=text)
    return result

# Function to set background image for main content area with opacity control
def set_main_bg(main_bg_image, opacity):
    """
    A function to set the background image of the main content area with opacity control

    Parameters:
    main_bg_image (str): Path to the background image file for main content
    opacity (float): Opacity value between 0.0 (fully transparent) and 1.0 (fully opaque)
    """
    main_bg_ext = main_bg_image.split('.')[-1]

    with open(main_bg_image, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0,0,0,{opacity}), rgba(0,0,0,{opacity})),
                                url("data:image/{main_bg_ext};base64,{encoded_string}");
            background-size: cover;
            background-position: left left center;
            background-repeat: no-repeat;
        }}
        [data-testid="stHeader"]{{
            background-color: rgba(0,0,0,0);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to set background image for sidebar with opacity control
def set_sidebar_bg(sidebar_bg_image, opacity):
    """
    A function to set the background image of the sidebar with opacity control

    Parameters:
    sidebar_bg_image (str): Path to the background image file for sidebar
    opacity (float): Opacity value between 0.0 (fully transparent) and 1.0 (fully opaque)
    """
    sidebar_bg_ext = sidebar_bg_image.split('.')[-1]

    with open(sidebar_bg_image, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"]::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: url(data:image/{sidebar_bg_ext};base64,{encoded_string});
            background-size: cover;
            background-position: center center;
            opacity: {opacity};
            z-index: -1;
        }}

        /* Remove white background from sidebar */
        [data-testid="stSidebar"] > div:first-child {{
            background-color: transparent;
        }}

        /* Improve text readability in sidebar */
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3, [data-testid="stSidebar"] .stMarkdown {{
            text-shadow: 1px 1px 2px rgba(0,0,0,0.7); /* Added subtle text shadow for better contrast */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set paths to your background images
main_bg_image = "main_background.jpg"
sidebar_bg_image = "sidebar_background.jpg"
try:
    set_main_bg(main_bg_image, opacity=0.9)
    set_sidebar_bg(sidebar_bg_image, opacity=0.2)
except FileNotFoundError:
    st.warning("Background image file(s) not found. Please check the file paths.")

# Function to rebuild the export content based on current history
def rebuild_qa_export_content():
    export_content = "# Question & Answer History\n\n"

    # Iterate through history in reverse to show newest first in the exported file
    for item in reversed(st.session_state.history):
        if "question" in item:
            timestamp = item.get("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            export_content += f"## Question: {item['question']}\n"
            export_content += f"Timestamp: {timestamp}\n\n"
            export_content += f"### Answer:\n{item['answer']}\n\n"

            if item.get('sources'):
                export_content += f"### Sources:\n{item['sources']}\n\n"

            export_content += "-" * 50 + "\n\n"

    st.session_state.qa_export_content = export_content


# Process URLs: load, and store
if process_url_clicked:
    valid_urls = [url.strip() for url in urls if url.strip()]
    if valid_urls:
        main_placeholder.text("Loading data from URLs...")
        loader = UnstructuredURLLoader(urls=valid_urls)
        data = loader.load()

        # Store the full documents for both summarization and direct QA
        st.session_state.documents = data

        # Reset summaries and QA related states
        st.session_state.summaries_generated = False
        st.session_state.article_summaries = []
        st.session_state.all_summaries_text = ""
        st.session_state.current_answer = None
        st.session_state.current_sources = None
        st.session_state.current_question = None
        st.session_state.is_uncertain = False

        main_placeholder.text("Processing complete. Sources added:")
        for url in valid_urls:
            st.sidebar.success(f"Added: {url}")
        time.sleep(1)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.history.append({
            "timestamp": timestamp,
            "action": "Processed URLs",
            "urls": valid_urls
        })
    else:
        st.sidebar.error("Please enter at least one valid URL.")

# Create tabs for Q&A and Summarization
tab1, tab2 = st.tabs(["Question & Answer", "Summarization"])

# Question and Answer tab
with tab1:
    query = st.text_input("Ask a question about these articles:", key="question_input")

    if query:
        st.session_state.current_question = query

        if st.session_state.documents:
            # Concatenate all document content for the LLM to process
            # It's important to be mindful of token limits here.
            # For very long documents, you might need a more sophisticated approach
            # like retrieval-augmented generation (RAG) with a small, relevant chunk size,
            # but without FAISS, we're doing a direct LLM call.
            # For now, we'll join the page_content of all documents.
            all_article_content = "\n\n".join([doc.page_content for doc in st.session_state.documents])

            # Optionally, if the combined content is too large, you could chunk it
            # and only pass relevant chunks or a selection. For simplicity,
            # given the no-FAISS constraint, we'll pass the combined content.
            # Be aware of the LLM's context window limits.

            qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
            with st.spinner("Finding answer..."):
                result = qa_chain.run(question=query, article_texts=all_article_content)

            st.session_state.current_answer = result
            # The sources are the URLs of the processed articles
            st.session_state.current_sources = "\n".join([doc.metadata.get('source', 'Unknown source') for doc in st.session_state.documents])

            uncertainty_phrases = [
                "i don't have enough information", "i don't know", "i cannot", "i can't",
                "not enough information", "uncertain", "insufficient",
                "no information", "unable to determine"
            ]

            st.session_state.is_uncertain = any(phrase in st.session_state.current_answer.lower()
                                                for phrase in uncertainty_phrases)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            history_item = {
                "timestamp": timestamp,
                "question": query,
                "answer": st.session_state.current_answer,
                "sources": st.session_state.current_sources,
                "uncertain": st.session_state.is_uncertain
            }
            st.session_state.history.append(history_item)

            rebuild_qa_export_content()
        else:
            st.error("Please process URLs first to retrieve information.")

    if st.session_state.current_answer:
        st.header("Answer")

        if st.session_state.is_uncertain:
            st.warning(st.session_state.current_answer)
        else:
            st.write(st.session_state.current_answer)

        if st.session_state.current_sources:
            st.subheader("Sources:")
            for source in st.session_state.current_sources.split("\n"):
                if source.strip():
                    st.write(source)

        if st.session_state.qa_export_content:
            st.download_button(
                label="Download All Q&A Results",
                data=st.session_state.qa_export_content,
                file_name=f"qa_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

# Summarization tab
with tab2:
    if st.session_state.documents:
        if not st.session_state.summaries_generated:
            summarize_button = st.button("Generate Summaries")

            if summarize_button:
                st.session_state.article_summaries = []
                st.session_state.all_summaries_text = "# Article Summaries\n\n"

                for i, doc in enumerate(st.session_state.documents):
                    with st.spinner(f"Summarizing article {i+1} from {doc.metadata.get('source', 'Unknown source')}..."):
                        summary = summarize_document(doc.page_content)

                        st.session_state.article_summaries.append({
                            "index": i+1,
                            "source": doc.metadata.get('source', 'Unknown source'),
                            "summary": summary
                        })

                        st.session_state.all_summaries_text += f"## Summary of Article {i+1} - {doc.metadata.get('source', 'Unknown source')}\n\n"
                        st.session_state.all_summaries_text += f"{summary}\n\n"
                        st.session_state.all_summaries_text += "-" * 50 + "\n\n"

                st.session_state.summaries_generated = True

                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.history.append({
                    "timestamp": timestamp,
                    "action": "Generated Summaries",
                    "content": st.session_state.all_summaries_text
                })

        if st.session_state.summaries_generated:
            st.header("Article Summaries")

            for summary_item in st.session_state.article_summaries:
                with st.expander(f"Summary of Article {summary_item['index']} - {summary_item['source']}"):
                    st.write(summary_item['summary'])

            st.download_button(
                label="Download All Summaries",
                data=st.session_state.all_summaries_text,
                file_name=f"article_summaries_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

            # if st.button("Regenerate Summaries"):
            #     st.session_state.summaries_generated = False
            #     st.session_state.article_summaries = []
            #     st.session_state.all_summaries_text = ""
            #     st.experimental_rerun()
    else:
        st.info("Please process URLs first to generate summaries.")
