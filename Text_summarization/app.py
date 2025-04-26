import streamlit as st
import validators
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# Page configuration
st.set_page_config(
    page_title="Langchain: Summarize Text from YT or Website",
    page_icon="👻💀👽👹"
)

st.title("👹 Langchain: Summarize Text from YouTube or Website")
st.subheader("Paste a URL and get a quick summary!")

# Sidebar for API key input
with st.sidebar:
    groq_api_key = st.text_input("Enter Groq API Key", value="", type="password")

# Main input for URL
generic_url = st.text_input("Enter YouTube or Website URL", label_visibility="collapsed")

# Setup LLM and prompt
llm = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)

prompt_template = """
Provide a summary of the following content in 300 words only:

Content: {text}
"""

prompt = PromptTemplate(input_variables=["text"], template=prompt_template)

# Extract YouTube video ID
def get_youtube_video_id(url):
    if "youtube.com" in url:
        return url.split("v=")[-1].split("&")[0]
    elif "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    return None

# Summarize button logic
if st.button("Summarize Now"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide both your API key and a valid URL.")
    elif not validators.url(generic_url):
        st.error("Invalid URL. Please check again.")
    else:
        try:
            with st.spinner("Processing..."):

                # If it's a YouTube URL
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    video_id = get_youtube_video_id(generic_url)
                    try:
                        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                    except (TranscriptsDisabled, NoTranscriptFound):
                        st.error("This YouTube video has no captions available. Try another video.")
                        st.stop()

                    # Join transcript into one string
                    text = " ".join([segment['text'] for segment in transcript_list])
                    docs = [Document(page_content=text)]

                # If it's a regular website
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        header={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                    )
                    docs = loader.load()

                # Run summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output = chain.run(docs)
                st.success("Summary generated successfully!")
                st.write(output)

        except Exception as e:
            st.error("An error occurred:")
            st.exception(e)
