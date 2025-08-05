from smolagents import CodeAgent, InferenceClientModel,load_tool,tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
from tools.visit_webpage import VisitWebpageTool
from duckduckgo_search import DDGS
from transformers import pipeline
import langdetect
import fitz  

from Gradio_UI import GradioUI

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"

@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return the top 5 summaries
    Args:
        query: topic used for the web search
    """
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)
        return "\n\n".join([r["body"] for r in results])

@tool
def visit_webpage(url: str) -> str:
    """A tool that visits a webpage and return its content as a formatted string 
    Args:
        url: url of the page to visit
    """
    visit_tool=VisitWebpageTool()
    return visit_tool.forward(url)

@tool
def summarize_document(text: str, max_length: int = 300) -> str:
    """Summarize a long document or text into a shorter summary.
    Args:
        text: the full text of the document to summarize
        max_length: max length of the summary
    """
    from transformers import pipeline
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
    return summary[0]['summary_text']


@tool
def extract_information_from_pdf(file_path: str, query: str) -> str:
    """Extract specific information from a PDF document based on a query.
    
    Args:
        file_path: path to the PDF file
        query: keyword or question to find in the document
    """
    try:
        doc = fitz.open(file_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()

        if query.lower() in full_text.lower():
            start = full_text.lower().index(query.lower())
            excerpt = full_text[max(0, start - 100):start + 300]
            return f"Found excerpt related to '{query}':\n\n{excerpt.strip()}"
        else:
            return f"No information found for '{query}'."

    except Exception as e:
        return f"Error reading PDF: {str(e)}"


@tool
def detect_language(text: str) -> str:
    """Detect the language of the input text.
    Args:
        text: text whose language should be detected
    """
    lang = langdetect.detect(text)
    return lang


@tool
def clean_text(text: str) -> str:
    """Clean and preprocess text by removing extra whitespace, newlines, etc.
    Args:
        text: raw text input
    """
    import re
    cleaned = re.sub(r'\s+', ' ', text).strip()
    return cleaned


final_answer = FinalAnswerTool()


model = InferenceClientModel (
max_tokens=2096,
temperature=0.5,
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
custom_role_conversions=None,
api_key="ADD YOUR API KEY"
)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[final_answer, web_search, visit_webpage, summarize_document, extract_information_from_pdf,
           detect_language, clean_text], 
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates)


GradioUI(agent).launch()