from smolagents import CodeAgent,DuckDuckGoSearchTool, InferenceClientModel,load_tool,tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
from tools.visit_webpage import VisitWebpageTool
from duckduckgo_search import DDGS

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

# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

final_answer = FinalAnswerTool()


model = InferenceClientModel (
max_tokens=2096,
temperature=0.5,
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',# it is possible that this model may be overloaded
custom_role_conversions=None,
)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[final_answer, web_search, image_generation_tool, visit_webpage], 
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates)


GradioUI(agent).launch()