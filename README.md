# AI-Agentic-RAG-System

This project is a **Retrieval-Augmented Generation (RAG) system** powered by an autonomous AI agent that uses modular tools to interact with diverse information sources. It leverages [Hugging Face’s `smolagents`](https://github.com/huggingface/smol-agents) library to enable reasoning and tool usage, allowing the agent to perform tasks such as retrieving web data, reading documents, or extracting relevant information.

## Key Features

- **Web Search Tool** – Search the web using DuckDuckGo and retrieve top summaries.
- **Webpage Reader** – Visit and extract content from live webpages.
- **PDF Extraction Tool** – Read and extract context-specific information from local PDF files.
- **Document Summarization Tool** – Summarize long documents using language models.
- **Agent-Oriented Architecture** – Based on `smolagents`, which allows flexible chaining of tools and reasoning steps.
- **Easy to Extend** – Tools are modular and can be customized or expanded for more use cases.
- **Gradio User Interface** – Complete Gradio UI for interacting with the agent in a user-friendly way. 

## Usage

To get started, open the `app.py` file and replace the `api_key` field inside the `InferenceClientModel` with your own Hugging Face API key (line 116). Then, run `app.py` to launch the Gradio User Interface. This will open a web-based chatbot where you can interact with the AI agent, ask questions, upload documents, and see it use tools such as web search, document summarization, and PDF reading.
