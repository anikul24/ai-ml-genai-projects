import os
from dotenv import load_dotenv

from langchain_core.tools import tool
from serpapi import GoogleSearch


load_dotenv(dotenv_path="./cred.env")

SERP_API_KEY = os.environ.get("SERP_API_KEY")



@tool('web_search')
def web_search(query: str) -> str:
    """
        Use this tool to search the internet for up-to-date information that you do not know.
        Useful for finding:
        - Current 2024/2025 IRS contribution limits (401k, IRA).
        - Recent changes to Social Security or tax laws.
        - Current market trends or inflation rates.

    """

    if not SERP_API_KEY:
        return "Error: SERP_API_KEY is not set in the environment variables."

    params = {
        "engine": "google",
        "q": query,
        "api_key": SERP_API_KEY,
        'num': 5,  # Number of results to return
        "gl": "us",
        "hl": "en"
    }

    try:
        search = GoogleSearch(params)

        results = search.get_dict().get("organic_results", [])
        if not results:
            return "No results found."
        
        formatted_results = []

        for result in results:
            title = result.get("title", "No Title")
            snippet = result.get("snippet", "No Snippet")
            link = result.get("link", "No Link")
            formatted_results.append(f"Sorce: {title}\nSnippet: {snippet}\nLink: {link}\n")

        return "\n".join(formatted_results)
    except Exception as e:
        return f"An error occurred during web search: {str(e)}"