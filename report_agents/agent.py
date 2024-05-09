from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser
import requests
from bs4 import BeautifulSoup
from langchain.schema.runnable import RunnablePassthrough
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
import json
from typing import List
from .templates import WRITER_SYSTEM_PROMPT, SUMMARY_TEMPLATE, RESEARCH_REPORT_TEMPLATE
import dotenv
import os

dotenv.load_dotenv(dotenv.find_dotenv(".env"))

class ReportAgent:
    def __init__(self):
        self.LLM = GoogleGenerativeAI(
            api_key=os.environ.get("GOOGLE_API_KEY"),
            model="gemini-pro",
            temperature=0,
            safety_settings=None,
        )
        self.search = DuckDuckGoSearchAPIWrapper()
        self.results_per_question = 3
        self.summary_prompt = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)
        self.search_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "user",
                    "Write 3 google search queries to search online that form an "
                    "objective opinion from the following: {question}\n"
                    "You must respond with a list of strings in the following format: "
                    '["query 1", "query 2", "query 3"].',
                ),
            ]
        )

    def _webSearch(self, query: str) -> List[str]:
        try:
            results = self.search.results(query, self.results_per_question)
            return [result.link for result in results]
        except Exception as e:
            return []

    def _scrapeText(self, url: str) -> str:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                return soup.get_text(separator=" ", strip=True)
        except Exception as e:
            return f"Failed to retrieve the webpage: {e}"

    def _collapseListOfLists(self, lol: List[List[str]]) -> str:
        content = ["\n\n".join(list) for list in lol]
        return "\n\n".join(content)
    
    def _createScrapeAndSummarizeChain(self):
        return RunnablePassthrough.assign(
            summary = RunnablePassthrough.assign(
                text = lambda x: self._scrapeText(x["url"])
            )
            | self.summary_prompt
            | self.LLM
            | StrOutputParser()
        ) | (lambda x: f"URL: {x["url"]}\n\nSUMMARY: {x["summary"]}")
        
    def _createWebSearchChain(self, scrape_and_summarize_chain):
        return (
            RunnablePassthrough.assign(urls=lambda x: self._webSearch(x["question"]))
            | (lambda x: [{"question": x["question"], "url": url} for url in x["urls"]])
            | scrape_and_summarize_chain.map()
        )
    
    def _createSearchQuestionChain(self):
        return self.search_prompt | self.LLM | StrOutputParser() | json.loads
    
    def _createFullResearchChain(self, search_question_chain, web_search_chain):
        return (
            search_question_chain
            | (lambda x: [{"question": q} for q in x])
            | web_search_chain.map()
        )
        
    def createChain(self):
        scrape_and_summarize_chain = self._createScrapeAndSummarizeChain()
        web_search_chain = self._createWebSearchChain(scrape_and_summarize_chain)
        search_question_chain = self._createSearchQuestionChain()
        full_research_chain = self._createFullResearchChain(
            search_question_chain, web_search_chain
        )

        prompt = ChatPromptTemplate.from_messages(
            [("system", WRITER_SYSTEM_PROMPT), ("user", RESEARCH_REPORT_TEMPLATE)]
        )

        return (
            RunnablePassthrough.assign(
                research_summary=full_research_chain | self._collapseListOfLists
            )
            | prompt
            | self.LLM
            | StrOutputParser()
        )
        
