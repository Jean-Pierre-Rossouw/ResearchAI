from fastapi import FastAPI
from langserve import add_routes
from report_agents.agent import ReportAgent

app = FastAPI()

reportAgent = ReportAgent()
reportChain = reportAgent.createChain()

add_routes(app, reportChain, path="/report")
