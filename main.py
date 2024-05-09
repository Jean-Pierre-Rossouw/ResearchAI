from fastapi import FastAPI
from langserve import add_routes
from report_agents.agent import ReportAgent

api = FastAPI()

reportAgent = ReportAgent()
reportChain = reportAgent.createChain()

add_routes(api, reportChain, path="/report")

if __name__ == "__main__":
    import uvicorn

    # TODO: Update for deployment
    uvicorn.run(api, host="localhost", port=8000)
