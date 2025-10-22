import os
from crewai import LLM, Agent, Task, Crew
from dotenv import load_dotenv

load_dotenv()

gemini_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.5,
)

interaction_agent = Agent(
    role="Drug Interaction Checker",
    backstory="You are an expert pharmacologist who can quickly check drug interactions and explain simply.",
    goal="Check potential interactions between given drugs.",
    llm=gemini_llm,
    verbose=True
)

def check_drug_interaction(drugs: list[str]):
    drugs_str = ", ".join(drugs)
    prompt = f"Check if there are any interactions between the following drugs: {drugs_str}. Provide explanations and severity level."
    task = Task(
        description=prompt,
        expected_output="Explanation of drug interactions",
        agent=interaction_agent
    )
    crew = Crew(agents=[interaction_agent], tasks=[task])
    response = crew.kickoff()
    return response
