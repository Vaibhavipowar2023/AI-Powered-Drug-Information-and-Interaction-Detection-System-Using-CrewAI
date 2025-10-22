import os
from crewai import LLM, Agent, Task, Crew
from dotenv import load_dotenv

load_dotenv()

# Initialize Gemini LLM
gemini_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.5,
)

drug_info_agent = Agent(
    role="Drug Information Provider",
    backstory="You are an expert pharmacologist providing detailed drug info.",
    goal="Retrieve detailed information about a specified drug.",
    llm=gemini_llm,
    verbose=True
)

def get_drug_info(drug_name: str):
    prompt = f"Provide detailed and simple-to-understand information about the drug {drug_name} including uses, side effects, and dosage."
    task = Task(
        description=prompt,
        expected_output="Detailed drug information",
        agent=drug_info_agent
    )
    crew = Crew(agents=[drug_info_agent], tasks=[task])
    response = crew.kickoff()
    return response
