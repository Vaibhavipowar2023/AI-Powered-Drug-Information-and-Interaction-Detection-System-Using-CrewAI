import os
from crewai import LLM, Agent, Task, Crew
from dotenv import load_dotenv

load_dotenv()

gemini_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.5,
)

summary_agent = Agent(
    role="Summary and Recommendation Generator",
    backstory="You summarize drug information and interaction results in a clear, simple language.",
    goal="Provide a clear summary and recommendation based on drug information and interactions.",
    llm=gemini_llm,
    verbose=True
)

def generate_summary(drug_info: str, interaction_info: str):
    prompt = f"Summarize the following drug information:\n{drug_info}\n\nAnd the following interaction information:\n{interaction_info}\n\nProvide a simple, clear summary and recommendations for users."
    task = Task(
        description=prompt,
        expected_output="Summary of drug info and interactions",
        agent=summary_agent
    )
    crew = Crew(agents=[summary_agent], tasks=[task])
    response = crew.kickoff()
    return response
