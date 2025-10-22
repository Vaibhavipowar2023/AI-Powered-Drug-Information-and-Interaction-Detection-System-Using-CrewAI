import os
from crewai import LLM, Agent, Task, Crew
from dotenv import load_dotenv

load_dotenv()  # Load GEMINI_API_KEY from .env

# Initialize Gemini LLM
gemini_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.5,
)

# Define the drug interaction checking agent
drug_interaction_agent = Agent(
    role="Drug Interaction Checker",
    backstory="You are an expert pharmacologist who can quickly check drug interactions and explain simply.",
    goal="Check potential interactions between given drugs",
    llm=gemini_llm,
    verbose=True
)


def check_drug_interaction(drug1: str, drug2: str):
    prompt = f"Check if there is any interaction between {drug1} and {drug2}. Provide a simple explanation."

    # Create Task with the prompt
    task = Task(
        description=prompt,
        expected_output="Explanation of drug interaction",
        agent=drug_interaction_agent
    )

    # Create Crew with agent and task
    crew = Crew(
        agents=[drug_interaction_agent],
        tasks=[task]
    )

    # Run the task and get response
    response = crew.kickoff()

    return response


if __name__ == "__main__":
    print("Simple Drug Interaction Checker")
    d1 = input("Enter first drug name: ")
    d2 = input("Enter second drug name: ")
    result = check_drug_interaction(d1, d2)
    print("\nInteraction Result:\n", result)
