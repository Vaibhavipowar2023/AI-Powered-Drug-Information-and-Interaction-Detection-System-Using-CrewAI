import os
from crewai import LLM, Agent, Task, Crew
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

load_dotenv()

gemini_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.5,
)

interaction_agent = Agent(
    role="Drug Interaction Checker",
    backstory="You are an expert pharmacologist checking drug interactions.",
    goal="Determine the level of drug interactions accurately.",
    llm=gemini_llm,
    verbose=True
)

def check_drug_interaction(drugs: list[str]):
    prompt = f"Check if there are any interactions between: {', '.join(drugs)}. Provide explanation and severity."
    task = Task(
        description=prompt,
        expected_output="Interaction explanation and severity",
        agent=interaction_agent
    )
    crew = Crew(agents=[interaction_agent], tasks=[task])
    response = crew.kickoff()
    # Extract actual text from CrewOutput
    return response.text if hasattr(response, 'text') else str(response)

def map_interaction_text_to_class(text: str) -> str:
    text = text.lower()
    if "no interaction" in text or "no significant" in text:
        return "no_interaction"
    elif "severe" in text or "contraindicated" in text or "major" in text:
        return "severe_interaction"
    elif "mild" in text or "moderate" in text:
        return "mild_interaction"
    else:
        return "unknown"

def evaluate_classification(test_dataset):
    y_true = []
    y_pred = []

    for drugs, true_label in test_dataset:
        interaction_output = check_drug_interaction(drugs)
        interaction_text = interaction_output.text if hasattr(interaction_output, 'text') else str(interaction_output)
        pred_label = map_interaction_text_to_class(interaction_text)

        print(f"Drugs: {drugs}")
        print(f"Predicted: {pred_label}, True: {true_label}")
        print(f"Agent Response:\n{interaction_text}\n{'-'*40}")

        y_true.append(true_label)
        y_pred.append(pred_label)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")

if __name__ == "__main__":
    test_data = [
        (["warfarin", "ibuprofen"], "mild_interaction"),
        (["aspirin", "paracetamol"], "no_interaction"),
        (["digoxin", "quinidine"], "severe_interaction"),
        # Add more test pairs here
    ]

    evaluate_classification(test_data)
