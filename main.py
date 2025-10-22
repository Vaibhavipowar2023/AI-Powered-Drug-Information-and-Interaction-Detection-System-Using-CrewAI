from agents.drug_info_agent import DrugInfoAgent
from agents.interaction_agent import InteractionAgent
from agents.summary_agent import SummaryAgent

def run_drug_interaction_checker(drug1, drug2):
    # Initialize agents
    drug_info_agent = DrugInfoAgent()
    interaction_agent = InteractionAgent()
    summary_agent = SummaryAgent()

    # Step 1: Get drug info
    d1_info = drug_info_agent.run(drug1)
    d2_info = drug_info_agent.run(drug2)

    # Step 2: Analyze interaction
    interaction_data = interaction_agent.run(d1_info, d2_info)

    # Step 3: Summarize
    summary = summary_agent.run(interaction_data)

    # Output
    print("\n--- Drug 1 Info ---\n", d1_info)
    print("\n--- Drug 2 Info ---\n", d2_info)
    print("\n--- Interaction Report ---\n", interaction_data["interaction"])
    print("\n--- Severity ---\n", interaction_data["severity"])
    print("\n--- Final Summary ---\n", summary)

if __name__ == "__main__":
    d1 = input("Enter first drug: ")
    d2 = input("Enter second drug: ")
    run_drug_interaction_checker(d1, d2)
