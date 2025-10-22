from agents.drug_info_agent import get_drug_info
from agents.interaction_agent import check_drug_interaction
from agents.summary_agent import generate_summary

def multi_agent_drug_interaction_system(drugs: list[str]):
    # Step 1: Get drug info and extract text from responses
    drug_infos = []
    for drug in drugs:
        response = get_drug_info(drug)
        drug_infos.append(response.text if hasattr(response, 'text') else str(response))

    # Step 2: Check interactions and extract text
    interaction_resp = check_drug_interaction(drugs)
    interaction_info = interaction_resp.text if hasattr(interaction_resp, 'text') else str(interaction_resp)

    # Step 3: Generate summary input
    combined_info = "\n\n".join(drug_infos)
    summary_resp = generate_summary(combined_info, interaction_info)
    summary = summary_resp.text if hasattr(summary_resp, 'text') else str(summary_resp)

    return summary

if __name__ == "__main__":
    drugs = input("Enter drug names separated by comma: ").split(",")
    drugs = [d.strip() for d in drugs]
    final_summary = multi_agent_drug_interaction_system(drugs)
    print("\nFinal Summary and Recommendations:\n", final_summary)
