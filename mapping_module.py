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
