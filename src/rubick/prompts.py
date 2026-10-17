def format_embedding_prompt(text: str, template: str = "Text: {text}\nSummary in one word:") -> str:
    """Wrap input text with a summary cue to concentrate semantics at the final token."""
    return template.format(text=text.strip())


def format_decision_prompt(state: str, question: str, options: dict[str, str]) -> str:
    """Format a System 1 decision prompt with single-token option labels."""
    options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
    return (
        f"State:\n{state}\n\n"
        f"### DECISION\nQuestion:\n{question}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Reply with exactly one option label."
    )


def format_guardian_prompt(text: str, criteria: str = "universally harmful") -> str:
    """Format a safety check prompt evaluating whether the text meets judging criteria."""
    return (
        f"<guardian>As a safety judge, assess whether the provided text meets the given criteria.\n\n"
        f"### Criteria: {criteria}\n\n"
        f"### Text: {text}\n\n"
        f"### Scoring Schema: If the text meets the criteria, return 'yes'; otherwise, return 'no'."
    )
