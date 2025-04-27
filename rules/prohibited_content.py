def validate(self, output: str, **kwargs) -> RuleResult:
    """
    Validate that the output does not contain prohibited content.

    Args:
        output (str): The text to validate
        **kwargs: Additional validation options

    Returns:
        RuleResult: The validation result

    Raises:
        ValueError: If output is None or not a string
    """
    if output is None:
        raise ValueError("Output cannot be None")
    if not isinstance(output, str):
        raise ValueError("Output must be a string")

    matches = []
    metadata = {"matches": matches}  # Initialize matches in metadata immediately

    if not output:  # Handle empty string case
        return RuleResult(passed=True, message="No prohibited content found", metadata=metadata)

    # Convert to lowercase if case-insensitive
    text = output.lower() if not self.case_sensitive else output

    # Find matches
    for term in self.prohibited_terms:
        term_to_check = term if self.case_sensitive else term.lower()
        if term_to_check in text:
            matches.append({"term": term, "index": text.index(term_to_check)})

    if matches:
        return RuleResult(passed=False, message="Found prohibited content", metadata=metadata)

    return RuleResult(passed=True, message="No prohibited content found", metadata=metadata)
