class PythonRule(Rule):
    def _validate_impl(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output contains valid Python code.
        """
        metadata = {"issues": []}

        # Skip validation if output is empty or whitespace
        if not output or output.isspace():
            return RuleResult(
                passed=True,
                message="Empty or whitespace-only output is valid Python",
                metadata=metadata,
            )

        try:
            # Try to parse the code
            ast.parse(output)

            # Check for common style issues
            lines = output.split("\n")
            for i, line in enumerate(lines, 1):
                # Check line length
                if len(line) > 120:
                    metadata["issues"].append(f"Line {i} exceeds 120 characters")

                # Check indentation (must be multiple of 4)
                leading_spaces = len(line) - len(line.lstrip())
                if leading_spaces % 4 != 0 and line.strip():
                    metadata["issues"].append(f"Line {i} has incorrect indentation")

            passed = len(metadata["issues"]) == 0
            message = "Python code validation " + ("passed" if passed else "failed")
            if not passed:
                message += ": " + "; ".join(metadata["issues"])

            return RuleResult(passed=passed, message=message, metadata=metadata)

        except SyntaxError as e:
            metadata["issues"].append(f"Syntax error: {str(e)}")
            return RuleResult(
                passed=False, message=f"Python code validation failed: {str(e)}", metadata=metadata
            )
        except Exception as e:
            metadata["issues"].append(f"Validation error: {str(e)}")
            return RuleResult(
                passed=False, message=f"Python code validation failed: {str(e)}", metadata=metadata
            )
