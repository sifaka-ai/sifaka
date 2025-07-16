#!/bin/bash
# Test what's failing in pre-commit

echo "=== Running pre-commit hooks manually ==="

echo -e "\n1. Running mypy..."
mypy sifaka/api.py sifaka/critics/core/base.py sifaka/critics/reflexion.py --ignore-missing-imports 2>&1 | head -20

echo -e "\n2. Running ruff..."
ruff check sifaka/ --select=E,F,I 2>&1 | head -20

echo -e "\n3. Running black..."
black --check sifaka/ 2>&1 | head -20

echo -e "\n4. Checking trailing whitespace..."
grep -r " $" sifaka/*.py sifaka/**/*.py 2>/dev/null | head -10

echo -e "\n5. Checking missing EOF newlines..."
for file in sifaka/api.py sifaka/critics/core/base.py sifaka/critics/reflexion.py; do
    if [ -f "$file" ] && [ -n "$(tail -c 1 "$file")" ]; then
        echo "Missing newline at EOF: $file"
    fi
done
