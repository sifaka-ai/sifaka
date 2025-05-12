#!/bin/bash

# Script to find all files that import from the refactored modules
# Usage: ./find_imports.sh

echo "Finding imports from sifaka.utils.config..."
grep -r "from sifaka.utils.config import" --include="*.py" . > config_imports.txt
echo "Found $(wc -l < config_imports.txt) files importing from sifaka.utils.config"

echo "Finding imports from sifaka.utils.errors..."
grep -r "from sifaka.utils.errors import" --include="*.py" . > errors_imports.txt
echo "Found $(wc -l < errors_imports.txt) files importing from sifaka.utils.errors"

echo "Finding imports from sifaka.core.dependency..."
grep -r "from sifaka.core.dependency import" --include="*.py" . > dependency_imports.txt
echo "Found $(wc -l < dependency_imports.txt) files importing from sifaka.core.dependency"

echo "Finding import statements with 'import sifaka.utils.config'..."
grep -r "import sifaka.utils.config" --include="*.py" . > config_imports_alt.txt
echo "Found $(wc -l < config_imports_alt.txt) files with 'import sifaka.utils.config'"

echo "Finding import statements with 'import sifaka.utils.errors'..."
grep -r "import sifaka.utils.errors" --include="*.py" . > errors_imports_alt.txt
echo "Found $(wc -l < errors_imports_alt.txt) files with 'import sifaka.utils.errors'"

echo "Finding import statements with 'import sifaka.core.dependency'..."
grep -r "import sifaka.core.dependency" --include="*.py" . > dependency_imports_alt.txt
echo "Found $(wc -l < dependency_imports_alt.txt) files with 'import sifaka.core.dependency'"

echo "Results saved to config_imports.txt, errors_imports.txt, dependency_imports.txt"
echo "and config_imports_alt.txt, errors_imports_alt.txt, dependency_imports_alt.txt"

# Create a summary file
echo "Creating summary file..."
echo "# Import Migration Summary" > import_migration_summary.txt
echo "" >> import_migration_summary.txt
echo "## Files to Update" >> import_migration_summary.txt
echo "" >> import_migration_summary.txt

echo "### Config Imports" >> import_migration_summary.txt
echo "" >> import_migration_summary.txt
cat config_imports.txt config_imports_alt.txt | sort | uniq | sed 's/^/- /' >> import_migration_summary.txt
echo "" >> import_migration_summary.txt

echo "### Errors Imports" >> import_migration_summary.txt
echo "" >> import_migration_summary.txt
cat errors_imports.txt errors_imports_alt.txt | sort | uniq | sed 's/^/- /' >> import_migration_summary.txt
echo "" >> import_migration_summary.txt

echo "### Dependency Imports" >> import_migration_summary.txt
echo "" >> import_migration_summary.txt
cat dependency_imports.txt dependency_imports_alt.txt | sort | uniq | sed 's/^/- /' >> import_migration_summary.txt

echo "Summary saved to import_migration_summary.txt"

# Make the script executable
chmod +x find_imports.sh
