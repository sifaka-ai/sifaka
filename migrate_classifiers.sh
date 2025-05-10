#!/bin/bash

# Migration script to replace the old classifiers system with the new one

# Set the base directory
BASE_DIR="/Users/evanvolgas/Documents/not_beam/sifaka"
CLASSIFIERS_DIR="$BASE_DIR/sifaka/classifiers"
V2_DIR="$CLASSIFIERS_DIR/v2"
BACKUP_DIR="$BASE_DIR/sifaka/classifiers_old_backup"

# Create backup directory
echo "Creating backup directory: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# Backup old classifiers files
echo "Backing up old classifiers files..."
cp -r "$CLASSIFIERS_DIR" "$BACKUP_DIR"

# Remove old classifiers files and directories (excluding v2)
echo "Removing old classifiers files and directories..."
find "$CLASSIFIERS_DIR" -type f -not -path "*/v2/*" -name "*.py" -exec rm {} \;
rm -rf "$CLASSIFIERS_DIR/interfaces"
rm -rf "$CLASSIFIERS_DIR/managers"
rm -rf "$CLASSIFIERS_DIR/strategies"

# Create examples directory if it doesn't exist
echo "Creating examples directory..."
mkdir -p "$CLASSIFIERS_DIR/examples"

# Move new classifiers files from v2 to main classifiers directory
echo "Moving new classifiers files..."
cp "$V2_DIR/__init__.py" "$CLASSIFIERS_DIR/__init__.py"
cp "$V2_DIR/adapters.py" "$CLASSIFIERS_DIR/adapters.py"
cp "$V2_DIR/classifier.py" "$CLASSIFIERS_DIR/classifier.py"
cp "$V2_DIR/config.py" "$CLASSIFIERS_DIR/config.py"
cp "$V2_DIR/engine.py" "$CLASSIFIERS_DIR/engine.py"
cp "$V2_DIR/errors.py" "$CLASSIFIERS_DIR/errors.py"
cp "$V2_DIR/factories.py" "$CLASSIFIERS_DIR/factories.py"
cp "$V2_DIR/interfaces.py" "$CLASSIFIERS_DIR/interfaces.py"
cp "$V2_DIR/plugins.py" "$CLASSIFIERS_DIR/plugins.py"
cp "$V2_DIR/result.py" "$CLASSIFIERS_DIR/result.py"
cp "$V2_DIR/state.py" "$CLASSIFIERS_DIR/state.py"

# Copy README.md
cp "$V2_DIR/README.md" "$CLASSIFIERS_DIR/README.md"

# Copy examples
cp "$V2_DIR/examples/simple_classifier.py" "$CLASSIFIERS_DIR/examples/simple_classifier.py"

# Remove v2 directory
echo "Removing v2 directory..."
rm -rf "$V2_DIR"

echo "Migration completed successfully!"
