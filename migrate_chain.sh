#!/bin/bash

# Migration script to replace the old chain system with the new one

# Set the base directory
BASE_DIR="/Users/evanvolgas/Documents/not_beam/sifaka"
CHAIN_DIR="$BASE_DIR/sifaka/chain"
V2_DIR="$CHAIN_DIR/v2"
BACKUP_DIR="$BASE_DIR/sifaka/chain_old_backup"

# Create backup directory
echo "Creating backup directory: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# Backup old chain files
echo "Backing up old chain files..."
cp -r "$CHAIN_DIR" "$BACKUP_DIR"

# Remove old chain files and directories (excluding v2)
echo "Removing old chain files and directories..."
find "$CHAIN_DIR" -type f -not -path "*/v2/*" -name "*.py" -exec rm {} \;
rm -rf "$CHAIN_DIR/formatters"
rm -rf "$CHAIN_DIR/interfaces"
rm -rf "$CHAIN_DIR/managers"
rm -rf "$CHAIN_DIR/strategies"

# Create examples directory if it doesn't exist
echo "Creating examples directory..."
mkdir -p "$CHAIN_DIR/examples"

# Move new chain files from v2 to main chain directory
echo "Moving new chain files..."
cp "$V2_DIR/__init__.py" "$CHAIN_DIR/__init__.py"
cp "$V2_DIR/adapters.py" "$CHAIN_DIR/adapters.py"
cp "$V2_DIR/chain.py" "$CHAIN_DIR/chain.py"
cp "$V2_DIR/config.py" "$CHAIN_DIR/config.py"
cp "$V2_DIR/engine.py" "$CHAIN_DIR/engine.py"
cp "$V2_DIR/errors.py" "$CHAIN_DIR/errors.py"
cp "$V2_DIR/factories.py" "$CHAIN_DIR/factories.py"
cp "$V2_DIR/interfaces.py" "$CHAIN_DIR/interfaces.py"
cp "$V2_DIR/plugins.py" "$CHAIN_DIR/plugins.py"
cp "$V2_DIR/result.py" "$CHAIN_DIR/result.py"
cp "$V2_DIR/state.py" "$CHAIN_DIR/state.py"

# Copy README.md
cp "$V2_DIR/README.md" "$CHAIN_DIR/README.md"

# Copy examples
cp "$V2_DIR/examples/simple_chain.py" "$CHAIN_DIR/examples/simple_chain.py"

# Remove v2 directory
echo "Removing v2 directory..."
rm -rf "$V2_DIR"

echo "Migration completed successfully!"
