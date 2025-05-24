#!/bin/bash

# Project cleanup script
# Removes old/redundant directories and organizes the project structure

set -e

echo "CARLA Project Cleanup Utility"
echo "============================"
echo ""

# Find and remove redundant directories
REDUNDANT_DIRS=(
    "training"
    "environments"
    "models"
    "rl_agents"
    "utils/data_logger.py"
    "logging"
    "setup"
)

echo "Checking for redundant directories..."

for dir in "${REDUNDANT_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        description=""
        case $dir in
            "training") description="Moved to app/training/" ;;
            "environments") description="Moved to app/environments/" ;;
            "models") description="Moved to app/models/" ;;
            "rl_agents") description="Moved to app/rl_agents/" ;;
            "logging") description="Replaced by app/config.py" ;;
            "setup") description="Replaced by scripts/" ;;
        esac
        echo "Found: $dir ($description)"
        
        read -p "Remove $dir? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$dir"
            echo "Removed $dir"
        fi
    fi
done

echo ""
echo "Checking for duplicate directories..."

# Check for both setup/ and scripts/ directories
if [ -d "setup" ] && [ -d "scripts" ]; then
    echo "Found both 'setup/' and 'scripts/' directories"
    echo "The 'setup/' directory is redundant as we now use 'scripts/'"
    
    read -p "Remove setup/ directory? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "setup"
        echo "Removed setup/ directory"
    fi
fi

echo ""
echo "Checking for empty directories..."

# Find and offer to remove empty directories (excluding necessary ones)
for empty_dir in $(find . -type d -empty -not -path "./.git*" -not -path "./data/*" -not -path "./models/*" 2>/dev/null); do
    echo "Empty directory: $empty_dir"
    
    read -p "Remove $empty_dir? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rmdir "$empty_dir" 2>/dev/null || echo "Could not remove $empty_dir"
    fi
done

echo ""
echo "Checking for Python cache directories..."

# Clean Python cache files
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

echo "Cleaned Python cache files"

echo ""
echo "Final project structure:"
tree -I '__pycache__|*.pyc|*.pyo|.git|.pytest_cache' -L 3 . 2>/dev/null || ls -la

echo ""
echo "Next steps:"
echo "1. Review the project structure"
echo "2. Update any import statements if needed"
echo "3. Test the application: ./scripts/run_local.sh test"
echo "4. Run the application: ./scripts/run_local.sh" 