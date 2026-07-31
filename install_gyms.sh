#!/bin/bash

# UserRL Gym Installation Script
echo "🚀 Installing UserRL Gyms..."

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GYMS_DIR="$SCRIPT_DIR/gyms"
UTILS_DIR="$GYMS_DIR/utils"

# Create utils directory if it doesn't exist
mkdir -p "$UTILS_DIR"

# Install tau-bench first (required for TauGym)
echo "📦 Installing tau-bench..."
if [ ! -d "$UTILS_DIR/tau-bench" ]; then
    cd "$UTILS_DIR"
    git clone https://github.com/sierra-research/tau-bench.git
    cd tau-bench
    pip install -e .
    echo "✅ tau-bench installed"
else
    echo "✅ tau-bench already installed"
fi

# List of gyms to install
GYMS=(
    "AlfworldGym"
    "FunctionGym" 
    "IntentionGym"
    "PersuadeGym"
    "SearchGym"
    "SWEGym"
    "TauGym"
    "TelepathyGym"
    "TravelGym"
    "TurtleGym"
)

# Install each gym
for gym in "${GYMS[@]}"; do
    echo "📦 Installing $gym..."
    cd "$GYMS_DIR/$gym"
    pip install -e .
    echo "✅ $gym installed"
done

echo "🎉 All gyms installed successfully!"
echo "Run: python -c \"import alfworldgym, functiongym, intentiongym, persuadegym, searchgym, swegym, taugym, telepathygym, travelgym, turtlegym; print('All gyms imported successfully!')\""
