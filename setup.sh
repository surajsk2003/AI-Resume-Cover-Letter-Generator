#!/bin/bash

# AI Resume & Cover Letter Generator - Setup Script
echo "🤖 AI Resume & Cover Letter Generator Setup"
echo "=========================================="

# Check Python version
echo "🐍 Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check pip
echo "📦 Checking pip..."
pip3 --version
if [ $? -ne 0 ]; then
    echo "❌ pip not found. Please install pip."
    exit 1
fi

# Install requirements
echo "⬇️ Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully!"
else
    echo "❌ Failed to install dependencies. Check the error messages above."
    exit 1
fi

echo ""
echo "🎉 Setup complete! You can now run:"
echo ""
echo "For local development:"
echo "  python3 main.py"
echo ""
echo "For AWS deployment:"
echo "  python3 deploy_aws.py"
echo ""
echo "📱 The web interface will open automatically in your browser!"
echo "🌐 For AWS: Configure security group to allow port 7860"