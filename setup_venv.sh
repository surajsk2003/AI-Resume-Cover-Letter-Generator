#!/bin/bash

# AI Resume & Cover Letter Generator - Virtual Environment Setup
echo "🤖 AI Resume & Cover Letter Generator - Virtual Environment Setup"
echo "================================================================="

# Check if we're already in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Please run this script from the project directory"
    exit 1
fi

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv ai_resume_env

if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

echo "✅ Virtual environment created!"

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source ai_resume_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully!"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source ai_resume_env/bin/activate"
echo ""
echo "Then you can run the application:"
echo "  python main.py"
echo ""
echo "To deactivate the environment:"
echo "  deactivate"
echo ""
echo "🚀 Ready to generate AI-powered resumes and cover letters!"