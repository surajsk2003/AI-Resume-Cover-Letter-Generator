# 🤖 AI Resume & Cover Letter Generator

**Generate professional resumes and cover letters using AI - No training required!**

Perfect for Mac users with 8GB RAM. Uses pre-trained BART and GPT-2 models for intelligent content generation.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
![Platform](https://img.shields.io/badge/platform-macOS-lightgrey.svg)

## 🎯 Features

- ✨ **Smart Resume Summarization** - BART AI extracts key points
- 📝 **Custom Cover Letters** - GPT-2 generates personalized content
- 📄 **Multiple Formats** - PDF, DOCX, TXT support
- 🌐 **Web Interface** - Beautiful drag-and-drop UI
- 💻 **CLI Mode** - Command line alternative
- 🔒 **Privacy First** - Everything runs locally
- 🍎 **Mac Optimized** - Memory efficient for 8GB RAM

## 🚀 Quick Start

### Option A: Virtual Environment Setup (Recommended)

```bash
# Navigate to project directory
cd "AI Resume & Cover Letter Generator"

# Run the automated setup script
./setup_venv.sh

# For future sessions, activate the environment:
source ai_resume_env/bin/activate

# Run the application
python main.py

# When done, deactivate:
deactivate
```

### Option B: Global Installation (Simple)

```bash
# Navigate to project directory
cd "AI Resume & Cover Letter Generator"

# Install dependencies globally
pip install -r requirements.txt

# Run the application
python main.py
```

### Option C: Manual Virtual Environment

```bash
# Create virtual environment
python3 -m venv ai_resume_env

# Activate environment
source ai_resume_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Run application
python main.py
```

## 🖥️ Choose Your Interface

**Option 1: Web Interface (Recommended)**
- Opens browser automatically at `http://localhost:7860`
- Beautiful drag-and-drop file upload
- Real-time processing status
- Side-by-side input/output

**Option 2: Command Line Interface**  
- Terminal-based interaction
- Step-by-step prompts
- Perfect for automation

## 🎯 How to Use

### Web Interface:
1. 📎 Upload your resume (PDF/DOCX/TXT) or paste text
2. 💼 Add job description
3. 🏢 Enter company name (optional)
4. 🚀 Click "Generate AI Resume & Cover Letter"
5. ✏️ Edit results and copy for use

### CLI Interface:
1. Choose file upload or text input
2. Paste job description
3. Add optional company details
4. Get instant AI-generated results

## 📋 What You Get

### Resume Summary
- Key achievements extracted by BART AI
- Skills highlighted intelligently
- Experience condensed professionally
- ATS-friendly format

### Cover Letter
- Job-specific customization using GPT-2
- Professional tone and structure
- Company-tailored content
- Ready-to-send format

## 🔧 Technical Details

### AI Models Used:
- **facebook/bart-large-cnn**: Resume summarization
- **gpt2**: Cover letter generation

### System Requirements:
- Python 3.8+ (tested with 3.10)
- 8GB RAM (minimum for Mac)
- macOS (optimized) / Windows / Linux
- Internet connection (first run only - for model downloads)

### Performance:
- **Resume processing**: 30-60 seconds
- **Cover letter generation**: 45-90 seconds  
- **Memory usage**: 4-6GB during processing
- **Models download**: ~2GB (one-time)

## 💡 Usage Tips

### For Best Results:
1. **Resume Quality**: Include clear sections for experience, skills, education
2. **Job Description**: Paste complete posting for better matching
3. **Company Info**: Adding company name personalizes cover letters
4. **Review & Edit**: Generated content is editable - customize as needed

### Supported File Types:
- **PDF**: Most resume formats
- **DOCX**: Microsoft Word documents  
- **TXT**: Plain text files

## 🛠️ Project Structure

```
AI Resume & Cover Letter Generator/
├── main.py              # Main application (local development)
├── deploy_aws.py        # AWS-optimized version
├── requirements.txt     # Python dependencies
├── setup_venv.sh        # Virtual environment setup
├── setup.sh            # Global setup script
├── sample_resume.txt    # Test resume file
├── ai_resume_env/       # Virtual environment (created by setup)
└── README.md           # This file
```

## 🚀 AWS Deployment (Free Tier)

For deploying to AWS EC2 free tier:

### 1. Launch EC2 Instance
- **Instance Type**: t2.micro (free tier)
- **OS**: Amazon Linux 2 or Ubuntu
- **Security Group**: Allow HTTP (80) and Custom (7860)

### 2. Setup on Server
```bash
# Update system
sudo yum update -y  # Amazon Linux
# OR
sudo apt update && sudo apt upgrade -y  # Ubuntu

# Install Python and git
sudo yum install python3 python3-pip git -y  # Amazon Linux
# OR  
sudo apt install python3 python3-pip python3-venv git -y  # Ubuntu

# Clone your project
git clone <your-repo-url>
cd "AI Resume & Cover Letter Generator"

# Setup virtual environment
python3 -m venv ai_resume_env
source ai_resume_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Run AWS-optimized version
python3 deploy_aws.py
```

### 3. Access Application
- Visit: `http://your-ec2-public-ip:7860`
- The AWS version includes memory optimizations for free tier

## 📊 Sample Outputs

### Resume Summary Example:
```
Experienced software engineer with 5+ years in full-stack development. 
Proficient in Python, JavaScript, and cloud technologies. Led cross-functional 
teams and delivered scalable applications serving 100K+ users.
```

### Cover Letter Sample:
```
Dear Hiring Manager,

I am writing to express my strong interest in the Software Engineer position 
at Google. With my background in full-stack development and experience leading 
technical teams, I am excited about the opportunity to contribute to your 
innovative projects...
```

## 🔍 Troubleshooting

### Common Issues:

**Virtual Environment Issues:**
```bash
# If pyenv shell doesn't work:
python3 -m venv ai_resume_env
source ai_resume_env/bin/activate

# Check Python version in venv:
python --version
```

**Memory Errors:**
- Close other applications
- Process shorter text chunks  
- Restart the application
- Try the CLI interface (uses less memory)

**PDF Won't Read:**
- Try converting to TXT first
- Check if PDF is text-based (not image)
- Use sample_resume.txt for testing

**Slow Performance:**
- First run downloads models (~2GB)
- Subsequent runs are much faster
- Consider using CLI for batch processing

**Import Errors:**
```bash
# In virtual environment:
pip install --upgrade -r requirements.txt

# If torch issues on Mac:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 🧪 Testing the Setup

Use the included sample file:
```bash
# Activate environment
source ai_resume_env/bin/activate

# Run application  
python main.py

# In web interface:
# 1. Upload sample_resume.txt
# 2. Paste any job description
# 3. Click generate
```

## 🔒 Privacy & Security

- **Local Processing**: No data sent to external APIs
- **No Permanent Storage**: Files processed in memory only
- **Memory Cleanup**: Automatic cleanup after processing
- **Open Source**: Full transparency of code
- **Virtual Environment**: Isolated from system Python

## 📈 Future Enhancements

### Planned Features:
- 🔄 Batch processing multiple resumes
- 🎨 Multiple professional templates  
- 🎯 Industry-specific optimizations
- 📊 ATS score analysis
- 💾 Local save/load functionality

## 🆘 Support

For issues:
1. Check troubleshooting section above
2. Try the sample resume file first
3. Ensure virtual environment is activated
4. Check system resources (memory/disk)

## 🏆 Perfect For

- 📊 **Job Seekers**: Customize applications quickly
- 🎯 **Career Changers**: Highlight transferable skills  
- 💼 **Professionals**: Maintain updated materials
- 🎓 **Students**: Create first professional documents
- 🚀 **Recruiters**: Help candidates improve applications

---

**Ready to revolutionize your job applications with AI?** 

Run `./setup_venv.sh` and get started! 🚀