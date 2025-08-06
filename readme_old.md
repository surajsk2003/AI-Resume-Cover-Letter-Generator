# > AI Resume & Cover Letter Generator

**Generate professional resumes and cover letters using AI - No training required!**

Perfect for Mac users with 8GB RAM. Uses pre-trained BART and GPT-2 models for intelligent content generation.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
![Platform](https://img.shields.io/badge/platform-macOS-lightgrey.svg)

## <¯ Features

- ( **Smart Resume Summarization** - BART AI extracts key points
- =Ý **Custom Cover Letters** - GPT-2 generates personalized content
- =Ä **Multiple Formats** - PDF, DOCX, TXT support
- < **Web Interface** - Beautiful drag-and-drop UI
- =» **CLI Mode** - Command line alternative
- = **Privacy First** - Everything runs locally
- <N **Mac Optimized** - Memory efficient for 8GB RAM

## =€ Quick Start

### 1. Setup Environment

```bash
# Clone or download the project
cd "AI Resume & Cover Letter Generator"

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python main.py
```

Choose your interface:
- **Option 1**: Web Interface (Recommended) - Opens browser automatically
- **Option 2**: Command Line Interface - Terminal-based interaction

### 3. Use the Tool

#### Web Interface:
1. =Î Upload your resume (PDF/DOCX/TXT) or paste text
2. =¼ Add job description
3. <â Enter company name (optional)
4. =€ Click "Generate AI Resume & Cover Letter"
5.  Edit results and copy for use

#### CLI Interface:
1. Choose file upload or text input
2. Paste job description
3. Add optional company details
4. Get instant AI-generated results

## =Ë What You Get

### Resume Summary
- Key achievements extracted
- Skills highlighted
- Experience condensed intelligently
- ATS-friendly format

### Cover Letter
- Job-specific customization
- Professional tone
- Company-tailored content
- Ready-to-send format

## =' Technical Details

### AI Models Used:
- **BART-large-CNN**: Resume summarization
- **GPT-2**: Cover letter generation

### System Requirements:
- Python 3.8+
- 8GB RAM (minimum)
- macOS (optimized) / Windows / Linux
- Internet connection (first run only - for model downloads)

### Performance:
- Resume processing: 30-60 seconds
- Cover letter generation: 45-90 seconds
- Memory usage: 4-6GB
- Models download: ~2GB (one-time)

## =¡ Usage Tips

### For Best Results:
1. **Resume Quality**: Include clear sections for experience, skills, education
2. **Job Description**: Paste complete posting for better matching
3. **Company Info**: Add company name for personalized cover letters
4. **Review & Edit**: Generated content is editable - customize as needed

### Supported File Types:
- **PDF**: Most resume formats
- **DOCX**: Microsoft Word documents  
- **TXT**: Plain text files

## =à Advanced Usage

### Memory Management
The app automatically:
- Monitors RAM usage
- Chunks long documents
- Cleans memory between operations
- Uses CPU-only processing (Mac optimized)

### Customization Options
- Adjust generation parameters in `main.py`
- Modify prompt templates for different styles
- Add new document processors for other formats

## =€ Deployment to AWS (Free Tier)

### For AWS Deployment:

1. **Create EC2 Instance** (t2.micro - Free Tier)
2. **Install dependencies**:
   ```bash
   sudo yum update
   python3 -m pip install -r requirements.txt
   ```
3. **Configure security group** - Allow port 7860
4. **Run with public access**:
   ```python
   app.launch(server_name="0.0.0.0", share=True)
   ```

### Environment Variables for Production:
```bash
export GRADIO_SERVER_PORT=7860
export GRADIO_SERVER_NAME="0.0.0.0"
```

## =Ê Sample Output

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

## = Troubleshooting

### Common Issues:

**Memory Error:**
- Close other applications
- Process shorter text chunks
- Restart the application

**PDF Won't Read:**
- Try converting to TXT first
- Check if PDF is text-based (not image)
- Use a different PDF reader to save as new file

**Slow Performance:**
- First run downloads models (~2GB)
- Subsequent runs are much faster
- Consider using CLI for batch processing

**Import Errors:**
```bash
pip install --upgrade -r requirements.txt
```

## <¨ Interface Screenshots

### Web Interface:
- Clean, modern design
- Drag-and-drop file upload
- Real-time processing status
- Side-by-side input/output
- Tabbed interface for different input methods

### CLI Interface:
- Step-by-step prompts
- Progress indicators
- Memory usage monitoring
- Clean text output

## = Privacy & Security

- **Local Processing**: No data sent to external APIs
- **No Storage**: Files are not saved permanently
- **Memory Cleanup**: Automatic memory management
- **Open Source**: Full transparency of code

## =È Roadmap

### Phase 1 (Complete):
-  Core AI functionality
-  Web & CLI interfaces
-  Mac optimization

### Phase 2 (Future):
- = Batch processing
- = Multiple templates
- = Skills matching
- = Version history

### Phase 3 (Advanced):
- = Custom model training
- = Industry-specific models
- = ATS optimization
- = Success analytics

## > Contributing

1. Fork the repository
2. Create feature branch
3. Make improvements
4. Test thoroughly
5. Submit pull request

## =Ä License

MIT License - Feel free to use for personal and commercial projects.

## <˜ Support

For issues or questions:
1. Check troubleshooting section
2. Review sample inputs/outputs
3. Try different file formats
4. Consider system requirements

## <Æ Success Stories

Perfect for:
- =Ê **Job Seekers**: Customize applications quickly
- <¯ **Career Changers**: Highlight transferable skills  
- =¼ **Professionals**: Maintain updated materials
- <“ **Students**: Create first professional documents

---

**Ready to revolutionize your job applications with AI? Get started now!** =€