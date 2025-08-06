# AI Resume & Cover Letter Generator
# Hugging Face Spaces Deployment Version

import os
import re
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import PyPDF2
import docx
from typing import Dict, List, Optional
import gradio as gr
import gc
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. CORE AI ENGINE
# ==========================================

class ResumeAIEngine:
    def __init__(self):
        """Initialize with pre-trained models"""
        print("🚀 Loading AI models...")
        
        # Force CPU usage for Hugging Face Spaces
        device = -1  # CPU only
        
        # Load models with memory optimization
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            tokenizer="facebook/bart-large-cnn",
            device=device,
            max_length=150,
            min_length=50,
            do_sample=False
        )
        
        self.generator = pipeline(
            "text-generation",
            model="gpt2",
            tokenizer="gpt2", 
            device=device,
            max_length=400,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=50256
        )
        
        print("✅ Models loaded successfully!")
        self._cleanup_memory()
    
    def _cleanup_memory(self):
        """Clean up memory after model loading"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    
    def extract_text_from_docx(self, docx_file) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(docx_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            return f"Error reading DOCX: {str(e)}"
    
    def process_resume_file(self, file) -> str:
        """Process uploaded resume file"""
        if file is None:
            return "No file uploaded."
        
        file_extension = file.name.lower().split('.')[-1]
        
        try:
            if file_extension == 'pdf':
                return self.extract_text_from_pdf(file)
            elif file_extension in ['docx', 'doc']:
                return self.extract_text_from_docx(file)
            elif file_extension == 'txt':
                return file.read().decode('utf-8')
            else:
                return "Unsupported file format. Please use PDF, DOCX, or TXT files."
        except Exception as e:
            return f"Error processing file: {str(e)}"
    
    def clean_text(self, text: str) -> str:
        """Clean and prepare text for AI processing"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()
    
    def generate_resume_summary(self, resume_text: str) -> str:
        """Generate AI summary of resume using BART"""
        try:
            # Clean and prepare text
            clean_resume = self.clean_text(resume_text)
            
            # Limit input length for BART (max 1024 tokens)
            if len(clean_resume) > 3000:  # Roughly 1000 tokens
                clean_resume = clean_resume[:3000]
            
            # Generate summary
            summary = self.summarizer(
                clean_resume,
                max_length=130,
                min_length=50,
                do_sample=False
            )
            
            result = summary[0]['summary_text']
            self._cleanup_memory()
            return result
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def generate_cover_letter(self, resume_summary: str, job_description: str, 
                            company_name: str = "", position_title: str = "") -> str:
        """Generate personalized cover letter using GPT-2"""
        try:
            # Create prompt for cover letter
            company_part = f" at {company_name}" if company_name else ""
            position_part = f" for the {position_title} position" if position_title else ""
            
            prompt = f"""Dear Hiring Manager,

I am writing to express my interest{position_part}{company_part}. Based on my background: {resume_summary[:200]}

The job requirements include: {job_description[:300]}

My relevant experience includes"""
            
            # Generate cover letter
            response = self.generator(
                prompt,
                max_length=len(prompt.split()) + 150,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256
            )
            
            # Extract and clean the generated text
            generated_text = response[0]['generated_text']
            cover_letter = generated_text[len(prompt):].strip()
            
            # Add professional closing if not present
            if not any(closing in cover_letter.lower() for closing in ['sincerely', 'regards', 'thank you']):
                cover_letter += "\n\nThank you for considering my application. I look forward to discussing how my experience can contribute to your team.\n\nBest regards,\n[Your Name]"
            
            self._cleanup_memory()
            return cover_letter
            
        except Exception as e:
            return f"Error generating cover letter: {str(e)}"

# ==========================================
# 2. WEB INTERFACE (HUGGING FACE SPACES)
# ==========================================

def create_web_app():
    """Create the Gradio web interface"""
    
    # Initialize AI engine
    ai_engine = ResumeAIEngine()
    
    # Custom CSS for terminal/coding theme
    custom_css = """
    .gradio-container {
        background: #0d1117 !important;
        color: #e6edf3 !important;
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    }
    .main-header {
        background: linear-gradient(135deg, #161b22 0%, #21262d 100%) !important;
        border: 1px solid #30363d !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        margin-bottom: 2rem !important;
    }
    .input-section {
        background: #161b22 !important;
        border: 1px solid #21262d !important;
        border-radius: 8px !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
    }
    .output-section {
        background: #0d1117 !important;
        border: 2px solid #238636 !important;
        border-radius: 8px !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
    }
    """
    
    # Create Gradio interface with dark theme
    with gr.Blocks(
        title="AI Resume Generator - Terminal Edition",
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="purple", 
            neutral_hue="slate",
            font=gr.themes.GoogleFont("JetBrains Mono")
        ).set(
            body_background_fill="#0d1117",
            body_text_color="#e6edf3",
            background_fill_primary="#161b22",
            background_fill_secondary="#21262d",
            border_color_primary="#30363d",
            color_accent="#58a6ff"
        ),
        css=custom_css
    ) as app:
        
        # Header
        with gr.Row(elem_classes="main-header"):
            gr.Markdown("""
            <div style="margin-top: 1.5rem;">
                <div style="font-family: 'JetBrains Mono', monospace; margin-bottom: 2rem;">
                    <div style="color: #7d8590; font-size: 14px; margin-bottom: 0.5rem;">
                        <span style="color: #39d353;">●</span> System Status: Online | 
                        <span style="color: #58a6ff;">Models:</span> BART + GPT-2 | 
                        <span style="color: #bc8cff;">Mode:</span> Local Processing
                    </div>
                    <h1 style="color: #e6edf3; font-size: 2.5rem; font-weight: 700; margin: 1rem 0; font-family: 'JetBrains Mono', monospace;">
                        <span style="color: #ff7b72;">class</span> 
                        <span style="color: #58a6ff;">AIResumeGenerator</span><span style="color: #e6edf3;">:</span>
                    </h1>
                    <div style="color: #7d8590; font-size: 16px; margin-bottom: 2rem; font-family: 'JetBrains Mono', monospace;">
                        <span style="color: #7d8590;">&quot;&quot;&quot;</span>
                        <div style="margin-left: 1rem;">Transform your career with AI-powered resume optimization</div>
                        <div style="margin-left: 1rem;">and personalized cover letter generation.</div>
                        <span style="color: #7d8590;">&quot;&quot;&quot;</span>
                    </div>
                </div>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 2rem;">
                    <div style="background: #21262d; border: 1px solid #30363d; border-left: 3px solid #39d353; padding: 1rem; border-radius: 6px;">
                        <div style="color: #39d353; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 12px;">
                            BART_SUMMARIZER
                        </div>
                        <div style="color: #7d8590; font-size: 11px; margin-top: 4px;">
                            facebook/bart-large-cnn
                        </div>
                    </div>
                    <div style="background: #21262d; border: 1px solid #30363d; border-left: 3px solid #58a6ff; padding: 1rem; border-radius: 6px;">
                        <div style="color: #58a6ff; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 12px;">
                            GPT2_GENERATOR
                        </div>
                        <div style="color: #7d8590; font-size: 11px; margin-top: 4px;">
                            transformers.gpt2
                        </div>
                    </div>
                    <div style="background: #21262d; border: 1px solid #30363d; border-left: 3px solid #bc8cff; padding: 1rem; border-radius: 6px;">
                        <div style="color: #bc8cff; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 12px;">
                            PRIVACY_MODE
                        </div>
                        <div style="color: #7d8590; font-size: 11px; margin-top: 4px;">
                            local_processing=True
                        </div>
                    </div>
                </div>
                """)
        
        # Main Interface Tabs
        with gr.Tabs():
            # Tab 1: File Upload
            with gr.Tab("🗂️ Upload Resume"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        <div style="background: #161b22; padding: 1rem; border-radius: 6px; border-left: 3px solid #39d353;">
                            <span style="color: #ff7b72;">resume_file</span> = <span style="color: #39d353;">input</span>(<span style="color: #a5d6ff;">"Upload resume file: "</span>)
                        </div>
                        """)
                        
                        file_input = gr.File(
                            label="📄 Resume File (PDF, DOCX, TXT)",
                            file_types=[".pdf", ".docx", ".txt"]
                        )
                        
                        gr.Markdown("""
                        <div style="background: #161b22; padding: 1rem; border-radius: 6px; border-left: 3px solid #58a6ff;">
                            <span style="color: #ff7b72;">job_description</span> = <span style="color: #39d353;">input</span>(<span style="color: #a5d6ff;">"Paste job posting: "</span>)
                        </div>
                        """)
                        
                        job_desc_input1 = gr.Textbox(
                            label="📋 Job Description",
                            placeholder="# Paste the complete job posting here\n# Include: requirements, responsibilities, skills, company info",
                            lines=8
                        )
                        
                        gr.Markdown("""
                        <div style="background: #161b22; padding: 1rem; border-radius: 6px; border-left: 3px solid #bc8cff;">
                            <span style="color: #39d353;"># Optional parameters for personalization</span>
                        </div>
                        """)
                        
                        with gr.Row():
                            company_input1 = gr.Textbox(
                                label="🏢 Company Name",
                                placeholder="# e.g., 'Google', 'Microsoft', 'Apple'",
                                lines=1
                            )
                            position_input1 = gr.Textbox(
                                label="💼 Position Title", 
                                placeholder="# e.g., 'Software Engineer', 'Data Scientist'",
                                lines=1
                            )
                        
                        gr.Markdown("""
                        <div style="background: #161b22; padding: 1rem; border-radius: 6px; border-left: 3px solid #f85149;">
                            <span style="color: #39d353;"># Pro tip:</span> More detailed input = better AI results
                        </div>
                        """)
                        
                        process_btn1 = gr.Button(
                            "🚀 Execute AI Generation",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        <div style="background: #0d1117; padding: 1rem; border-radius: 6px; border-left: 3px solid #f85149;">
                            <span style="color: #39d353;"># AI Generation Results</span>
                        </div>
                        """)
                        
                        gr.Markdown("""
                        <div style="background: #161b22; padding: 1rem; border-radius: 6px; border-left: 3px solid #ffa657;">
                            <span style="color: #ff7b72;">execution_status</span> = <span style="color: #39d353;">monitor_process</span>()
                        </div>
                        """)
                        
                        status1 = gr.Textbox(
                            label="⚡ Processing Status",
                            value="# Waiting for execution command...",
                            interactive=False,
                            lines=2
                        )
                        
                        gr.Markdown("""
                        <div style="background: #161b22; padding: 1rem; border-radius: 6px; border-left: 3px solid #39d353;">
                            <span style="color: #ff7b72;">bart_summary</span> = <span style="color: #39d353;">summarize_resume</span>()
                            <div style="color: #7d8590; font-size: 12px; margin-top: 4px;">
                                # AI-extracted key achievements and skills
                            </div>
                        </div>
                        """)
                        
                        summary_output1 = gr.Textbox(
                            label="📝 Resume Summary",
                            placeholder="# BART AI summary will be generated here...\n# Key achievements, skills, and experience extracted from resume",
                            lines=6,
                            interactive=True
                        )
                        
                        gr.Markdown("""
                        <div style="background: #161b22; padding: 1rem; border-radius: 6px; border-left: 3px solid #58a6ff;">
                            <span style="color: #ff7b72;">cover_letter</span> = <span style="color: #39d353;">generate_personalized_content</span>()
                            <div style="color: #7d8590; font-size: 12px; margin-top: 4px;">
                                # GPT-2 generated cover letter tailored to job posting
                            </div>
                        </div>
                        """)
                        
                        cover_letter_output1 = gr.Textbox(
                            label="💌 Cover Letter",
                            placeholder="# Personalized cover letter will be generated here...\n# Tailored to specific job requirements and company\n\n# Dear Hiring Manager,\n# [AI-generated content based on your resume and job posting]",
                            lines=12,
                            interactive=True
                        )
                        
                        gr.Markdown("""
                        <div style="background: #0d1117; padding: 1rem; border-radius: 6px; border-left: 3px solid #39d353;">
                            <span style="color: #39d353;"># Output ready for editing and copying</span>
                            <span style="color: #bc8cff;">def</span> <span style="color: #58a6ff;">edit_and_export</span>(content): <span style="color: #39d353;">return</span> modified_content
                        </div>
                        """)
            
            # Tab 2: Text Input
            with gr.Tab("📝 Paste Resume"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        <div style="background: #161b22; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #58a6ff; margin-bottom: 1.5rem;">
                            <h3 style="color: #58a6ff; margin: 0; font-family: 'JetBrains Mono', monospace;">
                                📝 Direct Text Input
                            </h3>
                            <p style="color: #6b7280; font-size: 0.95rem;">Copy and paste your resume content directly</p>
                        </div>
                        """)
                        
                        text_input = gr.Textbox(
                            label="📄 Resume Text",
                            placeholder="Paste your complete resume text here...\n\nInclude:\n- Contact information\n- Work experience\n- Education\n- Skills\n- Achievements",
                            lines=12
                        )
                        
                        job_desc_input2 = gr.Textbox(
                            label="📋 Job Description",
                            placeholder="Paste the complete job posting here...",
                            lines=6
                        )
                        
                        with gr.Row():
                            company_input2 = gr.Textbox(label="🏢 Company Name", placeholder="Optional")
                            position_input2 = gr.Textbox(label="💼 Position Title", placeholder="Optional")
                        
                        gr.Markdown("""
                        <div style="background: #1a1f2e; padding: 1rem; border-radius: 6px; border-left: 3px solid #ffa500;">
                            <h4 style="color: #ffa500; margin: 0 0 0.5rem 0;">💡 Pro Tips:</h4>
                            <ul style="color: #a0a9c0; font-size: 0.9rem; margin: 0; padding-left: 1.2rem;">
                                <li>Include quantifiable achievements (numbers, percentages)</li>
                                <li>Paste the complete job description for better matching</li>
                                <li>Add company name for personalized cover letters</li>
                            </ul>
                        </p>
                        </div>
                        """)
                        
                        process_btn2 = gr.Button(
                            "🚀 Generate AI Content", 
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        <div style="background: #161b22; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #39d353; margin-bottom: 1.5rem;">
                            <h3 style="color: #39d353; margin: 0; font-family: 'JetBrains Mono', monospace;">
                                🎯 Generated Results
                            </h3>
                            <p style="color: #6b7280; font-size: 0.95rem;">Your personalized resume summary and cover letter</p>
                        </div>
                        """)
                        
                        status2 = gr.Textbox(
                            label="⚡ Status",
                            value="Ready for processing...",
                            interactive=False,
                            lines=2
                        )
                        
                        summary_output2 = gr.Textbox(
                            label="📝 Resume Summary",
                            placeholder="AI-generated resume summary will appear here...",
                            lines=6,
                            interactive=True
                        )
                        
                        cover_letter_output2 = gr.Textbox(
                            label="💌 Cover Letter", 
                            placeholder="Personalized cover letter will appear here...",
                            lines=12,
                            interactive=True
                        )
                        
                        gr.Markdown("""
                        <div style="background: #1a1f2e; padding: 1rem; border-radius: 6px; border-left: 3px solid #39d353;">
                            <h4 style="color: #39d353; margin: 0 0 0.5rem 0;">✏️ Edit & Export</h4>
                            <p style="color: #a0a9c0; font-size: 0.9rem; margin: 0;">
                                Generated content is fully editable. Make any adjustments needed before copying.
                            </p>
                        </div>
                        """)
        
        # Processing functions
        def process_file_upload(file, job_desc, company, position):
            try:
                # Update status
                yield "🔄 Processing resume file...", "", ""
                
                # Process file
                resume_text = ai_engine.process_resume_file(file)
                if "Error" in resume_text:
                    yield f"❌ {resume_text}", "", ""
                    return
                
                yield "🔄 Generating AI summary...", "", ""
                
                # Generate summary
                summary = ai_engine.generate_resume_summary(resume_text)
                
                yield "🔄 Creating personalized cover letter...", summary, ""
                
                # Generate cover letter
                cover_letter = ai_engine.generate_cover_letter(summary, job_desc, company, position)
                
                yield "✅ Generation complete! Edit content above as needed.", summary, cover_letter
                
            except Exception as e:
                yield f"❌ Error: {str(e)}", "", ""
        
        def process_text_input(resume_text, job_desc, company, position):
            try:
                yield "🔄 Processing resume text...", "", ""
                
                if not resume_text or not resume_text.strip():
                    yield "❌ Please paste your resume text first.", "", ""
                    return
                
                yield "🔄 Generating AI summary...", "", ""
                
                # Generate summary
                summary = ai_engine.generate_resume_summary(resume_text)
                
                yield "🔄 Creating personalized cover letter...", summary, ""
                
                # Generate cover letter  
                cover_letter = ai_engine.generate_cover_letter(summary, job_desc, company, position)
                
                yield "✅ Generation complete! Edit content above as needed.", summary, cover_letter
                
            except Exception as e:
                yield f"❌ Error: {str(e)}", "", ""
        
        # Connect event handlers
        process_btn1.click(
            process_file_upload,
            inputs=[file_input, job_desc_input1, company_input1, position_input1],
            outputs=[status1, summary_output1, cover_letter_output1]
        )
        
        process_btn2.click(
            process_text_input,
            inputs=[text_input, job_desc_input2, company_input2, position_input2], 
            outputs=[status2, summary_output2, cover_letter_output2]
        )
    
    return app

# ==========================================
# 3. LAUNCH APPLICATION
# ==========================================

if __name__ == "__main__":
    print("💻 AI Resume Generator - Hugging Face Spaces Edition!")
    print("🚀 Starting web interface...")
    
    app = create_web_app()
    app.launch()