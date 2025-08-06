# AI Resume & Cover Letter Generator (No Training Required)
# Perfect for Mac with 8GB RAM - Uses pre-trained models only!

import os
import re
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import PyPDF2
import docx
from typing import Dict, List, Optional
import gradio as gr
import gc
import psutil
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. CORE AI ENGINE (NO TRAINING NEEDED!)
# ==========================================

class ResumeAIEngine:
    def __init__(self):
        """Initialize with pre-trained models"""
        print("🚀 Loading AI models...")
        
        # Force CPU usage to save memory on Mac
        device = -1  # CPU only
        
        # Load models with memory optimization
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=device,
            torch_dtype=torch.float32
        )
        
        self.generator = pipeline(
            "text-generation", 
            model="gpt2",
            device=device,
            torch_dtype=torch.float32
        )
        
        # Load tokenizer for custom generation
        self.gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        
        print("✅ Models loaded successfully!")
        self._print_memory_usage()
    
    def _print_memory_usage(self):
        """Monitor memory usage"""
        memory = psutil.virtual_memory()
        print(f"💾 Memory usage: {memory.percent:.1f}% ({memory.used/1024/1024/1024:.1f}GB/{memory.total/1024/1024/1024:.1f}GB)")
    
    def _clean_memory(self):
        """Clean up memory"""
        gc.collect()
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    def summarize_resume(self, resume_text: str) -> str:
        """
        Intelligently summarize a resume using BART
        """
        try:
            # Clean and prepare text
            resume_text = self._clean_text(resume_text)
            
            # Handle long resumes by chunking
            if len(resume_text) > 1000:
                chunks = self._chunk_text(resume_text, max_length=800)
                summaries = []
                
                for i, chunk in enumerate(chunks):
                    print(f"📝 Processing chunk {i+1}/{len(chunks)}")
                    
                    summary = self.summarizer(
                        chunk,
                        max_length=100,
                        min_length=30,
                        do_sample=False,
                        truncation=True
                    )
                    summaries.append(summary[0]['summary_text'])
                    
                    # Clean memory between chunks
                    if i % 2 == 0:
                        self._clean_memory()
                
                # Combine summaries
                combined_summary = " ".join(summaries)
                
                # Final summarization if too long
                if len(combined_summary) > 300:
                    final_summary = self.summarizer(
                        combined_summary,
                        max_length=150,
                        min_length=50,
                        do_sample=False
                    )
                    return final_summary[0]['summary_text']
                
                return combined_summary
            else:
                # Short resume - direct summarization
                summary = self.summarizer(
                    resume_text,
                    max_length=150,
                    min_length=50,
                    do_sample=False
                )
                return summary[0]['summary_text']
                
        except Exception as e:
            print(f"❌ Error in summarization: {e}")
            return self._fallback_summarize(resume_text)
    
    def generate_cover_letter(self, resume_summary: str, job_description: str, 
                            company_name: str = "", position: str = "") -> str:
        """
        Generate a customized cover letter using GPT-2
        """
        try:
            # Create smart prompt
            prompt = self._create_cover_letter_prompt(
                resume_summary, job_description, company_name, position
            )
            
            # Generate with improved parameters for better quality
            result = self.generator(
                prompt,
                max_length=len(prompt.split()) + 120,  # Shorter for more focused output
                temperature=0.4,  # Lower temperature for more coherent text
                do_sample=True,
                pad_token_id=self.gpt2_tokenizer.eos_token_id,
                repetition_penalty=1.2,  # Higher penalty to reduce repetition
                top_p=0.9,  # Add nucleus sampling for better quality
                num_return_sequences=1
            )
            
            # Extract and clean the generated text
            generated_text = result[0]['generated_text']
            cover_letter = self._extract_cover_letter(generated_text, prompt)
            
            return self._format_cover_letter(cover_letter, company_name, position)
            
        except Exception as e:
            print(f"❌ Error in cover letter generation: {e}")
            return self._fallback_cover_letter(resume_summary, job_description, company_name)
    
    def customize_resume_bullet(self, original_bullet: str, job_description: str) -> str:
        """
        Customize resume bullet points for specific job
        """
        try:
            prompt = f"""
            Original experience: {original_bullet}
            Job requirements: {job_description}
            
            Rewrite this experience to better match the job requirements:
            """
            
            result = self.generator(
                prompt,
                max_length=len(prompt.split()) + 50,
                temperature=0.6,
                do_sample=True,
                pad_token_id=self.gpt2_tokenizer.eos_token_id
            )
            
            generated = result[0]['generated_text']
            # Extract the rewritten part
            rewritten = generated.split("Rewrite this experience to better match the job requirements:")[-1].strip()
            
            return rewritten[:200]  # Limit length
            
        except Exception as e:
            return original_bullet
    
    # ==========================================
    # HELPER METHODS
    # ==========================================
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\-\.\,\;\:\!\?]', '', text)
        return text
    
    def _chunk_text(self, text: str, max_length: int = 800) -> List[str]:
        """Split text into manageable chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(' '.join(current_chunk)) > max_length:
                chunks.append(' '.join(current_chunk[:-1]))
                current_chunk = [word]
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _create_cover_letter_prompt(self, resume_summary: str, job_description: str, 
                                  company_name: str, position: str) -> str:
        """Create an effective prompt for cover letter generation"""
        company_part = f" at {company_name}" if company_name else ""
        position_part = f"for the {position} position" if position else "for this position"
        
        prompt = f"""Write a professional cover letter based on the following:

Candidate Background: {resume_summary[:150]}

Job Description: {job_description[:200]}

Company: {company_name if company_name else 'this company'}
Position: {position if position else 'this role'}

Cover Letter:

Dear Hiring Manager,

I am excited to apply {position_part}{company_part}. My background in data science and machine learning aligns perfectly with your requirements. I have experience with"""
        
        return prompt
    
    def _extract_cover_letter(self, generated_text: str, original_prompt: str) -> str:
        """Extract the cover letter part from generated text"""
        # Find where the actual cover letter starts
        start_markers = ["Dear Hiring Manager", "I am writing to express"]
        
        for marker in start_markers:
            if marker in generated_text:
                cover_letter = generated_text.split(marker, 1)[-1]
                return marker + cover_letter
        
        # Fallback: return everything after the prompt
        if len(generated_text) > len(original_prompt):
            return generated_text[len(original_prompt):].strip()
        
        return generated_text
    
    def _format_cover_letter(self, cover_letter: str, company_name: str, position: str) -> str:
        """Format and improve the cover letter"""
        # Ensure proper structure
        if not cover_letter.startswith("Dear"):
            cover_letter = f"Dear Hiring Manager,\n\n{cover_letter}"
        
        # Add closing if missing
        if not any(closing in cover_letter.lower() for closing in ["sincerely", "best regards", "thank you"]):
            cover_letter += f"\n\nThank you for considering my application. I look forward to discussing how my experience can contribute to {company_name if company_name else 'your team'}.\n\nBest regards,\n[Your Name]"
        
        # Clean up formatting
        cover_letter = re.sub(r'\n\s*\n\s*\n', '\n\n', cover_letter)  # Remove triple line breaks
        
        return cover_letter.strip()
    
    def _fallback_summarize(self, text: str) -> str:
        """Fallback summarization method"""
        sentences = text.split('.')[:5]  # Take first 5 sentences
        return '. '.join(sentences) + '.'
    
    def _fallback_cover_letter(self, resume_summary: str, job_description: str, company_name: str) -> str:
        """Fallback cover letter generation"""
        company_part = f" at {company_name}" if company_name else ""
        
        return f"""Dear Hiring Manager,

I am excited to apply for this position{company_part}. Based on my background in {resume_summary[:100]}, I believe I would be a strong fit for your team.

The job requirements align well with my experience, particularly in the areas mentioned in your posting. I am eager to contribute my skills and learn from your team.

Thank you for considering my application. I look forward to discussing this opportunity further.

Best regards,
[Your Name]"""

# ==========================================
# 2. DOCUMENT PROCESSING
# ==========================================

class DocumentProcessor:
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            print(f"❌ Error reading PDF: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = []
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
            return "\n".join(text).strip()
        except Exception as e:
            print(f"❌ Error reading DOCX: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_txt(file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            print(f"❌ Error reading TXT: {e}")
            return ""

# ==========================================
# 3. WEB INTERFACE WITH GRADIO
# ==========================================

def create_web_app():
    """Create a beautiful web interface"""
    
    # Initialize AI engine
    ai_engine = ResumeAIEngine()
    doc_processor = DocumentProcessor()
    
    def process_resume_file(file, job_description, company_name, position_title):
        """Process uploaded resume file and generate outputs"""
        if file is None:
            return "❌ Please upload a resume file", "", ""
        
        try:
            # Extract text based on file type
            file_path = file.name
            if file_path.lower().endswith('.pdf'):
                resume_text = doc_processor.extract_text_from_pdf(file_path)
            elif file_path.lower().endswith('.docx'):
                resume_text = doc_processor.extract_text_from_docx(file_path)
            else:
                resume_text = doc_processor.extract_text_from_txt(file_path)
            
            if not resume_text.strip():
                return "❌ Could not extract text from file", "", ""
            
            # Generate summary
            print("🔍 Generating resume summary...")
            resume_summary = ai_engine.summarize_resume(resume_text)
            
            # Generate cover letter if job description provided
            cover_letter = ""
            if job_description.strip():
                print("✍️ Generating cover letter...")
                cover_letter = ai_engine.generate_cover_letter(
                    resume_summary, 
                    job_description, 
                    company_name, 
                    position_title
                )
            
            return resume_summary, cover_letter, "✅ Processing complete!"
            
        except Exception as e:
            return f"❌ Error processing file: {str(e)}", "", ""
    
    def process_manual_resume(resume_text, job_description, company_name, position_title):
        """Process manually entered resume text"""
        if not resume_text.strip():
            return "❌ Please enter resume text", "", ""
        
        try:
            # Generate summary
            print("🔍 Generating resume summary...")
            resume_summary = ai_engine.summarize_resume(resume_text)
            
            # Generate cover letter if job description provided
            cover_letter = ""
            if job_description.strip():
                print("✍️ Generating cover letter...")
                cover_letter = ai_engine.generate_cover_letter(
                    resume_summary, 
                    job_description, 
                    company_name, 
                    position_title
                )
            
            return resume_summary, cover_letter, "✅ Processing complete!"
            
        except Exception as e:
            return f"❌ Error processing resume: {str(e)}", "", ""
    
    # Create coding/developer-style UI with terminal aesthetics
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Fira+Code:wght@400;500;600&display=swap');
    
    :root {
        --bg-primary: #0d1117;
        --bg-secondary: #161b22;
        --bg-tertiary: #21262d;
        --border-primary: #30363d;
        --border-accent: #58a6ff;
        --text-primary: #e6edf3;
        --text-secondary: #7d8590;
        --text-accent: #58a6ff;
        --success: #238636;
        --error: #da3633;
        --warning: #d29922;
        --terminal-green: #39d353;
        --terminal-blue: #58a6ff;
        --terminal-purple: #bc8cff;
        --terminal-orange: #ff7b72;
    }
    
    .gradio-container {
        background: var(--bg-primary) !important;
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
        color: var(--text-primary) !important;
        min-height: 100vh !important;
    }
    
    .main-header {
        background: var(--bg-secondary) !important;
        border: 2px solid var(--border-primary) !important;
        border-radius: 8px !important;
        padding: 2rem !important;
        margin: 1rem 0 !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3) !important;
        position: relative !important;
    }
    
    .main-header::before {
        content: "// AI Resume Generator Terminal v2.0" !important;
        position: absolute !important;
        top: 8px !important;
        left: 12px !important;
        font-size: 12px !important;
        color: var(--text-secondary) !important;
        font-weight: 400 !important;
    }
    
    .code-block {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-primary) !important;
        border-left: 4px solid var(--terminal-blue) !important;
        border-radius: 6px !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
        font-family: 'Fira Code', monospace !important;
        position: relative !important;
    }
    
    .code-block::before {
        content: "●●●" !important;
        position: absolute !important;
        top: 8px !important;
        right: 12px !important;
        color: var(--terminal-orange) !important;
        font-size: 12px !important;
    }
    
    .terminal-window {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-primary) !important;
        border-radius: 8px !important;
        padding: 0 !important;
        margin: 1rem 0 !important;
        overflow: hidden !important;
    }
    
    .terminal-header {
        background: var(--bg-tertiary) !important;
        padding: 8px 16px !important;
        border-bottom: 1px solid var(--border-primary) !important;
        font-size: 12px !important;
        color: var(--text-secondary) !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }
    
    .terminal-header::before {
        content: "⬤ ⬤ ⬤" !important;
        color: #ff5f57 #ffbd2e #28ca42 !important;
        margin-right: auto !important;
    }
    
    .terminal-body {
        padding: 1.5rem !important;
        background: var(--bg-secondary) !important;
        min-height: 200px !important;
    }
    
    .generate-btn {
        background: linear-gradient(135deg, var(--terminal-blue), var(--terminal-purple)) !important;
        color: var(--text-primary) !important;
        border: 2px solid var(--border-accent) !important;
        padding: 12px 24px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        font-family: 'JetBrains Mono', monospace !important;
        border-radius: 6px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: none !important;
        letter-spacing: 0.5px !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .generate-btn::before {
        content: "$ " !important;
        color: var(--terminal-green) !important;
        font-weight: bold !important;
    }
    
    .generate-btn:hover {
        background: linear-gradient(135deg, var(--terminal-purple), var(--terminal-blue)) !important;
        border-color: var(--terminal-green) !important;
        box-shadow: 0 0 20px rgba(88, 166, 255, 0.3) !important;
        transform: translateY(-1px) !important;
    }
    
    .input-field {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-primary) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
        font-family: 'JetBrains Mono', monospace !important;
        padding: 12px !important;
        transition: border-color 0.3s ease !important;
    }
    
    .input-field:focus {
        border-color: var(--terminal-blue) !important;
        box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.1) !important;
        outline: none !important;
    }
    
    .output-terminal {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-primary) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        margin: 1rem 0 !important;
    }
    
    .status-success {
        color: var(--terminal-green) !important;
        font-weight: 600 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .status-success::before {
        content: "[SUCCESS] " !important;
        color: var(--terminal-green) !important;
    }
    
    .status-error {
        color: var(--error) !important;
        font-weight: 600 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .status-error::before {
        content: "[ERROR] " !important;
        color: var(--error) !important;
    }
    
    .tab-nav {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-primary) !important;
        border-radius: 8px 8px 0 0 !important;
    }
    
    .tab-nav button {
        background: var(--bg-tertiary) !important;
        color: var(--text-secondary) !important;
        border: none !important;
        padding: 12px 20px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 13px !important;
        border-radius: 6px 6px 0 0 !important;
        transition: all 0.3s ease !important;
        margin: 4px 2px 0 2px !important;
    }
    
    .tab-nav button.selected {
        background: var(--bg-primary) !important;
        color: var(--text-accent) !important;
        border-bottom: 2px solid var(--terminal-blue) !important;
    }
    
    .tab-nav button:hover {
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }
    
    .accordion {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-primary) !important;
        border-radius: 6px !important;
        margin: 8px 0 !important;
        overflow: hidden !important;
    }
    
    .accordion summary {
        background: var(--bg-secondary) !important;
        padding: 12px 16px !important;
        cursor: pointer !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 600 !important;
        color: var(--text-accent) !important;
        border-bottom: 1px solid var(--border-primary) !important;
        transition: background-color 0.3s ease !important;
    }
    
    .accordion summary:hover {
        background: var(--bg-tertiary) !important;
    }
    
    .accordion[open] summary {
        border-bottom: 1px solid var(--border-primary) !important;
    }
    
    .code-comment {
        color: var(--text-secondary) !important;
        font-style: italic !important;
    }
    
    .syntax-highlight .keyword {
        color: var(--terminal-purple) !important;
        font-weight: 600 !important;
    }
    
    .syntax-highlight .string {
        color: var(--terminal-green) !important;
    }
    
    .syntax-highlight .function {
        color: var(--terminal-blue) !important;
    }
    
    .syntax-highlight .comment {
        color: var(--text-secondary) !important;
        font-style: italic !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px !important;
        height: 8px !important;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-primary) !important;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-primary) !important;
        border-radius: 4px !important;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-secondary) !important;
    }
    
    /* Loading animation */
    @keyframes terminal-blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    .terminal-cursor::after {
        content: "▋" !important;
        color: var(--terminal-green) !important;
        animation: terminal-blink 1s infinite !important;
    }
    
    /* Glowing effect for active elements */
    .glow {
        box-shadow: 0 0 10px rgba(88, 166, 255, 0.3) !important;
    }
    
    /* File upload styling */
    .file-upload {
        border: 2px dashed var(--border-primary) !important;
        border-radius: 8px !important;
        padding: 2rem !important;
        text-align: center !important;
        background: var(--bg-tertiary) !important;
        transition: all 0.3s ease !important;
    }
    
    .file-upload:hover {
        border-color: var(--terminal-blue) !important;
        background: var(--bg-secondary) !important;
    }
    """
    
    with gr.Blocks(
        title="AI Resume Generator Terminal", 
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
        
        # Coding-style Header
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
            </div>
            """, elem_classes="main-header")
        
        with gr.Tab("🗂️ upload_resume.py", elem_classes="tab-nav"):
            with gr.Row(equal_height=True):
                # Input Section - Terminal Style
                with gr.Column(scale=1, elem_classes="terminal-window"):
                    gr.Markdown("""
                    <div class="terminal-header">
                        <span style="color: #39d353;">user@ai-resume-gen</span>:<span style="color: #58a6ff;">~/input</span>$ 
                        <span style="color: #e6edf3;">python upload_resume.py</span>
                    </div>
                    <div class="terminal-body">
                        <div style="color: #7d8590; font-size: 13px; margin-bottom: 1rem;">
                            <span style="color: #39d353;"># Initialize input parameters</span>                            <span style="color: #bc8cff;">def</span> <span style="color: #58a6ff;">collect_user_data</span>():
                        </div>
                    </div>
                    """)
                    
                    gr.Markdown("""
                    <div style="color: #7d8590; font-size: 12px; margin: 1rem 0;">
                        <span style="color: #ff7b72;">resume_file</span> = <span style="color: #39d353;">input</span>(<span style="color: #a5d6ff;">"Upload resume file: "</span>)
                    </div>
                    """)
                    
                    file_input = gr.File(
                        label="resume_file = ",
                        file_types=[".pdf", ".docx", ".txt"],
                        interactive=True,
                        show_label=True,
                        elem_classes="input-field file-upload"
                    )
                    
                    gr.Markdown("""
                    <div style="color: #7d8590; font-size: 12px; margin: 1rem 0;">
                        <span style="color: #ff7b72;">job_description</span> = <span style="color: #39d353;">input</span>(<span style="color: #a5d6ff;">"Paste job posting: "</span>)
                    </div>
                    """)
                    
                    job_desc_input1 = gr.Textbox(
                        label="job_description = ",
                        placeholder="# Paste the complete job posting here\n# Include: requirements, responsibilities, skills, company info\n\n",
                        lines=8,
                        max_lines=12,
                        show_copy_button=True,
                        elem_classes="input-field"
                    )
                    
                    gr.Markdown("""
                    <div style="color: #7d8590; font-size: 12px; margin: 1rem 0;">
                        <span style="color: #39d353;"># Optional parameters for personalization</span>
                    </div>
                    """)
                    
                    with gr.Row():
                        company_input1 = gr.Textbox(
                            label="company_name = ",
                            placeholder="# e.g., 'Google', 'Microsoft', 'Apple'",
                            scale=2,
                            elem_classes="input-field"
                        )
                        
                        position_input1 = gr.Textbox(
                            label="position_title = ", 
                            placeholder="# e.g., 'Software Engineer', 'Data Scientist'",
                            scale=2,
                            elem_classes="input-field"
                        )
                    
                    gr.Markdown("""
                    <div style="background: #21262d; border: 1px solid #30363d; border-radius: 6px; padding: 1rem; margin: 1rem 0;">
                        <div style="color: #7d8590; font-size: 11px;">
                            <span style="color: #39d353;"># Pro tip:</span> More detailed input = better AI results                            <span style="color: #39d353;"># Function:</span> process_resume_data(file, job_desc, company, position)
                        </div>
                    </div>
                    """)
                    
                    process_btn1 = gr.Button(
                        "Execute AI Generation", 
                        variant="primary",
                        elem_classes="generate-btn",
                        size="lg"
                    )
                
                # Output Section - Terminal Style
                with gr.Column(scale=1, elem_classes="terminal-window"):
                    gr.Markdown("""
                    <div class="terminal-header">
                        <span style="color: #39d353;">ai-engine@localhost</span>:<span style="color: #58a6ff;">~/output</span>$ 
                        <span style="color: #e6edf3;">python generate_results.py</span>
                    </div>
                    <div class="terminal-body">
                        <div style="color: #7d8590; font-size: 13px; margin-bottom: 1rem;">
                            <span style="color: #39d353;"># AI Generation Results</span>                            <span style="color: #bc8cff;">import</span> <span style="color: #e6edf3;">bart_summarizer, gpt2_generator</span>
                        </div>
                    </div>
                    """)
                    
                    gr.Markdown("""
                    <div style="color: #7d8590; font-size: 12px; margin: 1rem 0;">
                        <span style="color: #ff7b72;">execution_status</span> = <span style="color: #39d353;">monitor_process</span>()
                    </div>
                    """)
                    
                    status1 = gr.Textbox(
                        label="execution_status = ", 
                        interactive=False,
                        show_label=True,
                        placeholder="# Waiting for execution command...",
                        elem_classes="input-field"
                    )
                    
                    gr.Markdown("""
                    <div style="background: #21262d; border: 1px solid #30363d; border-left: 3px solid #39d353; border-radius: 6px; padding: 1rem; margin: 1rem 0;">
                        <div style="color: #39d353; font-size: 12px; font-weight: 600; margin-bottom: 0.5rem;">
                            bart_summary = summarize_resume()
                        </div>
                        <div style="color: #7d8590; font-size: 11px;">
                            # AI-extracted key achievements and skills
                        </div>
                    </div>
                    """)
                    
                    summary_output1 = gr.Textbox(
                        label="bart_summary = ",
                        lines=6,
                        interactive=True,
                        placeholder="# BART AI summary will be generated here...\n# Key achievements, skills, and experience extracted from resume",
                        show_copy_button=True,
                        elem_classes="input-field"
                    )
                    
                    gr.Markdown("""
                    <div style="background: #21262d; border: 1px solid #30363d; border-left: 3px solid #58a6ff; border-radius: 6px; padding: 1rem; margin: 1rem 0;">
                        <div style="color: #58a6ff; font-size: 12px; font-weight: 600; margin-bottom: 0.5rem;">
                            cover_letter = generate_personalized_content()
                        </div>
                        <div style="color: #7d8590; font-size: 11px;">
                            # GPT-2 generated cover letter tailored to job posting
                        </div>
                    </div>
                    """)
                    
                    cover_letter_output1 = gr.Textbox(
                        label="cover_letter = ",
                        lines=12,
                        interactive=True,
                        placeholder="# Personalized cover letter will be generated here...\n# Tailored to specific job requirements and company\n\n# Dear Hiring Manager,\n# [AI-generated content based on your resume and job posting]",
                        show_copy_button=True,
                        elem_classes="input-field"
                    )
                    
                    gr.Markdown("""
                    <div style="background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 1rem; margin: 1rem 0;">
                        <div style="color: #7d8590; font-size: 11px;">
                            <span style="color: #39d353;"># Output ready for editing and copying</span>                            <span style="color: #bc8cff;">def</span> <span style="color: #58a6ff;">edit_and_export</span>(content): <span style="color: #39d353;">return</span> modified_content                            <span style="color: #7d8590;"># Modify the generated content above as needed</span>
                        </div>
                    </div>
                    """)
            
            process_btn1.click(
                process_resume_file,
                inputs=[file_input, job_desc_input1, company_input1, position_input1],
                outputs=[summary_output1, cover_letter_output1, status1]
            )
        
        with gr.Tab("📝 paste_resume.py", elem_classes="tab-nav"):
            with gr.Row(equal_height=True):
                # Input Section
                with gr.Column(scale=1, elem_classes="input-section"):
                    gr.Markdown("""
                    <div style="text-align: center; margin-bottom: 1.5rem;">
                        <h3 style="color: #374151; font-weight: 700; font-size: 1.5rem;">📝 Paste Your Resume</h3>
                        <p style="color: #6b7280; font-size: 0.95rem;">Copy and paste your resume content directly</p>
                    </div>
                    """)
                    
                    text_input = gr.Textbox(
                        label="📄 Resume Content",
                        placeholder="📝 Paste your complete resume text here...\n\nInclude all sections:\n• Contact Information\n• Professional Summary\n• Work Experience\n• Education\n• Skills\n• Projects/Achievements",
                        lines=14,
                        max_lines=20,
                        show_copy_button=True
                    )
                    
                    job_desc_input2 = gr.Textbox(
                        label="💼 Job Description",
                        placeholder="📋 Paste the complete job posting here...\n\nInclude:\n• Job requirements\n• Responsibilities  \n• Required skills\n• Company information",
                        lines=8,
                        max_lines=12,
                        show_copy_button=True
                    )
                    
                    with gr.Row():
                        company_input2 = gr.Textbox(
                            label="🏢 Company Name",
                            placeholder="e.g., Google, Microsoft, Apple...",
                            scale=2
                        )
                        
                        position_input2 = gr.Textbox(
                            label="💼 Position Title",
                            placeholder="e.g., Software Engineer, Data Scientist...",
                            scale=2
                        )
                    
                    gr.Markdown("""
                    <div style="text-align: center; margin: 1.5rem 0;">
                        <p style="color: #6b7280; font-size: 0.85rem; font-style: italic;">
                            📋 <strong>Quick Tip:</strong> Well-formatted resume text produces better AI results!
                        </p>
                    </div>
                    """)
                    
                    process_btn2 = gr.Button(
                        "🚀 Generate AI-Powered Resume & Cover Letter", 
                        variant="primary",
                        elem_classes="generate-btn",
                        size="lg"
                    )
                
                # Output Section (same as first tab)
                with gr.Column(scale=1, elem_classes="output-section"):
                    gr.Markdown("""
                    <div style="text-align: center; margin-bottom: 1.5rem;">
                        <h3 style="color: #374151; font-weight: 700; font-size: 1.5rem;">✨ AI-Generated Results</h3>
                        <p style="color: #6b7280; font-size: 0.95rem;">Your personalized resume summary and cover letter</p>
                    </div>
                    """)
                    
                    status2 = gr.Textbox(
                        label="📊 Processing Status", 
                        interactive=False,
                        show_label=True,
                        placeholder="Ready to generate your AI-powered content..."
                    )
                    
                    with gr.Accordion("📋 Resume Summary", open=True):
                        summary_output2 = gr.Textbox(
                            label="",
                            lines=6,
                            interactive=True,
                            placeholder="Your AI-generated resume summary will appear here...",
                            show_copy_button=True,
                            show_label=False
                        )
                    
                    with gr.Accordion("✍️ Cover Letter", open=True):
                        cover_letter_output2 = gr.Textbox(
                            label="",
                            lines=12,
                            interactive=True,
                            placeholder="Your personalized cover letter will be generated here...",
                            show_copy_button=True,
                            show_label=False
                        )
                    
                    gr.Markdown("""
                    <div style="text-align: center; margin-top: 1rem; padding: 1rem; background: rgba(59, 130, 246, 0.1); border-radius: 10px;">
                        <p style="color: #3b82f6; font-size: 0.9rem; font-weight: 500; margin: 0;">
                            📝 <strong>Edit & Copy:</strong> You can modify the generated content before using it!
                        </p>
                    </div>
                    """)
            
            process_btn2.click(
                process_manual_resume,
                inputs=[text_input, job_desc_input2, company_input2, position_input2],
                outputs=[summary_output2, cover_letter_output2, status2]
            )
        
        with gr.Tab("📖 documentation.md", elem_classes="tab-nav"):
            with gr.Column(elem_classes="code-block"):
                gr.Markdown("""
                <div style="font-family: 'JetBrains Mono', monospace; color: #e6edf3;">
                    <div style="color: #7d8590; font-size: 12px; margin-bottom: 1rem;">
                        <span style="color: #39d353;"># AI Resume Generator Documentation</span>                        <span style="color: #39d353;"># Version:</span> 2.0                        <span style="color: #39d353;"># Models:</span> BART + GPT-2                        <span style="color: #39d353;"># Mode:</span> Local Processing
                    </div>
                    
                    <h3 style="color: #58a6ff; font-size: 1.5rem; margin: 1.5rem 0;">
                        <span style="color: #bc8cff;">class</span> SystemInfo:
                    </h3>
                    
                    <div style="margin-left: 2rem; color: #7d8590; font-size: 14px;">
                        <div style="margin: 0.5rem 0;">
                            <span style="color: #ff7b72;">models</span> = {                            <span style="margin-left: 2rem; color: #39d353;">"summarizer"</span>: <span style="color: #a5d6ff;">"facebook/bart-large-cnn"</span>,                            <span style="margin-left: 2rem; color: #39d353;">"generator"</span>: <span style="color: #a5d6ff;">"gpt2"</span>                            }
                        </div>
                        
                        <div style="margin: 1rem 0;">
                            <span style="color: #ff7b72;">requirements</span> = {                            <span style="margin-left: 2rem; color: #39d353;">"ram"</span>: <span style="color: #a5d6ff;">"8GB minimum"</span>,                            <span style="margin-left: 2rem; color: #39d353;">"cpu_only"</span>: <span style="color: #79c0ff;">True</span>,                            <span style="margin-left: 2rem; color: #39d353;">"privacy"</span>: <span style="color: #a5d6ff;">"100% local processing"</span>                            }
                        </div>
                        
                        <div style="margin: 1rem 0;">
                            <span style="color: #ff7b72;">performance</span> = {                            <span style="margin-left: 2rem; color: #39d353;">"resume_processing"</span>: <span style="color: #a5d6ff;">"30-60 seconds"</span>,                            <span style="margin-left: 2rem; color: #39d353;">"cover_letter_gen"</span>: <span style="color: #a5d6ff;">"45-90 seconds"</span>                            }
                        </div>
                    </div>
                    
                    <h3 style="color: #58a6ff; font-size: 1.5rem; margin: 2rem 0 1rem 0;">
                        <span style="color: #bc8cff;">def</span> <span style="color: #58a6ff;">usage_guide</span>():
                    </h3>
                    
                    <div style="margin-left: 2rem; color: #7d8590; font-size: 14px;">
                        <span style="color: #39d353;"># Step 1: Input your resume</span>                        resume = upload_file() <span style="color: #7d8590;">or</span> paste_text()<br>                        
                        <span style="color: #39d353;"># Step 2: Add job posting</span>                        job_desc = input(<span style="color: #a5d6ff;">"Complete job description"</span>)<br>                        
                        <span style="color: #39d353;"># Step 3: Optional personalization</span>                        company = input(<span style="color: #a5d6ff;">"Company name"</span>)  <span style="color: #7d8590;"># Optional</span>                        position = input(<span style="color: #a5d6ff;">"Position title"</span>)  <span style="color: #7d8590;"># Optional</span><br>                        
                        <span style="color: #39d353;"># Step 4: Execute AI generation</span>                        results = ai_engine.process(resume, job_desc, company, position)<br>                        
                        <span style="color: #39d353;"># Step 5: Edit and export</span>                        <span style="color: #bc8cff;">return</span> edit_and_copy(results)
                    </div>
                    
                    <h3 style="color: #58a6ff; font-size: 1.5rem; margin: 2rem 0 1rem 0;">
                        <span style="color: #bc8cff;">def</span> <span style="color: #58a6ff;">pro_tips</span>():
                    </h3>
                    
                    <div style="margin-left: 2rem; color: #7d8590; font-size: 14px;">
                        tips = [                        <span style="margin-left: 2rem; color: #a5d6ff;">"Include quantifiable achievements in resume"</span>,                        <span style="margin-left: 2rem; color: #a5d6ff;">"Paste complete job posting for best results"</span>,                        <span style="margin-left: 2rem; color: #a5d6ff;">"Add company name for personalized cover letters"</span>,                        <span style="margin-left: 2rem; color: #a5d6ff;">"Review and edit AI-generated content"</span>                        ]                        <span style="color: #bc8cff;">return</span> tips
                    </div>
                    
                    <div style="background: #21262d; border: 1px solid #30363d; padding: 1rem; border-radius: 6px; margin: 2rem 0; color: #7d8590; font-size: 12px;">
                        <span style="color: #39d353;"># System Status: READY</span>                        <span style="color: #39d353;"># Models: LOADED</span>                        <span style="color: #39d353;"># Privacy: LOCAL_PROCESSING_ONLY</span>                        <span style="color: #39d353;"># Cost: FREE_FOREVER</span>
                    </div>
                </div>
                """)
    
    return app

# ==========================================
# 4. COMMAND LINE INTERFACE (ALTERNATIVE)
# ==========================================

def run_cli():
    """Command line interface for the tool"""
    print("🤖 AI Resume & Cover Letter Generator")
    print("=" * 50)
    
    ai_engine = ResumeAIEngine()
    doc_processor = DocumentProcessor()
    
    # Get resume
    print("\n📄 Resume Input:")
    print("1. Upload file (PDF/DOCX/TXT)")
    print("2. Paste text")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    resume_text = ""
    if choice == "1":
        file_path = input("📎 Enter file path: ").strip()
        if file_path.lower().endswith('.pdf'):
            resume_text = doc_processor.extract_text_from_pdf(file_path)
        elif file_path.lower().endswith('.docx'):
            resume_text = doc_processor.extract_text_from_docx(file_path)
        else:
            resume_text = doc_processor.extract_text_from_txt(file_path)
    else:
        print("📝 Paste your resume (press Enter twice to finish):")
        lines = []
        while True:
            line = input()
            if line == "" and len(lines) > 0:
                break
            lines.append(line)
        resume_text = "\n".join(lines)
    
    # Get job description
    print("\n💼 Job Description:")
    print("📝 Paste job description (press Enter twice to finish):")
    job_lines = []
    while True:
        line = input()
        if line == "":
            break
        job_lines.append(line)
    job_description = "\n".join(job_lines)
    
    # Optional details
    company_name = input("\n🏢 Company name (optional): ").strip()
    position = input("💼 Position title (optional): ").strip()
    
    # Process
    print("\n🔄 Processing...")
    
    # Generate summary
    resume_summary = ai_engine.summarize_resume(resume_text)
    print(f"\n📋 Resume Summary:\n{resume_summary}")
    
    # Generate cover letter
    if job_description:
        cover_letter = ai_engine.generate_cover_letter(
            resume_summary, job_description, company_name, position
        )
        print(f"\n✍️ Cover Letter:\n{cover_letter}")
    
    print("\n✅ Complete!")

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("💻 AI Resume Generator - Terminal Edition!")
    print("Choose interface:")
    print("1. 🌐 Web Interface (Recommended)")
    print("2. 💻 Command Line Interface")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        run_cli()
    else:
        print("\n🚀 Starting terminal-style web interface...")
        print("📱 Open your browser to the URL shown below")
        
        app = create_web_app()
        app.launch(
            share=False,  # Set to True if you want public link
            server_port=7860,
            inbrowser=True  # Automatically open browser
        )
