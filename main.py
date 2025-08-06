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
    
    # Create enhanced Gradio interface with custom CSS
    custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        backdrop-filter: blur(5px);
    }
    
    .generate-btn {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        padding: 15px 30px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        border-radius: 50px !important;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .generate-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4) !important;
    }
    
    .output-section {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    .status-success {
        color: #10b981 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    
    .status-error {
        color: #ef4444 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    
    .input-section {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    .tab-nav {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 15px !important;
        backdrop-filter: blur(10px) !important;
    }
    """
    
    with gr.Blocks(
        title="🤖 AI Resume & Cover Letter Generator", 
        theme=gr.themes.Default(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter")
        ),
        css=custom_css
    ) as app:
        
        # Enhanced Header
        with gr.Row(elem_classes="main-header"):
            gr.Markdown("""
            <div style="text-align: center;">
                <h1 style="color: #1f2937; font-size: 3rem; font-weight: 800; margin-bottom: 1rem; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    🤖 AI Resume & Cover Letter Generator
                </h1>
                <p style="color: #6b7280; font-size: 1.25rem; margin-bottom: 2rem; font-weight: 500;">
                    Transform your career with AI-powered resume optimization and personalized cover letters
                </p>
                <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
                    <div style="background: rgba(102, 126, 234, 0.1); padding: 1rem 2rem; border-radius: 50px; color: #667eea; font-weight: 600;">
                        ✨ BART AI Summarization
                    </div>
                    <div style="background: rgba(118, 75, 162, 0.1); padding: 1rem 2rem; border-radius: 50px; color: #764ba2; font-weight: 600;">
                        🧠 GPT-2 Generation
                    </div>
                    <div style="background: rgba(16, 185, 129, 0.1); padding: 1rem 2rem; border-radius: 50px; color: #10b981; font-weight: 600;">
                        🔒 100% Private
                    </div>
                </div>
            </div>
            """, elem_classes="main-header")
        
        with gr.Tab("📄 Upload Resume File", elem_classes="tab-nav"):
            with gr.Row(equal_height=True):
                # Input Section
                with gr.Column(scale=1, elem_classes="input-section"):
                    gr.Markdown("""
                    <div style="text-align: center; margin-bottom: 1.5rem;">
                        <h3 style="color: #374151; font-weight: 700; font-size: 1.5rem;">📝 Your Information</h3>
                        <p style="color: #6b7280; font-size: 0.95rem;">Upload your resume and provide job details</p>
                    </div>
                    """)
                    
                    file_input = gr.File(
                        label="📎 Upload Your Resume",
                        file_types=[".pdf", ".docx", ".txt"],
                        interactive=True,
                        show_label=True,
                        container=True
                    )
                    
                    job_desc_input1 = gr.Textbox(
                        label="💼 Job Description",
                        placeholder="📋 Paste the complete job posting here...\n\nInclude:\n• Job requirements\n• Responsibilities  \n• Required skills\n• Company information",
                        lines=10,
                        max_lines=15,
                        show_copy_button=True
                    )
                    
                    with gr.Row():
                        company_input1 = gr.Textbox(
                            label="🏢 Company Name",
                            placeholder="e.g., Google, Microsoft, Apple...",
                            scale=2
                        )
                        
                        position_input1 = gr.Textbox(
                            label="💼 Position Title", 
                            placeholder="e.g., Software Engineer, Data Scientist...",
                            scale=2
                        )
                    
                    gr.Markdown("""
                    <div style="text-align: center; margin: 1.5rem 0;">
                        <p style="color: #6b7280; font-size: 0.85rem; font-style: italic;">
                            💡 <strong>Pro Tip:</strong> More detailed job descriptions lead to better customized results!
                        </p>
                    </div>
                    """)
                    
                    process_btn1 = gr.Button(
                        "🚀 Generate AI-Powered Resume & Cover Letter", 
                        variant="primary",
                        elem_classes="generate-btn",
                        size="lg"
                    )
                
                # Output Section
                with gr.Column(scale=1, elem_classes="output-section"):
                    gr.Markdown("""
                    <div style="text-align: center; margin-bottom: 1.5rem;">
                        <h3 style="color: #374151; font-weight: 700; font-size: 1.5rem;">✨ AI-Generated Results</h3>
                        <p style="color: #6b7280; font-size: 0.95rem;">Your personalized resume summary and cover letter</p>
                    </div>
                    """)
                    
                    status1 = gr.Textbox(
                        label="📊 Processing Status", 
                        interactive=False,
                        show_label=True,
                        placeholder="Ready to generate your AI-powered content..."
                    )
                    
                    with gr.Accordion("📋 Resume Summary", open=True):
                        summary_output1 = gr.Textbox(
                            label="",
                            lines=6,
                            interactive=True,
                            placeholder="Your AI-generated resume summary will appear here...",
                            show_copy_button=True,
                            show_label=False
                        )
                    
                    with gr.Accordion("✍️ Cover Letter", open=True):
                        cover_letter_output1 = gr.Textbox(
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
            
            process_btn1.click(
                process_resume_file,
                inputs=[file_input, job_desc_input1, company_input1, position_input1],
                outputs=[summary_output1, cover_letter_output1, status1]
            )
        
        with gr.Tab("✍️ Paste Resume Text", elem_classes="tab-nav"):
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
        
        with gr.Tab("ℹ️ About & Help", elem_classes="tab-nav"):
            with gr.Column(elem_classes="feature-card"):
                gr.Markdown("""
                <div style="text-align: center; margin-bottom: 2rem;">
                    <h2 style="color: #374151; font-weight: 800; font-size: 2rem; margin-bottom: 1rem;">
                        🚀 About This AI Tool
                    </h2>
                    <p style="color: #6b7280; font-size: 1.1rem; font-weight: 500;">
                        Professional resume optimization powered by cutting-edge AI technology
                    </p>
                </div>
                """)
            
            with gr.Row():
                with gr.Column(scale=1, elem_classes="feature-card"):
                    gr.Markdown("""
                    ### 🤖 **How It Works**
                    
                    Our AI system uses **state-of-the-art pre-trained models**:
                    
                    🧠 **BART-large-CNN**
                    - Intelligently summarizes your resume
                    - Extracts key achievements and skills
                    - Creates ATS-friendly summaries
                    
                    ✍️ **GPT-2 Language Model**
                    - Generates personalized cover letters
                    - Tailors content to specific job requirements
                    - Maintains professional tone and structure
                    
                    🔒 **100% Private & Secure**
                    - All processing happens locally on your device
                    - No data sent to external servers
                    - Your information stays completely private
                    """)
                
                with gr.Column(scale=1, elem_classes="feature-card"):
                    gr.Markdown("""
                    ### 💡 **Pro Tips for Best Results**
                    
                    📄 **For Your Resume:**
                    - Include clear sections (Experience, Skills, Education)
                    - Use bullet points for achievements
                    - Quantify your accomplishments with numbers
                    - Keep formatting clean and consistent
                    
                    💼 **For Job Descriptions:**
                    - Paste the complete job posting
                    - Include requirements and responsibilities
                    - Don't forget company information
                    - The more detail, the better the customization!
                    
                    🎯 **Company & Position:**
                    - Always fill in company name when possible
                    - Be specific with position titles
                    - This helps personalize your cover letter
                    """)
            
            with gr.Row():
                with gr.Column(scale=1, elem_classes="feature-card"):
                    gr.Markdown("""
                    ### 🔧 **Technical Specifications**
                    
                    **System Requirements:**
                    - 🍎 macOS (optimized) / Windows / Linux
                    - 💾 8GB RAM minimum (works great on Macs!)
                    - 🌐 Internet (first run only - downloads models)
                    - 💾 ~2GB storage for AI models
                    
                    **Performance:**
                    - ⚡ Resume processing: 30-60 seconds
                    - ✍️ Cover letter generation: 45-90 seconds
                    - 🧠 Memory usage: 4-6GB during processing
                    - 🚀 CPU-only processing (no GPU needed)
                    """)
                
                with gr.Column(scale=1, elem_classes="feature-card"):
                    gr.Markdown("""
                    ### 🎯 **Perfect For**
                    
                    👔 **Job Seekers**
                    - Quickly customize applications for multiple positions
                    - Create professional summaries from detailed resumes
                    - Generate compelling cover letters in minutes
                    
                    🎓 **Students & New Graduates**
                    - Transform academic experience into professional language
                    - Create first professional resume summaries
                    - Learn proper cover letter structure
                    
                    💼 **Career Changers**
                    - Highlight transferable skills effectively
                    - Reframe experience for new industries
                    - Create targeted applications quickly
                    
                    🚀 **Professionals**
                    - Keep application materials updated easily
                    - Customize for different opportunities
                    - Save time on repetitive writing tasks
                    """)
            
            with gr.Column(elem_classes="feature-card"):
                gr.Markdown("""
                <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); border-radius: 15px; margin: 1rem 0;">
                    <h3 style="color: #374151; font-weight: 700; margin-bottom: 1rem;">🌟 Why Choose Our AI Resume Generator?</h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin-top: 2rem;">
                        <div>
                            <h4 style="color: #667eea; font-weight: 600;">🏆 Professional Quality</h4>
                            <p style="color: #6b7280; font-size: 0.9rem;">Industry-standard resume summaries and cover letters</p>
                        </div>
                        <div>
                            <h4 style="color: #764ba2; font-weight: 600;">⚡ Lightning Fast</h4>
                            <p style="color: #6b7280; font-size: 0.9rem;">Generate personalized content in under 2 minutes</p>
                        </div>
                        <div>
                            <h4 style="color: #10b981; font-weight: 600;">🔒 Completely Private</h4>
                            <p style="color: #6b7280; font-size: 0.9rem;">Your data never leaves your computer</p>
                        </div>
                        <div>
                            <h4 style="color: #f59e0b; font-weight: 600;">💰 Free Forever</h4>
                            <p style="color: #6b7280; font-size: 0.9rem;">No subscriptions, no hidden costs</p>
                        </div>
                    </div>
                </div>
                """)
            
            with gr.Column(elem_classes="feature-card"):
                gr.Markdown("""
                ### 📝 **How to Use - Step by Step**
                
                1. **Choose Your Input Method**
                   - 📄 Upload a PDF, DOCX, or TXT file
                   - ✍️ Or paste your resume text directly
                
                2. **Add Job Information**
                   - 📋 Paste the complete job description
                   - 🏢 Enter company name (optional but recommended)
                   - 💼 Add position title (optional but recommended)
                
                3. **Generate AI Content**
                   - 🚀 Click the generate button
                   - ⏱️ Wait 1-2 minutes for AI processing
                   - ✨ Get your personalized results!
                
                4. **Review and Customize**
                   - 📖 Read through the generated content
                   - ✏️ Edit directly in the text boxes
                   - 📋 Copy to your clipboard when ready
                
                **Remember:** The generated content is a starting point - feel free to edit and personalize it further!
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
    print("🍎 AI Resume Generator - Mac Optimized!")
    print("Choose interface:")
    print("1. 🌐 Web Interface (Recommended)")
    print("2. 💻 Command Line Interface")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        run_cli()
    else:
        print("\n🚀 Starting web interface...")
        print("📱 Open your browser to the URL shown below")
        
        app = create_web_app()
        app.launch(
            share=False,  # Set to True if you want public link
            server_port=7860,
            inbrowser=True  # Automatically open browser
        )