# AWS Deployment Script
# Optimized version for AWS EC2 Free Tier deployment

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

# AWS-specific optimizations
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
os.environ['HF_HOME'] = '/tmp/hf_home'

class ResumeAIEngine:
    def __init__(self):
        """Initialize with pre-trained models - AWS optimized"""
        print("🚀 Loading AI models for AWS deployment...")
        
        # Force CPU usage to save memory
        device = -1  # CPU only
        
        # Load models with aggressive memory optimization for AWS
        try:
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=device,
                torch_dtype=torch.float32,
                model_kwargs={"cache_dir": "/tmp/transformers_cache"}
            )
            
            self.generator = pipeline(
                "text-generation", 
                model="gpt2",
                device=device,
                torch_dtype=torch.float32,
                model_kwargs={"cache_dir": "/tmp/transformers_cache"}
            )
            
            # Load tokenizer for custom generation
            self.gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir="/tmp/transformers_cache")
            self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
            
            print("✅ Models loaded successfully!")
            self._print_memory_usage()
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("💡 Make sure you have enough disk space and memory")
            raise
    
    def _print_memory_usage(self):
        """Monitor memory usage"""
        memory = psutil.virtual_memory()
        print(f"💾 Memory usage: {memory.percent:.1f}% ({memory.used/1024/1024/1024:.1f}GB/{memory.total/1024/1024/1024:.1f}GB)")
    
    def _clean_memory(self):
        """Aggressive memory cleanup for AWS"""
        gc.collect()
        # Clear any cached data
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def summarize_resume(self, resume_text: str) -> str:
        """AWS-optimized resume summarization"""
        try:
            # More aggressive text cleaning for AWS
            resume_text = self._clean_text(resume_text)
            
            # Smaller chunks for AWS memory constraints
            if len(resume_text) > 800:
                chunks = self._chunk_text(resume_text, max_length=600)
                summaries = []
                
                for i, chunk in enumerate(chunks):
                    print(f"📝 Processing chunk {i+1}/{len(chunks)}")
                    
                    summary = self.summarizer(
                        chunk,
                        max_length=80,  # Reduced for AWS
                        min_length=20,
                        do_sample=False,
                        truncation=True
                    )
                    summaries.append(summary[0]['summary_text'])
                    
                    # More frequent memory cleaning
                    self._clean_memory()
                
                combined_summary = " ".join(summaries)
                
                if len(combined_summary) > 250:
                    final_summary = self.summarizer(
                        combined_summary,
                        max_length=120,
                        min_length=40,
                        do_sample=False
                    )
                    return final_summary[0]['summary_text']
                
                return combined_summary
            else:
                summary = self.summarizer(
                    resume_text,
                    max_length=120,
                    min_length=40,
                    do_sample=False
                )
                return summary[0]['summary_text']
                
        except Exception as e:
            print(f"❌ Error in summarization: {e}")
            return self._fallback_summarize(resume_text)
    
    def generate_cover_letter(self, resume_summary: str, job_description: str, 
                            company_name: str = "", position: str = "") -> str:
        """AWS-optimized cover letter generation"""
        try:
            prompt = self._create_cover_letter_prompt(
                resume_summary, job_description, company_name, position
            )
            
            # Reduced parameters for AWS
            result = self.generator(
                prompt,
                max_length=len(prompt.split()) + 100,  # Smaller output
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.gpt2_tokenizer.eos_token_id,
                repetition_penalty=1.1,
                num_return_sequences=1
            )
            
            generated_text = result[0]['generated_text']
            cover_letter = self._extract_cover_letter(generated_text, prompt)
            
            return self._format_cover_letter(cover_letter, company_name, position)
            
        except Exception as e:
            print(f"❌ Error in cover letter generation: {e}")
            return self._fallback_cover_letter(resume_summary, job_description, company_name)
    
    # Include all helper methods from main.py here (abbreviated for brevity)
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s\-\.\,\;\:\!\?]', '', text)
        return text
    
    def _chunk_text(self, text: str, max_length: int = 600) -> List[str]:
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
        company_part = f" at {company_name}" if company_name else ""
        position_part = f"for the {position} position" if position else "for this position"
        
        prompt = f"""Resume Summary: {resume_summary[:150]}

Job Requirements: {job_description[:250]}

Professional Cover Letter {position_part}{company_part}:

Dear Hiring Manager,

I am writing to express my strong interest"""
        
        return prompt
    
    def _extract_cover_letter(self, generated_text: str, original_prompt: str) -> str:
        start_markers = ["Dear Hiring Manager", "I am writing to express"]
        
        for marker in start_markers:
            if marker in generated_text:
                cover_letter = generated_text.split(marker, 1)[-1]
                return marker + cover_letter
        
        if len(generated_text) > len(original_prompt):
            return generated_text[len(original_prompt):].strip()
        
        return generated_text
    
    def _format_cover_letter(self, cover_letter: str, company_name: str, position: str) -> str:
        if not cover_letter.startswith("Dear"):
            cover_letter = f"Dear Hiring Manager,\n\n{cover_letter}"
        
        if not any(closing in cover_letter.lower() for closing in ["sincerely", "best regards", "thank you"]):
            cover_letter += f"\n\nThank you for considering my application. I look forward to discussing how my experience can contribute to {company_name if company_name else 'your team'}.\n\nBest regards,\n[Your Name]"
        
        cover_letter = re.sub(r'\n\s*\n\s*\n', '\n\n', cover_letter)
        return cover_letter.strip()
    
    def _fallback_summarize(self, text: str) -> str:
        sentences = text.split('.')[:3]  # Even fewer sentences for AWS
        return '. '.join(sentences) + '.'
    
    def _fallback_cover_letter(self, resume_summary: str, job_description: str, company_name: str) -> str:
        company_part = f" at {company_name}" if company_name else ""
        
        return f"""Dear Hiring Manager,

I am excited to apply for this position{company_part}. Based on my background in {resume_summary[:80]}, I believe I would be a strong fit for your team.

The job requirements align well with my experience, particularly in the areas mentioned in your posting. I am eager to contribute my skills and learn from your team.

Thank you for considering my application. I look forward to discussing this opportunity further.

Best regards,
[Your Name]"""

# Document processor (same as main.py but with error handling for AWS)
class DocumentProcessor:
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
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
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            print(f"❌ Error reading TXT: {e}")
            return ""

def create_aws_app():
    """Create web app optimized for AWS deployment"""
    
    print("🌐 Initializing AWS-optimized AI Resume Generator...")
    
    # Initialize with error handling
    try:
        ai_engine = ResumeAIEngine()
        doc_processor = DocumentProcessor()
    except Exception as e:
        print(f"❌ Failed to initialize AI engine: {e}")
        return None
    
    def process_resume_file(file, job_description, company_name, position_title):
        if file is None:
            return "❌ Please upload a resume file", "", ""
        
        try:
            file_path = file.name
            if file_path.lower().endswith('.pdf'):
                resume_text = doc_processor.extract_text_from_pdf(file_path)
            elif file_path.lower().endswith('.docx'):
                resume_text = doc_processor.extract_text_from_docx(file_path)
            else:
                resume_text = doc_processor.extract_text_from_txt(file_path)
            
            if not resume_text.strip():
                return "❌ Could not extract text from file", "", ""
            
            print("🔍 Generating resume summary...")
            resume_summary = ai_engine.summarize_resume(resume_text)
            
            cover_letter = ""
            if job_description.strip():
                print("✍️ Generating cover letter...")
                cover_letter = ai_engine.generate_cover_letter(
                    resume_summary, job_description, company_name, position_title
                )
            
            return resume_summary, cover_letter, "✅ Processing complete!"
            
        except Exception as e:
            return f"❌ Error processing file: {str(e)}", "", ""
    
    def process_manual_resume(resume_text, job_description, company_name, position_title):
        if not resume_text.strip():
            return "❌ Please enter resume text", "", ""
        
        try:
            print("🔍 Generating resume summary...")
            resume_summary = ai_engine.summarize_resume(resume_text)
            
            cover_letter = ""
            if job_description.strip():
                print("✍️ Generating cover letter...")
                cover_letter = ai_engine.generate_cover_letter(
                    resume_summary, job_description, company_name, position_title
                )
            
            return resume_summary, cover_letter, "✅ Processing complete!"
            
        except Exception as e:
            return f"❌ Error processing resume: {str(e)}", "", ""
    
    # Create Gradio interface with AWS-friendly settings
    with gr.Blocks(
        title="🤖 AI Resume & Cover Letter Generator - AWS", 
        theme=gr.themes.Soft()
    ) as app:
        
        gr.Markdown("""
        # 🤖 AI Resume & Cover Letter Generator
        ### Running on AWS Free Tier
        
        **Upload your resume** and **paste a job description** to generate:
        - ✨ **Smart resume summary** (using BART AI)
        - 📝 **Customized cover letter** (using GPT-2 AI)
        
        *Optimized for AWS deployment - please be patient during processing!*
        """)
        
        with gr.Tab("📄 Upload Resume File"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(
                        label="📎 Upload Resume (PDF, DOCX, TXT)",
                        file_types=[".pdf", ".docx", ".txt"]
                    )
                    
                    job_desc_input1 = gr.Textbox(
                        label="💼 Job Description",
                        placeholder="Paste the job posting here...",
                        lines=6
                    )
                    
                    company_input1 = gr.Textbox(
                        label="🏢 Company Name (Optional)",
                        placeholder="e.g., Amazon, Google, Microsoft"
                    )
                    
                    position_input1 = gr.Textbox(
                        label="💼 Position Title (Optional)",
                        placeholder="e.g., Software Engineer, Data Scientist"
                    )
                    
                    process_btn1 = gr.Button("🚀 Generate AI Resume & Cover Letter", variant="primary")
                
                with gr.Column(scale=1):
                    status1 = gr.Textbox(label="📊 Status", interactive=False)
                    
                    summary_output1 = gr.Textbox(
                        label="📋 AI-Generated Resume Summary",
                        lines=6,
                        interactive=True
                    )
                    
                    cover_letter_output1 = gr.Textbox(
                        label="✍️ AI-Generated Cover Letter",
                        lines=10,
                        interactive=True
                    )
            
            process_btn1.click(
                process_resume_file,
                inputs=[file_input, job_desc_input1, company_input1, position_input1],
                outputs=[summary_output1, cover_letter_output1, status1]
            )
        
        with gr.Tab("✍️ Paste Resume Text"):
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="📝 Resume Text",
                        placeholder="Paste your full resume text here...",
                        lines=10
                    )
                    
                    job_desc_input2 = gr.Textbox(
                        label="💼 Job Description",
                        placeholder="Paste the job posting here...",
                        lines=6
                    )
                    
                    company_input2 = gr.Textbox(
                        label="🏢 Company Name (Optional)",
                        placeholder="e.g., Amazon, Google, Microsoft"
                    )
                    
                    position_input2 = gr.Textbox(
                        label="💼 Position Title (Optional)",
                        placeholder="e.g., Software Engineer, Data Scientist"
                    )
                    
                    process_btn2 = gr.Button("🚀 Generate AI Resume & Cover Letter", variant="primary")
                
                with gr.Column(scale=1):
                    status2 = gr.Textbox(label="📊 Status", interactive=False)
                    
                    summary_output2 = gr.Textbox(
                        label="📋 AI-Generated Resume Summary",
                        lines=6,
                        interactive=True
                    )
                    
                    cover_letter_output2 = gr.Textbox(
                        label="✍️ AI-Generated Cover Letter",
                        lines=10,
                        interactive=True
                    )
            
            process_btn2.click(
                process_manual_resume,
                inputs=[text_input, job_desc_input2, company_input2, position_input2],
                outputs=[summary_output2, cover_letter_output2, status2]
            )
        
        with gr.Tab("☁️ AWS Info"):
            gr.Markdown("""
            ## 🚀 AWS Deployment Information
            
            This version is optimized for **AWS EC2 Free Tier** deployment:
            
            ### 🔧 Optimizations Made:
            - **Reduced memory usage**: Smaller model parameters
            - **Aggressive caching**: Models cached in `/tmp/`
            - **Frequent cleanup**: Memory management between operations
            - **Smaller outputs**: Reduced generation length
            
            ### ⚡ Performance Notes:
            - **First run**: ~3-5 minutes (model download)
            - **Subsequent runs**: 1-3 minutes per operation
            - **Memory usage**: ~2-4GB during processing
            
            ### 🛠️ AWS Setup Commands:
            ```bash
            # On your EC2 instance:
            sudo yum update -y
            sudo yum install python3 python3-pip -y
            pip3 install -r requirements.txt
            
            # Run the app:
            python3 deploy_aws.py
            ```
            
            ### 🌐 Public Access:
            The app will be available at: `http://your-ec2-ip:7860`
            
            Make sure to configure your **Security Group** to allow inbound traffic on port 7860!
            """)
    
    return app

if __name__ == "__main__":
    print("☁️ AI Resume Generator - AWS Deployment Version!")
    
    app = create_aws_app()
    
    if app is None:
        print("❌ Failed to create app. Check your system resources.")
        exit(1)
    
    print("\n🚀 Starting AWS-optimized web interface...")
    print("🌍 Will be available on all network interfaces")
    
    # AWS-friendly launch configuration
    app.launch(
        server_name="0.0.0.0",  # Listen on all interfaces
        server_port=7860,       # Standard port
        share=False,            # Don't use gradio sharing for AWS
        inbrowser=False,        # Don't try to open browser on server
        show_error=True,        # Show errors for debugging
        quiet=False             # Show startup logs
    )