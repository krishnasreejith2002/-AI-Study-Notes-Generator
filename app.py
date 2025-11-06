# ==========================================================
# 📘 AI Study Notes Generator - Streamlit App
# Author: Krishna S
# GenAI Capstone Project 2025
# ==========================================================

import streamlit as st
from transformers import pipeline
import PyPDF2
import docx
from datetime import datetime
import io
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Study Notes Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .note-section {
        background-color: #f0f7ff;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1E88E5;
    }
    .flashcard {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 2px solid #ff9800;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .stProgress .st-bo {
        background-color: #1E88E5;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================================
# 🔧 CACHE MODEL LOADING
# ==========================================================

@st.cache_resource(show_spinner=False)
def load_summarizer():
    """Load the summarization model (cached)"""
    return pipeline("summarization", model="facebook/bart-large-cnn")

# ==========================================================
# 📄 TEXT EXTRACTION FUNCTIONS
# ==========================================================

def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_docx(uploaded_file):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(uploaded_file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return ""

def extract_text_from_txt(uploaded_file):
    """Extract text from TXT file"""
    try:
        text = uploaded_file.read().decode('utf-8')
        return text.strip()
    except Exception as e:
        st.error(f"Error reading TXT: {str(e)}")
        return ""

# ==========================================================
# ✂️ TEXT PROCESSING
# ==========================================================

def split_text_into_chunks(text, max_tokens=400):
    """Split text into manageable chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunk = ' '.join(words[i:i + max_tokens])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
    return chunks

# ==========================================================
# 🧠 NOTE GENERATION FUNCTIONS
# ==========================================================

def generate_comprehensive_summary(text, summarizer, detail_level=3, progress_bar=None):
    """Generate comprehensive summary"""
    max_length = 100 + (detail_level * 30)
    min_length = 30 + (detail_level * 10)
    
    chunks = split_text_into_chunks(text, max_tokens=400)
    summaries = []
    
    for i, chunk in enumerate(chunks):
        try:
            if progress_bar:
                progress_bar.progress((i + 1) / len(chunks))
            
            summary = summarizer(
                chunk[:3000],
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )[0]['summary_text']
            summaries.append(summary)
        except:
            continue
    
    if not summaries:
        return "Could not generate summary."
    
    return "\n\n".join([f"📌 Section {i+1}:\n{s}" for i, s in enumerate(summaries)])

def generate_bullet_points(text, summarizer, detail_level=3, progress_bar=None):
    """Generate bullet points"""
    chunks = split_text_into_chunks(text, max_tokens=400)
    all_points = []
    
    for i, chunk in enumerate(chunks):
        try:
            if progress_bar:
                progress_bar.progress((i + 1) / len(chunks))
            
            summary = summarizer(
                chunk[:3000],
                max_length=120,
                min_length=30,
                do_sample=False
            )[0]['summary_text']
            
            sentences = summary.split('. ')
            points = [f"• {s.strip()}{'.' if not s.endswith('.') else ''}" 
                     for s in sentences if len(s.strip()) > 10]
            all_points.extend(points)
        except:
            continue
    
    return "\n".join(all_points) if all_points else "Could not generate bullet points."

def generate_flashcards(text, summarizer, detail_level=3, progress_bar=None):
    """Generate flashcards"""
    num_cards = detail_level * 5
    chunks = split_text_into_chunks(text, max_tokens=500)
    flashcards = []
    
    for i, chunk in enumerate(chunks[:num_cards]):
        try:
            if progress_bar:
                progress_bar.progress((i + 1) / min(len(chunks), num_cards))
            
            summary = summarizer(
                chunk[:3000],
                max_length=80,
                min_length=20,
                do_sample=False
            )[0]['summary_text']
            
            sentences = summary.split('. ')
            if len(sentences) > 0:
                question = f"What is important about: {sentences[0][:50]}...?"
                flashcard = f"Q: {question}\nA: {summary}"
                flashcards.append(flashcard)
        except:
            continue
    
    return "\n\n".join([f"--- Card {i+1} ---\n{card}" for i, card in enumerate(flashcards)]) if flashcards else "Could not generate flashcards."

def generate_qa_format(text, summarizer, detail_level=3, progress_bar=None):
    """Generate Q&A format"""
    chunks = split_text_into_chunks(text, max_tokens=400)
    qa_pairs = []
    num_questions = detail_level * 3
    
    for i, chunk in enumerate(chunks[:num_questions]):
        try:
            if progress_bar:
                progress_bar.progress((i + 1) / min(len(chunks), num_questions))
            
            summary = summarizer(
                chunk[:3000],
                max_length=100,
                min_length=30,
                do_sample=False
            )[0]['summary_text']
            
            words = summary.split()
            if len(words) > 10:
                question = f"Q{i+1}: Explain - {' '.join(words[:8])}...?"
                answer = f"A{i+1}: {summary}"
                qa_pairs.append(f"{question}\n{answer}")
        except:
            continue
    
    return "\n\n".join(qa_pairs) if qa_pairs else "Could not generate Q&A pairs."

def generate_concept_map(text, summarizer, detail_level=3, progress_bar=None):
    """Generate concept map"""
    chunks = split_text_into_chunks(text, max_tokens=500)
    concepts = []
    
    for i, chunk in enumerate(chunks[:detail_level * 2]):
        try:
            if progress_bar:
                progress_bar.progress((i + 1) / min(len(chunks), detail_level * 2))
            
            summary = summarizer(
                chunk[:3000],
                max_length=60,
                min_length=20,
                do_sample=False
            )[0]['summary_text']
            concepts.append(summary)
        except:
            continue
    
    if not concepts:
        return "Could not generate concept map."
    
    result = "📊 CONCEPT MAP\n" + "="*50 + "\n\nMain Topic\n|\n"
    for i, concept in enumerate(concepts, 1):
        result += f"├─ Concept {i}\n│  └─ {concept}\n│\n"
    
    return result

# ==========================================================
# 🎨 MAIN APP INTERFACE
# ==========================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">📚 AI Study Notes Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Transform your study materials into structured notes using AI</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'generated_notes' not in st.session_state:
        st.session_state.generated_notes = None
    if 'text_extracted' not in st.session_state:
        st.session_state.text_extracted = ""
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.markdown("### 📋 Note Format")
        note_type = st.selectbox(
            "Choose format:",
            ["Comprehensive Summary", "Bullet Points", "Flashcards", "Q&A Format", "Concept Map", "All Formats"]
        )
        
        st.markdown("### 🎯 Detail Level")
        detail_level = st.slider(
            "Adjust detail level:",
            min_value=1,
            max_value=5,
            value=3,
            help="1 = Brief, 5 = Very Detailed"
        )
        
        st.markdown("---")
        
        st.markdown("### ℹ️ About")
        st.info("""
        **AI Study Notes Generator**
        
        Uses AI to transform your study materials into structured notes.
        
        ✨ Features:
        - Multiple note formats
        - PDF/DOCX/TXT support
        - Adjustable detail levels
        - Instant download
        
        🎓 GenAI Capstone Project 2025
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📝 Generate Notes", "📄 View Notes", "📖 Guide"])
    
    with tab1:
        st.header("Upload Your Study Material")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['txt', 'pdf', 'docx'],
            help="Supported formats: TXT, PDF, DOCX"
        )
        
        # Text area for direct input
        st.markdown("**OR paste text directly:**")
        pasted_text = st.text_area(
            "Paste your content here:",
            height=200,
            placeholder="Paste your study material here..."
        )
        
        # Process input
        input_text = ""
        
        if uploaded_file:
            with st.spinner("Extracting text from file..."):
                if uploaded_file.type == "application/pdf":
                    input_text = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    input_text = extract_text_from_docx(uploaded_file)
                elif uploaded_file.type == "text/plain":
                    input_text = extract_text_from_txt(uploaded_file)
                
                if input_text:
                    st.success(f"✅ Extracted {len(input_text)} characters from {uploaded_file.name}")
        
        if pasted_text:
            input_text = pasted_text
        
        # Display statistics
        if input_text:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Characters", f"{len(input_text):,}")
            with col2:
                st.metric("Words", f"{len(input_text.split()):,}")
            with col3:
                reading_time = len(input_text.split()) // 200
                st.metric("Reading Time", f"~{reading_time} min")
        
        # Generate button
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            generate_btn = st.button(
                "🚀 Generate Study Notes",
                type="primary",
                use_container_width=True
            )
        
        # Generate notes
        if generate_btn:
            if not input_text:
                st.error("⚠️ Please provide some study material!")
            else:
                # Load model
                with st.spinner("Loading AI model..."):
                    summarizer = load_summarizer()
                
                st.success("✅ Model loaded!")
                
                # Generate notes
                results = {}
                
                if note_type == "All Formats":
                    formats = ["Comprehensive Summary", "Bullet Points", "Flashcards", "Q&A Format", "Concept Map"]
                else:
                    formats = [note_type]
                
                for fmt in formats:
                    st.markdown(f"### Generating {fmt}...")
                    progress_bar = st.progress(0)
                    
                    try:
                        if fmt == "Comprehensive Summary":
                            result = generate_comprehensive_summary(input_text, summarizer, detail_level, progress_bar)
                        elif fmt == "Bullet Points":
                            result = generate_bullet_points(input_text, summarizer, detail_level, progress_bar)
                        elif fmt == "Flashcards":
                            result = generate_flashcards(input_text, summarizer, detail_level, progress_bar)
                        elif fmt == "Q&A Format":
                            result = generate_qa_format(input_text, summarizer, detail_level, progress_bar)
                        elif fmt == "Concept Map":
                            result = generate_concept_map(input_text, summarizer, detail_level, progress_bar)
                        
                        results[fmt] = result
                        progress_bar.empty()
                        st.success(f"✅ {fmt} complete!")
                        
                    except Exception as e:
                        st.error(f"Error generating {fmt}: {str(e)}")
                        progress_bar.empty()
                
                # Save to session state
                if results:
                    st.session_state.generated_notes = {
                        'content': results,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'detail_level': detail_level,
                        'original_length': len(input_text)
                    }
                    
                    st.balloons()
                    st.success("🎉 All notes generated successfully! Check the 'View Notes' tab.")
    
    with tab2:
        st.header("Your Generated Notes")
        
        if st.session_state.generated_notes:
            notes = st.session_state.generated_notes
            
            # Display metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Generated", notes['timestamp'].split()[1])
            with col2:
                st.metric("Detail Level", f"{notes['detail_level']}/5")
            with col3:
                st.metric("Formats", len(notes['content']))
            
            st.markdown("---")
            
            # Display each note format
            for fmt, content in notes['content'].items():
                with st.expander(f"📝 {fmt}", expanded=True):
                    st.markdown(f'<div class="note-section">{content.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
            
            # Download options
            st.markdown("---")
            st.subheader("💾 Download Your Notes")
            
            # Prepare download content
            download_content = f"""
{'='*60}
📚 AI STUDY NOTES GENERATOR
Generated: {notes['timestamp']}
Detail Level: {notes['detail_level']}/5
{'='*60}

"""
            for fmt, content in notes['content'].items():
                download_content += f"\n\n{'='*60}\n{fmt.upper()}\n{'='*60}\n\n{content}\n"
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📄 Download as TXT",
                    data=download_content,
                    file_name=f"study_notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col2:
                markdown_content = download_content.replace('=', '#')
                st.download_button(
                    label="📝 Download as Markdown",
                    data=markdown_content,
                    file_name=f"study_notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
        
        else:
            st.info("📭 No notes generated yet. Go to the 'Generate Notes' tab to create your study notes!")
    
    with tab3:
        st.header("📖 User Guide")
        
        st.markdown("""
        ### How to Use
        
        1. **Upload or Paste Content**
           - Upload PDF, DOCX, or TXT files
           - Or paste text directly
        
        2. **Choose Settings**
           - Select note format from sidebar
           - Adjust detail level (1-5)
        
        3. **Generate Notes**
           - Click "Generate Study Notes"
           - Wait for AI to process
        
        4. **Download**
           - View notes in "View Notes" tab
           - Download as TXT or Markdown
        
        ### Note Formats Explained
        
        - **Comprehensive Summary**: Detailed overview with sections
        - **Bullet Points**: Quick-scan structured points
        - **Flashcards**: Q&A pairs for memorization
        - **Q&A Format**: Study questions with answers
        - **Concept Map**: Visual relationship mapping
        - **All Formats**: Generate everything at once
        
        ### Tips for Best Results
        
        ✅ Use clear, well-structured source material  
        ✅ For long documents, expect longer processing time  
        ✅ Higher detail levels = more comprehensive notes  
        ✅ Try different formats for different subjects  
        ✅ Combine AI notes with your own annotations  
        
        ### Technical Details
        
        - **Model**: Facebook BART (CNN fine-tuned)
        - **Processing**: Chunk-based summarization
        - **Privacy**: All processing happens in real-time
        - **No Storage**: Files are not saved on server
        
        ---
        
        **GenAI Capstone Project 2025**  
        Made with ❤️ by Krishna S
        """)

# Run the app
if __name__ == "__main__":
    main()
