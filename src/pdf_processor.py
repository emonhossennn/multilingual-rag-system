"""
PDF Processing Module for Bengali Text Extraction

This module handles PDF text extraction with special focus on Bengali Unicode support.
Uses PyMuPDF (fitz) for robust text extraction and handles common formatting issues.
"""

import fitz  # PyMuPDF
import re
import logging
from typing import List, Dict, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    PDF processor optimized for Bengali text extraction.
    
    Handles:
    - Unicode Bengali text extraction
    - Text cleaning and normalization
    - Metadata extraction
    - Error handling for corrupted PDFs
    """
    
    def __init__(self):
        self.supported_formats = ['.pdf']
        
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, any]:
        """
        Extract text from PDF with metadata.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Dict containing extracted text, metadata, and page information
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
                
            if pdf_path.suffix.lower() not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {pdf_path.suffix}")
            
            doc = fitz.open(pdf_path)
            
            extracted_data = {
                'text': '',
                'pages': [],
                'metadata': self._extract_metadata(doc),
                'total_pages': len(doc),
                'file_path': str(pdf_path)
            }
            
            logger.info(f"Processing PDF: {pdf_path.name} ({len(doc)} pages)")
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                # Clean and normalize the text
                cleaned_text = self._clean_text(page_text)
                
                if cleaned_text.strip():  # Only add non-empty pages
                    page_info = {
                        'page_number': page_num + 1,
                        'text': cleaned_text,
                        'char_count': len(cleaned_text),
                        'word_count': len(cleaned_text.split())
                    }
                    
                    extracted_data['pages'].append(page_info)
                    extracted_data['text'] += cleaned_text + '\n\n'
            
            doc.close()
            
            logger.info(f"Successfully extracted text from {len(extracted_data['pages'])} pages")
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise
    
    def _extract_metadata(self, doc) -> Dict[str, str]:
        """Extract metadata from PDF document."""
        metadata = doc.metadata
        return {
            'title': metadata.get('title', ''),
            'author': metadata.get('author', ''),
            'subject': metadata.get('subject', ''),
            'creator': metadata.get('creator', ''),
            'producer': metadata.get('producer', ''),
            'creation_date': metadata.get('creationDate', ''),
            'modification_date': metadata.get('modDate', '')
        }
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Handles:
        - Unicode normalization for Bengali text
        - Removing extra whitespaces
        - Fixing common OCR errors
        - Preserving paragraph structure
        """
        if not text:
            return ""
        
        # Normalize Unicode (important for Bengali text)
        import unicodedata
        text = unicodedata.normalize('NFC', text)
        
        # Remove excessive whitespace while preserving paragraph breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple line breaks to double
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r' +\n', '\n', text)  # Remove trailing spaces before newlines
        text = re.sub(r'\n +', '\n', text)  # Remove leading spaces after newlines
        
        # Fix common OCR issues for Bengali text
        text = self._fix_bengali_ocr_errors(text)
        
        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r'\n\d+\n', '\n', text)  # Standalone page numbers
        text = re.sub(r'^Page \d+.*?\n', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def _fix_bengali_ocr_errors(self, text: str) -> str:
        """
        Fix common OCR errors in Bengali text.
        
        This method can be expanded based on specific OCR error patterns
        found in the HSC Bangla documents.
        """
        # Common Bengali OCR error corrections
        corrections = {
            # Add specific corrections based on your PDF content
            'ও': 'ও',  # Normalize similar looking characters
            'া': 'া',  # Normalize vowel marks
        }
        
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
        
        return text
    
    def validate_extraction(self, extracted_data: Dict) -> bool:
        """
        Validate the quality of text extraction.
        
        Returns True if extraction seems successful, False otherwise.
        """
        if not extracted_data['text'].strip():
            logger.warning("No text extracted from PDF")
            return False
        
        # Check for reasonable Bengali content
        bengali_chars = sum(1 for char in extracted_data['text'] 
                          if '\u0980' <= char <= '\u09FF')  # Bengali Unicode range
        total_chars = len(extracted_data['text'])
        
        if total_chars > 0:
            bengali_ratio = bengali_chars / total_chars
            if bengali_ratio < 0.1:  # Less than 10% Bengali characters
                logger.warning(f"Low Bengali content ratio: {bengali_ratio:.2%}")
        
        logger.info(f"Extraction validation: {bengali_chars} Bengali chars out of {total_chars} total")
        return True


def main():
    """Test the PDF processor with a sample file."""
    processor = PDFProcessor()
    
    # Test with sample PDF (you'll need to provide the actual path)
    pdf_path = "data/hsc_bangla_1st_paper.pdf"
    
    try:
        result = processor.extract_text_from_pdf(pdf_path)
        print(f"Successfully processed: {result['metadata']['title']}")
        print(f"Total pages: {result['total_pages']}")
        print(f"Total text length: {len(result['text'])} characters")
        print(f"First 200 characters: {result['text'][:200]}...")
        
    except FileNotFoundError:
        print(f"Please place your HSC Bangla PDF at: {pdf_path}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()