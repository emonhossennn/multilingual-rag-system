"""
Document Chunking Module

Implements semantic chunking strategies optimized for Bengali literature content.
Focuses on preserving context and meaning while maintaining optimal chunk sizes for retrieval.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a document chunk with metadata."""
    text: str
    chunk_id: str
    source_page: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    word_count: Optional[int] = None
    metadata: Optional[Dict] = None


class DocumentChunker:
    """
    Advanced document chunker for Bengali literature.
    
    Implements multiple chunking strategies:
    1. Semantic paragraph-based chunking
    2. Sentence-based chunking with context preservation
    3. Fixed-size chunking with overlap
    """
    
    def __init__(self, 
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 min_chunk_size: int = 100):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target size for chunks (in characters)
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum size for a valid chunk
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
    def chunk_document(self, 
                      text: str, 
                      strategy: str = "semantic_paragraph",
                      metadata: Optional[Dict] = None) -> List[Chunk]:
        """
        Chunk document using specified strategy.
        
        Args:
            text: Input text to chunk
            strategy: Chunking strategy ("semantic_paragraph", "sentence", "fixed_size")
            metadata: Additional metadata to attach to chunks
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        logger.info(f"Chunking document using {strategy} strategy")
        logger.info(f"Input text length: {len(text)} characters")
        
        if strategy == "semantic_paragraph":
            chunks = self._semantic_paragraph_chunking(text, metadata)
        elif strategy == "sentence":
            chunks = self._sentence_based_chunking(text, metadata)
        elif strategy == "fixed_size":
            chunks = self._fixed_size_chunking(text, metadata)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        
        # Filter out chunks that are too small
        valid_chunks = [chunk for chunk in chunks if len(chunk.text) >= self.min_chunk_size]
        
        logger.info(f"Created {len(valid_chunks)} valid chunks from {len(chunks)} total chunks")
        return valid_chunks
    
    def _semantic_paragraph_chunking(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """
        Chunk based on semantic paragraphs with context preservation.
        
        This method is optimized for Bengali literature where paragraph
        boundaries often represent semantic units.
        """
        chunks = []
        
        # Split into paragraphs
        paragraphs = self._split_into_paragraphs(text)
        
        current_chunk = ""
        chunk_counter = 0
        start_char = 0
        
        for i, paragraph in enumerate(paragraphs):
            # Check if adding this paragraph would exceed chunk size
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if len(potential_chunk) <= self.chunk_size or not current_chunk:
                current_chunk = potential_chunk
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunk = self._create_chunk(
                        text=current_chunk,
                        chunk_id=f"chunk_{chunk_counter}",
                        start_char=start_char,
                        end_char=start_char + len(current_chunk),
                        metadata=metadata
                    )
                    chunks.append(chunk)
                    chunk_counter += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
                start_char = start_char + len(current_chunk) - len(overlap_text) if overlap_text else start_char + len(current_chunk)
        
        # Add the last chunk
        if current_chunk:
            chunk = self._create_chunk(
                text=current_chunk,
                chunk_id=f"chunk_{chunk_counter}",
                start_char=start_char,
                end_char=start_char + len(current_chunk),
                metadata=metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _sentence_based_chunking(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """
        Chunk based on sentences with context preservation.
        
        Uses NLTK sentence tokenizer with Bengali language support.
        """
        chunks = []
        
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        current_chunk = ""
        chunk_counter = 0
        start_char = 0
        
        for sentence in sentences:
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.chunk_size or not current_chunk:
                current_chunk = potential_chunk
            else:
                # Save current chunk
                if current_chunk:
                    chunk = self._create_chunk(
                        text=current_chunk,
                        chunk_id=f"sent_chunk_{chunk_counter}",
                        start_char=start_char,
                        end_char=start_char + len(current_chunk),
                        metadata=metadata
                    )
                    chunks.append(chunk)
                    chunk_counter += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                start_char = start_char + len(current_chunk) - len(overlap_text) if overlap_text else start_char + len(current_chunk)
        
        # Add the last chunk
        if current_chunk:
            chunk = self._create_chunk(
                text=current_chunk,
                chunk_id=f"sent_chunk_{chunk_counter}",
                start_char=start_char,
                end_char=start_char + len(current_chunk),
                metadata=metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _fixed_size_chunking(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """
        Fixed-size chunking with overlap.
        
        Simple but effective chunking strategy that maintains consistent chunk sizes.
        """
        chunks = []
        chunk_counter = 0
        
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to break at word boundary
            if end < len(text):
                # Look for the last space within reasonable distance
                last_space = text.rfind(' ', start, end)
                if last_space > start + self.chunk_size * 0.8:  # At least 80% of target size
                    end = last_space
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk = self._create_chunk(
                    text=chunk_text,
                    chunk_id=f"fixed_chunk_{chunk_counter}",
                    start_char=start,
                    end_char=end,
                    metadata=metadata
                )
                chunks.append(chunk)
                chunk_counter += 1
            
            # Move start position with overlap
            start = max(start + self.chunk_size - self.chunk_overlap, end)
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs, handling Bengali text properly."""
        # Split on double newlines or more
        paragraphs = re.split(r'\n\s*\n+', text)
        
        # Clean and filter paragraphs
        cleaned_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if para and len(para) > 20:  # Filter out very short paragraphs
                cleaned_paragraphs.append(para)
        
        return cleaned_paragraphs
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from the end of current chunk."""
        if len(text) <= overlap_size:
            return text
        
        # Try to break at sentence boundary for better context
        overlap_text = text[-overlap_size:]
        
        # Look for sentence ending in the overlap
        sentence_endings = ['.', '।', '?', '!']  # Include Bengali sentence ending
        for ending in sentence_endings:
            last_sentence = overlap_text.rfind(ending)
            if last_sentence > overlap_size * 0.5:  # At least half the overlap size
                return overlap_text[last_sentence + 1:].strip()
        
        # If no good sentence boundary, try word boundary
        last_space = overlap_text.rfind(' ')
        if last_space > overlap_size * 0.7:
            return overlap_text[last_space:].strip()
        
        return overlap_text
    
    def _create_chunk(self, 
                     text: str, 
                     chunk_id: str,
                     start_char: int,
                     end_char: int,
                     metadata: Optional[Dict] = None) -> Chunk:
        """Create a Chunk object with computed metadata."""
        word_count = len(word_tokenize(text))
        
        chunk_metadata = {
            'char_count': len(text),
            'word_count': word_count,
            'chunk_strategy': 'semantic_paragraph'
        }
        
        if metadata:
            chunk_metadata.update(metadata)
        
        return Chunk(
            text=text,
            chunk_id=chunk_id,
            start_char=start_char,
            end_char=end_char,
            word_count=word_count,
            metadata=chunk_metadata
        )
    
    def analyze_chunks(self, chunks: List[Chunk]) -> Dict:
        """Analyze chunk statistics for optimization."""
        if not chunks:
            return {}
        
        chunk_sizes = [len(chunk.text) for chunk in chunks]
        word_counts = [chunk.word_count for chunk in chunks if chunk.word_count]
        
        analysis = {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'avg_word_count': sum(word_counts) / len(word_counts) if word_counts else 0,
            'total_characters': sum(chunk_sizes),
            'size_distribution': {
                'small_chunks': len([s for s in chunk_sizes if s < 200]),
                'medium_chunks': len([s for s in chunk_sizes if 200 <= s < 400]),
                'large_chunks': len([s for s in chunk_sizes if s >= 400])
            }
        }
        
        return analysis


def main():
    """Test the chunker with sample text."""
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    
    # Sample Bengali text for testing
    sample_text = """
    অনুপম একজন সাধারণ মানুষ ছিল। তার জীবনে অনেক ঘটনা ঘটেছিল। 
    সে তার মামার কাছে অনেক কিছু শিখেছিল। মামা ছিলেন তার ভাগ্য দেবতা।
    
    শুম্ভুনাথ ছিলেন একজন সুপুরুষ। অনুপমের ভাষায় তিনি ছিলেন আদর্শ মানুষ।
    তার চরিত্রে অনেক ভালো গুণ ছিল।
    
    কল্যাণীর বিয়ের সময় তার বয়স ছিল মাত্র পনেরো বছর। 
    সে ছিল খুবই সুন্দর এবং বুদ্ধিমতী একটি মেয়ে।
    """
    
    chunks = chunker.chunk_document(sample_text, strategy="semantic_paragraph")
    
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} (ID: {chunk.chunk_id}):")
        print(f"Text: {chunk.text[:100]}...")
        print(f"Length: {len(chunk.text)} characters")
        print(f"Word count: {chunk.word_count}")
    
    # Analyze chunks
    analysis = chunker.analyze_chunks(chunks)
    print(f"\nChunk Analysis: {analysis}")


if __name__ == "__main__":
    main()