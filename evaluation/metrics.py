"""
Evaluation Metrics for RAG System

Implements various metrics for evaluating RAG system performance.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class BaseMetric(ABC):
    """Base class for evaluation metrics."""
    
    @abstractmethod
    def calculate(self, **kwargs) -> float:
        """Calculate the metric score."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the metric name."""
        pass


class GroundednessMetric(BaseMetric):
    """
    Evaluates if the answer is grounded in the retrieved context.
    
    Measures how well the generated answer is supported by the retrieved documents.
    """
    
    def calculate(self, query: str, answer: str, retrieved_chunks: List[Dict]) -> float:
        """
        Calculate groundedness score.
        
        Args:
            query: User query
            answer: Generated answer
            retrieved_chunks: Retrieved document chunks
            
        Returns:
            Groundedness score between 0-1
        """
        if not retrieved_chunks or not answer.strip():
            return 0.0
        
        # Combine all retrieved text
        context = " ".join([chunk.get('document', '') for chunk in retrieved_chunks])
        
        if not context.strip():
            return 0.0
        
        # Method 1: Keyword overlap
        answer_words = set(self._tokenize(answer.lower()))
        context_words = set(self._tokenize(context.lower()))
        
        if not answer_words:
            return 0.0
        
        overlap = len(answer_words.intersection(context_words))
        keyword_score = overlap / len(answer_words)
        
        # Method 2: Direct quote detection
        quote_score = self._calculate_quote_score(answer, retrieved_chunks)
        
        # Method 3: Semantic similarity (if embeddings available)
        semantic_score = self._calculate_semantic_similarity(answer, context)
        
        # Combine scores with weights
        final_score = (
            0.4 * keyword_score +
            0.3 * quote_score +
            0.3 * semantic_score
        )
        
        return min(final_score, 1.0)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        # Handle both English and Bengali text
        tokens = re.findall(r'\b\w+\b', text)
        return [token for token in tokens if len(token) > 1]
    
    def _calculate_quote_score(self, answer: str, retrieved_chunks: List[Dict]) -> float:
        """Calculate score based on direct quotes from context."""
        answer_lower = answer.lower()
        quote_score = 0.0
        
        for chunk in retrieved_chunks:
            chunk_text = chunk.get('document', '').lower()
            
            # Check for direct quotes (3+ word sequences)
            chunk_words = self._tokenize(chunk_text)
            answer_words = self._tokenize(answer_lower)
            
            for i in range(len(chunk_words) - 2):
                trigram = ' '.join(chunk_words[i:i+3])
                if trigram in ' '.join(answer_words):
                    quote_score += 0.2
        
        return min(quote_score, 1.0)
    
    def _calculate_semantic_similarity(self, answer: str, context: str) -> float:
        """Calculate semantic similarity between answer and context."""
        try:
            # Simple approach: check if key concepts from context appear in answer
            # This is a placeholder for more sophisticated semantic similarity
            
            # Extract key phrases (simple approach)
            context_phrases = self._extract_key_phrases(context)
            answer_phrases = self._extract_key_phrases(answer)
            
            if not context_phrases or not answer_phrases:
                return 0.0
            
            # Calculate phrase overlap
            overlap = len(set(context_phrases).intersection(set(answer_phrases)))
            return overlap / len(context_phrases) if context_phrases else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text."""
        # Simple approach: extract noun phrases and important words
        words = self._tokenize(text.lower())
        
        # Filter out common words (simple stopword removal)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        key_words = [word for word in words if word not in stopwords and len(word) > 2]
        
        return key_words[:10]  # Return top 10 key words
    
    def get_name(self) -> str:
        return "groundedness"


class RelevanceMetric(BaseMetric):
    """
    Evaluates the relevance of retrieved documents to the query.
    
    Measures how well the retrieval system finds relevant documents.
    """
    
    def calculate(self, query: str, retrieved_chunks: List[Dict]) -> float:
        """
        Calculate relevance score.
        
        Args:
            query: User query
            retrieved_chunks: Retrieved document chunks
            
        Returns:
            Relevance score between 0-1
        """
        if not retrieved_chunks:
            return 0.0
        
        # Method 1: Use similarity scores from retrieval
        similarity_scores = [
            chunk.get('similarity_score', 0.0) 
            for chunk in retrieved_chunks
        ]
        
        if similarity_scores:
            avg_similarity = np.mean(similarity_scores)
        else:
            avg_similarity = 0.0
        
        # Method 2: Query-document keyword overlap
        query_words = set(self._tokenize(query.lower()))
        
        if not query_words:
            return avg_similarity
        
        overlap_scores = []
        for chunk in retrieved_chunks:
            chunk_text = chunk.get('document', '')
            chunk_words = set(self._tokenize(chunk_text.lower()))
            
            if chunk_words:
                overlap = len(query_words.intersection(chunk_words))
                overlap_score = overlap / len(query_words)
                overlap_scores.append(overlap_score)
        
        avg_overlap = np.mean(overlap_scores) if overlap_scores else 0.0
        
        # Method 3: Position-based scoring (earlier results should be more relevant)
        position_scores = []
        for i, chunk in enumerate(retrieved_chunks):
            # Give higher scores to earlier results
            position_score = 1.0 / (i + 1)
            position_scores.append(position_score)
        
        avg_position_score = np.mean(position_scores) if position_scores else 0.0
        
        # Combine scores
        final_score = (
            0.5 * avg_similarity +
            0.3 * avg_overlap +
            0.2 * avg_position_score
        )
        
        return min(final_score, 1.0)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        tokens = re.findall(r'\b\w+\b', text)
        return [token for token in tokens if len(token) > 1]
    
    def get_name(self) -> str:
        return "relevance"


class AnswerQualityMetric(BaseMetric):
    """
    Evaluates the quality of generated answers.
    
    Measures various aspects of answer quality including completeness,
    coherence, and appropriateness.
    """
    
    def calculate(self, query: str, answer: str, language: str, expected_answer: str = None) -> Dict[str, float]:
        """
        Calculate answer quality metrics.
        
        Args:
            query: User query
            answer: Generated answer
            language: Detected language
            expected_answer: Expected answer (if available)
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        # Length appropriateness
        metrics['length_score'] = self._calculate_length_score(answer)
        
        # Language consistency
        metrics['language_consistency'] = self._calculate_language_consistency(answer, language)
        
        # Query relevance
        metrics['query_relevance'] = self._calculate_query_relevance(query, answer)
        
        # Completeness
        metrics['completeness'] = self._calculate_completeness(query, answer)
        
        # Coherence
        metrics['coherence'] = self._calculate_coherence(answer)
        
        # Expected answer similarity (if provided)
        if expected_answer:
            metrics['expected_similarity'] = self._calculate_expected_similarity(answer, expected_answer)
        
        # Overall quality score
        weights = {
            'length_score': 0.15,
            'language_consistency': 0.20,
            'query_relevance': 0.25,
            'completeness': 0.20,
            'coherence': 0.20
        }
        
        overall_score = sum(
            metrics.get(metric, 0.0) * weight 
            for metric, weight in weights.items()
        )
        
        metrics['overall_quality'] = overall_score
        
        return metrics
    
    def _calculate_length_score(self, answer: str) -> float:
        """Calculate appropriateness of answer length."""
        word_count = len(answer.split())
        
        if word_count < 2:
            return 0.1  # Too short
        elif word_count > 150:
            return 0.7  # Too long
        elif 5 <= word_count <= 50:
            return 1.0  # Optimal length
        else:
            return 0.8  # Acceptable length
    
    def _calculate_language_consistency(self, answer: str, expected_language: str) -> float:
        """Calculate language consistency."""
        # Simple language detection based on character sets
        bengali_chars = sum(1 for char in answer if '\u0980' <= char <= '\u09FF')
        total_chars = len(answer)
        
        if total_chars == 0:
            return 0.0
        
        bengali_ratio = bengali_chars / total_chars
        
        if expected_language == 'bengali':
            return bengali_ratio if bengali_ratio > 0.1 else 0.3
        else:  # English
            return (1 - bengali_ratio) if bengali_ratio < 0.1 else 0.3
    
    def _calculate_query_relevance(self, query: str, answer: str) -> float:
        """Calculate how well the answer addresses the query."""
        query_words = set(self._tokenize(query.lower()))
        answer_words = set(self._tokenize(answer.lower()))
        
        if not query_words:
            return 0.0
        
        # Direct word overlap
        overlap = len(query_words.intersection(answer_words))
        direct_relevance = overlap / len(query_words)
        
        # Check for question-answer patterns
        pattern_score = self._check_qa_patterns(query, answer)
        
        return min((direct_relevance + pattern_score) / 2, 1.0)
    
    def _calculate_completeness(self, query: str, answer: str) -> float:
        """Calculate completeness of the answer."""
        # Check if answer seems complete (not cut off)
        if answer.endswith('...') or answer.endswith('।।।'):
            return 0.5
        
        # Check if answer addresses the question type
        query_lower = query.lower()
        answer_lower = answer.lower()
        
        # Question type patterns
        if any(word in query_lower for word in ['কে', 'who', 'কাকে', 'whom']):
            # Should contain a name or person reference
            if any(char.isupper() for char in answer) or 'নাম' in answer_lower:
                return 1.0
            return 0.6
        
        if any(word in query_lower for word in ['কত', 'how many', 'how much', 'বয়স']):
            # Should contain numbers
            if any(char.isdigit() for char in answer) or any(num in answer_lower for num in ['এক', 'দুই', 'তিন', 'চার', 'পাঁচ']):
                return 1.0
            return 0.6
        
        if any(word in query_lower for word in ['কী', 'what', 'কেমন', 'how']):
            # Should be descriptive
            if len(answer.split()) >= 3:
                return 1.0
            return 0.7
        
        return 0.8  # Default completeness
    
    def _calculate_coherence(self, answer: str) -> float:
        """Calculate coherence of the answer."""
        sentences = answer.split('।') if '।' in answer else answer.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return 1.0  # Single sentence is coherent
        
        # Check for repetition
        unique_sentences = set(sentences)
        if len(unique_sentences) < len(sentences):
            return 0.6  # Some repetition
        
        # Check for logical flow (simple heuristic)
        # This is a placeholder for more sophisticated coherence analysis
        return 0.9
    
    def _calculate_expected_similarity(self, answer: str, expected_answer: str) -> float:
        """Calculate similarity with expected answer."""
        answer_words = set(self._tokenize(answer.lower()))
        expected_words = set(self._tokenize(expected_answer.lower()))
        
        if not expected_words:
            return 0.0
        
        overlap = len(answer_words.intersection(expected_words))
        return overlap / len(expected_words)
    
    def _check_qa_patterns(self, query: str, answer: str) -> float:
        """Check for appropriate question-answer patterns."""
        query_lower = query.lower()
        answer_lower = answer.lower()
        
        # Bengali patterns
        if 'কাকে' in query_lower and any(name in answer_lower for name in ['নাথ', 'মামা', 'বাবা']):
            return 0.3
        
        if 'বয়স' in query_lower and any(num in answer for num in ['১', '২', '৩', '৪', '৫', '৬', '৭', '৮', '৯', '০']):
            return 0.3
        
        # English patterns
        if 'who' in query_lower and any(char.isupper() for char in answer):
            return 0.3
        
        if 'age' in query_lower and any(char.isdigit() for char in answer):
            return 0.3
        
        return 0.0
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        tokens = re.findall(r'\b\w+\b', text)
        return [token for token in tokens if len(token) > 1]
    
    def get_name(self) -> str:
        return "answer_quality"


class PerformanceMetric(BaseMetric):
    """
    Evaluates system performance metrics.
    
    Measures processing time, throughput, and resource usage.
    """
    
    def calculate(self, processing_times: List[float]) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            processing_times: List of processing times for queries
            
        Returns:
            Dictionary of performance metrics
        """
        if not processing_times:
            return {}
        
        metrics = {
            'avg_processing_time': np.mean(processing_times),
            'median_processing_time': np.median(processing_times),
            'min_processing_time': np.min(processing_times),
            'max_processing_time': np.max(processing_times),
            'std_processing_time': np.std(processing_times),
            'queries_per_second': len(processing_times) / sum(processing_times) if sum(processing_times) > 0 else 0,
            'total_processing_time': sum(processing_times)
        }
        
        return metrics
    
    def get_name(self) -> str:
        return "performance"