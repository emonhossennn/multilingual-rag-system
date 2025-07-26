"""
Conversation Memory Management

Handles short-term (conversation history) and long-term (document corpus) memory
for the multilingual RAG system.
"""

import logging
import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Interaction:
    """Represents a single conversation interaction."""
    timestamp: float
    user_query: str
    system_response: str
    retrieved_chunks: List[Dict]
    metadata: Optional[Dict] = None


class ConversationMemory:
    """
    Manages conversation memory for the RAG system.
    
    Features:
    - Short-term memory: Recent conversation history
    - Context window management
    - Conversation persistence
    - Memory cleanup and optimization
    """
    
    def __init__(self, 
                 max_interactions_per_conversation: int = 10,
                 max_context_length: int = 2000,
                 memory_file: Optional[str] = None):
        """
        Initialize conversation memory.
        
        Args:
            max_interactions_per_conversation: Maximum interactions to keep per conversation
            max_context_length: Maximum context length in characters
            memory_file: File to persist memory (optional)
        """
        self.max_interactions = max_interactions_per_conversation
        self.max_context_length = max_context_length
        self.memory_file = memory_file
        
        # Store conversations: conversation_id -> deque of interactions
        self.conversations: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.max_interactions)
        )
        
        # Conversation metadata
        self.conversation_metadata: Dict[str, Dict] = {}
        
        # Load existing memory if file provided
        if self.memory_file:
            self._load_memory()
    
    def add_interaction(self, 
                       conversation_id: str,
                       user_query: str,
                       system_response: str,
                       retrieved_chunks: List[Dict],
                       metadata: Optional[Dict] = None) -> None:
        """
        Add a new interaction to conversation memory.
        
        Args:
            conversation_id: Unique conversation identifier
            user_query: User's query
            system_response: System's response
            retrieved_chunks: Retrieved document chunks
            metadata: Additional metadata
        """
        interaction = Interaction(
            timestamp=time.time(),
            user_query=user_query,
            system_response=system_response,
            retrieved_chunks=retrieved_chunks,
            metadata=metadata or {}
        )
        
        # Add to conversation
        self.conversations[conversation_id].append(interaction)
        
        # Update conversation metadata
        if conversation_id not in self.conversation_metadata:
            self.conversation_metadata[conversation_id] = {
                'created_at': time.time(),
                'total_interactions': 0,
                'languages_used': set()
            }
        
        self.conversation_metadata[conversation_id]['total_interactions'] += 1
        self.conversation_metadata[conversation_id]['last_updated'] = time.time()
        
        # Detect and store language
        language = self._detect_language(user_query)
        self.conversation_metadata[conversation_id]['languages_used'].add(language)
        
        logger.debug(f"Added interaction to conversation {conversation_id}")
        
        # Persist if file specified
        if self.memory_file:
            self._save_memory()
    
    def get_context(self, 
                   conversation_id: str,
                   max_interactions: Optional[int] = None) -> str:
        """
        Get conversation context for a given conversation.
        
        Args:
            conversation_id: Conversation identifier
            max_interactions: Maximum interactions to include (default: all recent)
            
        Returns:
            Formatted context string
        """
        if conversation_id not in self.conversations:
            return ""
        
        interactions = list(self.conversations[conversation_id])
        
        # Limit interactions if specified
        if max_interactions:
            interactions = interactions[-max_interactions:]
        
        # Build context string
        context_parts = []
        total_length = 0
        
        # Add interactions in reverse order (most recent first) until we hit length limit
        for interaction in reversed(interactions):
            interaction_text = f"User: {interaction.user_query}\nAssistant: {interaction.system_response}\n"
            
            if total_length + len(interaction_text) > self.max_context_length:
                break
            
            context_parts.insert(0, interaction_text)
            total_length += len(interaction_text)
        
        return "\n".join(context_parts)
    
    def get_conversation_summary(self, conversation_id: str) -> Dict:
        """
        Get a summary of a conversation.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            Dictionary with conversation summary
        """
        if conversation_id not in self.conversations:
            return {}
        
        interactions = list(self.conversations[conversation_id])
        metadata = self.conversation_metadata.get(conversation_id, {})
        
        # Analyze conversation
        total_queries = len(interactions)
        languages = list(metadata.get('languages_used', set()))
        
        # Get recent topics (simplified - could be enhanced with NLP)
        recent_queries = [i.user_query for i in interactions[-3:]]
        
        return {
            'conversation_id': conversation_id,
            'total_interactions': total_queries,
            'languages_used': languages,
            'created_at': metadata.get('created_at'),
            'last_updated': metadata.get('last_updated'),
            'recent_queries': recent_queries,
            'duration_minutes': (metadata.get('last_updated', 0) - metadata.get('created_at', 0)) / 60
        }
    
    def get_all_conversations(self) -> List[Dict]:
        """Get summaries of all conversations."""
        return [
            self.get_conversation_summary(conv_id)
            for conv_id in self.conversations.keys()
        ]
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """
        Clear a specific conversation.
        
        Args:
            conversation_id: Conversation to clear
            
        Returns:
            True if successful
        """
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            if conversation_id in self.conversation_metadata:
                del self.conversation_metadata[conversation_id]
            
            logger.info(f"Cleared conversation: {conversation_id}")
            
            if self.memory_file:
                self._save_memory()
            
            return True
        
        return False
    
    def clear_all_conversations(self) -> None:
        """Clear all conversations."""
        self.conversations.clear()
        self.conversation_metadata.clear()
        
        logger.info("Cleared all conversations")
        
        if self.memory_file:
            self._save_memory()
    
    def cleanup_old_conversations(self, max_age_days: int = 7) -> int:
        """
        Clean up conversations older than specified days.
        
        Args:
            max_age_days: Maximum age in days
            
        Returns:
            Number of conversations cleaned up
        """
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        conversations_to_remove = []
        
        for conv_id, metadata in self.conversation_metadata.items():
            last_updated = metadata.get('last_updated', 0)
            if current_time - last_updated > max_age_seconds:
                conversations_to_remove.append(conv_id)
        
        # Remove old conversations
        for conv_id in conversations_to_remove:
            self.clear_conversation(conv_id)
        
        logger.info(f"Cleaned up {len(conversations_to_remove)} old conversations")
        return len(conversations_to_remove)
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        total_interactions = sum(
            len(interactions) for interactions in self.conversations.values()
        )
        
        # Language distribution
        all_languages = set()
        for metadata in self.conversation_metadata.values():
            all_languages.update(metadata.get('languages_used', set()))
        
        return {
            'total_conversations': len(self.conversations),
            'total_interactions': total_interactions,
            'languages_used': list(all_languages),
            'avg_interactions_per_conversation': total_interactions / len(self.conversations) if self.conversations else 0,
            'memory_file': self.memory_file
        }
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection."""
        # Check for Bengali characters
        bengali_chars = sum(1 for char in text if '\u0980' <= char <= '\u09FF')
        if bengali_chars > len(text) * 0.1:  # More than 10% Bengali characters
            return 'bengali'
        else:
            return 'english'
    
    def _save_memory(self) -> None:
        """Save memory to file."""
        if not self.memory_file:
            return
        
        try:
            # Convert deques to lists for JSON serialization
            serializable_conversations = {}
            for conv_id, interactions in self.conversations.items():
                serializable_conversations[conv_id] = [
                    asdict(interaction) for interaction in interactions
                ]
            
            # Convert sets to lists in metadata
            serializable_metadata = {}
            for conv_id, metadata in self.conversation_metadata.items():
                serializable_metadata[conv_id] = metadata.copy()
                if 'languages_used' in serializable_metadata[conv_id]:
                    serializable_metadata[conv_id]['languages_used'] = list(
                        serializable_metadata[conv_id]['languages_used']
                    )
            
            data = {
                'conversations': serializable_conversations,
                'metadata': serializable_metadata,
                'saved_at': time.time()
            }
            
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Memory saved to {self.memory_file}")
            
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
    
    def _load_memory(self) -> None:
        """Load memory from file."""
        if not self.memory_file:
            return
        
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Restore conversations
            for conv_id, interactions_data in data.get('conversations', {}).items():
                interactions = deque(maxlen=self.max_interactions)
                for interaction_data in interactions_data:
                    interaction = Interaction(**interaction_data)
                    interactions.append(interaction)
                self.conversations[conv_id] = interactions
            
            # Restore metadata
            for conv_id, metadata in data.get('metadata', {}).items():
                # Convert language lists back to sets
                if 'languages_used' in metadata:
                    metadata['languages_used'] = set(metadata['languages_used'])
                self.conversation_metadata[conv_id] = metadata
            
            logger.info(f"Memory loaded from {self.memory_file}")
            
        except FileNotFoundError:
            logger.info(f"Memory file {self.memory_file} not found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading memory: {e}")


def main():
    """Test the conversation memory."""
    # Initialize memory
    memory = ConversationMemory(memory_file="test_memory.json")
    
    # Test conversation
    conv_id = str(uuid.uuid4())
    
    # Add some interactions
    memory.add_interaction(
        conversation_id=conv_id,
        user_query="অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        system_response="শুম্ভুনাথ",
        retrieved_chunks=[{"document": "test chunk", "similarity": 0.9}]
    )
    
    memory.add_interaction(
        conversation_id=conv_id,
        user_query="Who is Anupam's fortune deity?",
        system_response="Uncle (Mama)",
        retrieved_chunks=[{"document": "test chunk 2", "similarity": 0.8}]
    )
    
    # Get context
    context = memory.get_context(conv_id)
    print(f"Context:\n{context}")
    
    # Get conversation summary
    summary = memory.get_conversation_summary(conv_id)
    print(f"\nConversation Summary: {summary}")
    
    # Get stats
    stats = memory.get_stats()
    print(f"\nMemory Stats: {stats}")


if __name__ == "__main__":
    main()