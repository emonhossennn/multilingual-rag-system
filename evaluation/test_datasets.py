"""
Test Datasets for RAG Evaluation

Provides curated test datasets for evaluating the multilingual RAG system.
"""

from typing import List, Dict, Any
import json
from pathlib import Path


class HSCBanglaTestDataset:
    """Test dataset specifically for HSC Bangla 1st Paper content."""
    
    @staticmethod
    def get_character_analysis_questions() -> List[Dict[str, Any]]:
        """Questions about character analysis from Bengali literature."""
        return [
            {
                "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
                "expected_answer": "শুম্ভুনাথ",
                "category": "character_analysis",
                "difficulty": "easy",
                "source": "অপরিচিতা - রবীন্দ্রনাথ ঠাকুর"
            },
            {
                "query": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
                "expected_answer": "মামা",
                "category": "character_analysis", 
                "difficulty": "easy",
                "source": "অপরিচিতা - রবীন্দ্রনাথ ঠাকুর"
            },
            {
                "query": "অনুপমের চরিত্রের বৈশিষ্ট্য কী?",
                "expected_answer": "দুর্বল, নির্ভরশীল, সিদ্ধান্তহীন",
                "category": "character_analysis",
                "difficulty": "medium",
                "source": "অপরিচিতা - রবীন্দ্রনাথ ঠাকুর"
            },
            {
                "query": "কল্যাণীর চরিত্রে কোন গুণগুলো প্রকাশ পেয়েছে?",
                "expected_answer": "আত্মমর্যাদাবোধ, স্বাধীনচেতা, বুদ্ধিমতী",
                "category": "character_analysis",
                "difficulty": "medium",
                "source": "অপরিচিতা - রবীন্দ্রনাথ ঠাকুর"
            }
        ]
    
    @staticmethod
    def get_factual_questions() -> List[Dict[str, Any]]:
        """Factual questions requiring specific information."""
        return [
            {
                "query": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
                "expected_answer": "১৫ বছর",
                "category": "factual",
                "difficulty": "easy",
                "source": "অপরিচিতা - রবীন্দ্রনাথ ঠাকুর"
            },
            {
                "query": "অপরিচিতা গল্পটি কোন সালে প্রকাশিত হয়?",
                "expected_answer": "১৯১৬ সাল",
                "category": "factual",
                "difficulty": "hard",
                "source": "সাহিত্যের ইতিহাস"
            },
            {
                "query": "রবীন্দ্রনাথ ঠাকুর কোন সালে নোবেল পুরস্কার পান?",
                "expected_answer": "১৯১৩ সাল",
                "category": "factual",
                "difficulty": "medium",
                "source": "সাহিত্যের ইতিহাস"
            }
        ]
    
    @staticmethod
    def get_grammar_questions() -> List[Dict[str, Any]]:
        """Grammar and language-related questions."""
        return [
            {
                "query": "সন্ধি কাকে বলে?",
                "expected_answer": "দুটি বর্ণের মিলনকে সন্ধি বলে",
                "category": "grammar",
                "difficulty": "easy",
                "source": "বাংলা ব্যাকরণ"
            },
            {
                "query": "উপসর্গের কাজ কী?",
                "expected_answer": "নতুন অর্থ সৃষ্টি করা",
                "category": "grammar",
                "difficulty": "medium",
                "source": "বাংলা ব্যাকরণ"
            },
            {
                "query": "সমাস কত প্রকার ও কী কী?",
                "expected_answer": "ছয় প্রকার: দ্বন্দ্ব, কর্মধারয়, তৎপুরুষ, বহুব্রীহি, অব্যয়ীভাব, দ্বিগু",
                "category": "grammar",
                "difficulty": "hard",
                "source": "বাংলা ব্যাকরণ"
            }
        ]
    
    @staticmethod
    def get_literature_history_questions() -> List[Dict[str, Any]]:
        """Questions about Bengali literature history."""
        return [
            {
                "query": "বাংলা সাহিত্যের যুগ বিভাগ কেমন?",
                "expected_answer": "প্রাচীন যুগ, মধ্য যুগ, আধুনিক যুগ",
                "category": "literature_history",
                "difficulty": "medium",
                "source": "সাহিত্যের ইতিহাস"
            },
            {
                "query": "আধুনিক যুগের সূচনা কীভাবে হয়?",
                "expected_answer": "ফোর্ট উইলিয়াম কলেজ প্রতিষ্ঠার মাধ্যমে",
                "category": "literature_history",
                "difficulty": "hard",
                "source": "সাহিত্যের ইতিহাস"
            },
            {
                "query": "কাজী নজরুল ইসলামের প্রধান কাব্যগ্রন্থ কোনগুলো?",
                "expected_answer": "অগ্নিবীণা, বিষের বাঁশি, ভাঙার গান",
                "category": "literature_history",
                "difficulty": "medium",
                "source": "সাহিত্যের ইতিহাস"
            }
        ]
    
    @staticmethod
    def get_translation_questions() -> List[Dict[str, Any]]:
        """Cross-language questions for testing multilingual capability."""
        return [
            {
                "query": "Who is referred to as a good man in Anupam's language?",
                "expected_answer": "Shumbhunath",
                "category": "translation",
                "difficulty": "easy",
                "source": "Aparichita - Rabindranath Tagore"
            },
            {
                "query": "What was Kalyani's age at the time of marriage?",
                "expected_answer": "15 years",
                "category": "translation",
                "difficulty": "easy",
                "source": "Aparichita - Rabindranath Tagore"
            },
            {
                "query": "Who is mentioned as Anupam's fortune deity?",
                "expected_answer": "Uncle (Mama)",
                "category": "translation",
                "difficulty": "easy",
                "source": "Aparichita - Rabindranath Tagore"
            }
        ]
    
    @staticmethod
    def get_thematic_questions() -> List[Dict[str, Any]]:
        """Questions about themes and literary analysis."""
        return [
            {
                "query": "অপরিচিতা গল্পের মূল বিষয়বস্তু কী?",
                "expected_answer": "নারীর আত্মমর্যাদাবোধ ও সামাজিক কুসংস্কার",
                "category": "thematic_analysis",
                "difficulty": "medium",
                "source": "অপরিচিতা - রবীন্দ্রনাথ ঠাকুর"
            },
            {
                "query": "গল্পে যৌতুক প্রথার সমালোচনা কীভাবে করা হয়েছে?",
                "expected_answer": "কল্যাণীর বিয়ে ভেঙে যাওয়ার মাধ্যমে",
                "category": "thematic_analysis",
                "difficulty": "hard",
                "source": "অপরিচিতা - রবীন্দ্রনাথ ঠাকুর"
            }
        ]


def create_test_dataset(categories: List[str] = None, difficulty: str = None) -> List[Dict[str, Any]]:
    """
    Create a comprehensive test dataset.
    
    Args:
        categories: List of categories to include (optional)
        difficulty: Difficulty level filter (easy/medium/hard) (optional)
        
    Returns:
        List of test questions
    """
    dataset = HSCBanglaTestDataset()
    
    # Get all question types
    all_questions = []
    all_questions.extend(dataset.get_character_analysis_questions())
    all_questions.extend(dataset.get_factual_questions())
    all_questions.extend(dataset.get_grammar_questions())
    all_questions.extend(dataset.get_literature_history_questions())
    all_questions.extend(dataset.get_translation_questions())
    all_questions.extend(dataset.get_thematic_questions())
    
    # Filter by categories if specified
    if categories:
        all_questions = [q for q in all_questions if q['category'] in categories]
    
    # Filter by difficulty if specified
    if difficulty:
        all_questions = [q for q in all_questions if q['difficulty'] == difficulty]
    
    return all_questions


def create_custom_test_dataset(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create a custom test dataset from user-provided questions.
    
    Args:
        questions: List of question dictionaries
        
    Returns:
        Validated and formatted test dataset
    """
    formatted_questions = []
    
    for i, question in enumerate(questions):
        # Ensure required fields
        formatted_question = {
            "query": question.get("query", ""),
            "expected_answer": question.get("expected_answer", ""),
            "category": question.get("category", "custom"),
            "difficulty": question.get("difficulty", "medium"),
            "source": question.get("source", "custom"),
            "id": question.get("id", f"custom_{i}")
        }
        
        # Validate required fields
        if not formatted_question["query"]:
            continue
        
        formatted_questions.append(formatted_question)
    
    return formatted_questions


def save_test_dataset(dataset: List[Dict[str, Any]], filepath: str):
    """Save test dataset to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)


def load_test_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Load test dataset from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_dataset_statistics(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get statistics about a test dataset."""
    if not dataset:
        return {}
    
    # Count by category
    categories = {}
    difficulties = {}
    sources = {}
    
    for question in dataset:
        category = question.get('category', 'unknown')
        difficulty = question.get('difficulty', 'unknown')
        source = question.get('source', 'unknown')
        
        categories[category] = categories.get(category, 0) + 1
        difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
        sources[source] = sources.get(source, 0) + 1
    
    return {
        'total_questions': len(dataset),
        'categories': categories,
        'difficulties': difficulties,
        'sources': sources,
        'avg_query_length': sum(len(q['query'].split()) for q in dataset) / len(dataset),
        'languages_detected': _detect_languages(dataset)
    }


def _detect_languages(dataset: List[Dict[str, Any]]) -> Dict[str, int]:
    """Detect languages in the dataset."""
    languages = {'bengali': 0, 'english': 0, 'mixed': 0}
    
    for question in dataset:
        query = question.get('query', '')
        bengali_chars = sum(1 for char in query if '\u0980' <= char <= '\u09FF')
        total_chars = len(query)
        
        if total_chars == 0:
            continue
        
        bengali_ratio = bengali_chars / total_chars
        
        if bengali_ratio > 0.5:
            languages['bengali'] += 1
        elif bengali_ratio < 0.1:
            languages['english'] += 1
        else:
            languages['mixed'] += 1
    
    return languages


# Example usage and testing
if __name__ == "__main__":
    # Create a sample dataset
    dataset = create_test_dataset()
    
    print(f"Created dataset with {len(dataset)} questions")
    
    # Get statistics
    stats = get_dataset_statistics(dataset)
    print(f"Dataset statistics: {stats}")
    
    # Save dataset
    save_test_dataset(dataset, "hsc_bangla_test_dataset.json")
    print("Dataset saved to hsc_bangla_test_dataset.json")
    
    # Show sample questions
    print("\nSample questions:")
    for i, question in enumerate(dataset[:3]):
        print(f"{i+1}. {question['query']}")
        print(f"   Expected: {question['expected_answer']}")
        print(f"   Category: {question['category']}")
        print()