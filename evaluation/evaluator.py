"""
Comprehensive RAG System Evaluator

Main evaluation orchestrator that combines all metrics and provides
detailed analysis of RAG system performance.
"""

import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path

from .metrics import (
    GroundednessMetric, RelevanceMetric, AnswerQualityMetric, PerformanceMetric
)
from .test_datasets import create_test_dataset, get_dataset_statistics

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Comprehensive evaluator for RAG system performance.
    
    Combines multiple metrics to provide detailed analysis of:
    - Groundedness: How well answers are supported by context
    - Relevance: Quality of document retrieval
    - Answer Quality: Overall quality of generated responses
    - Performance: Speed and efficiency metrics
    """
    
    def __init__(self, rag_system, output_dir: str = "evaluation_results"):
        """
        Initialize the evaluator.
        
        Args:
            rag_system: The RAG system to evaluate
            output_dir: Directory to save evaluation results
        """
        self.rag_system = rag_system
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize metrics
        self.groundedness_metric = GroundednessMetric()
        self.relevance_metric = RelevanceMetric()
        self.answer_quality_metric = AnswerQualityMetric()
        self.performance_metric = PerformanceMetric()
        
        # Evaluation history
        self.evaluation_history = []
        
        logger.info(f"RAG Evaluator initialized with output directory: {output_dir}")
    
    def evaluate_single_query(self, 
                             query: str, 
                             expected_answer: str = None,
                             category: str = "general") -> Dict[str, Any]:
        """
        Evaluate a single query.
        
        Args:
            query: User query
            expected_answer: Expected answer (optional)
            category: Question category
            
        Returns:
            Detailed evaluation results
        """
        start_time = time.time()
        
        try:
            # Get response from RAG system
            response = self.rag_system.query(query)
            processing_time = time.time() - start_time
            
            # Calculate metrics
            groundedness_score = self.groundedness_metric.calculate(
                query=query,
                answer=response.answer,
                retrieved_chunks=response.retrieved_chunks
            )
            
            relevance_score = self.relevance_metric.calculate(
                query=query,
                retrieved_chunks=response.retrieved_chunks
            )
            
            quality_metrics = self.answer_quality_metric.calculate(
                query=query,
                answer=response.answer,
                language=response.query_language,
                expected_answer=expected_answer
            )
            
            # Compile results
            evaluation_result = {
                'query': query,
                'answer': response.answer,
                'expected_answer': expected_answer,
                'category': category,
                'query_language': response.query_language,
                'processing_time': processing_time,
                'retrieved_chunks_count': len(response.retrieved_chunks),
                'metrics': {
                    'groundedness': groundedness_score,
                    'relevance': relevance_score,
                    'system_confidence': response.confidence_score,
                    **quality_metrics
                },
                'retrieved_chunks': [
                    {
                        'text': chunk['document'][:200] + '...' if len(chunk['document']) > 200 else chunk['document'],
                        'similarity_score': chunk['similarity_score'],
                        'metadata': chunk.get('metadata', {})
                    }
                    for chunk in response.retrieved_chunks
                ],
                'timestamp': datetime.now().isoformat()
            }
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error evaluating query '{query}': {e}")
            return {
                'query': query,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def comprehensive_evaluation(self, 
                                test_queries: List[Dict[str, Any]] = None,
                                save_results: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive evaluation on a set of test queries.
        
        Args:
            test_queries: List of test queries. If None, uses default dataset
            save_results: Whether to save results to files
            
        Returns:
            Comprehensive evaluation results
        """
        if test_queries is None:
            logger.info("Using default HSC Bangla test dataset")
            test_queries = create_test_dataset()
        
        logger.info(f"Starting comprehensive evaluation with {len(test_queries)} queries")
        
        # Initialize results structure
        results = {
            'evaluation_id': f"eval_{int(time.time())}",
            'timestamp': datetime.now().isoformat(),
            'total_queries': len(test_queries),
            'dataset_statistics': get_dataset_statistics(test_queries),
            'individual_results': [],
            'aggregate_metrics': {},
            'category_performance': {},
            'performance_analysis': {},
            'system_info': self.rag_system.get_system_stats()
        }
        
        # Track metrics for aggregation
        all_groundedness = []
        all_relevance = []
        all_confidence = []
        all_processing_times = []
        all_quality_metrics = {
            'length_score': [],
            'language_consistency': [],
            'query_relevance': [],
            'completeness': [],
            'coherence': [],
            'overall_quality': []
        }
        
        category_metrics = {}
        
        # Evaluate each query
        for i, test_item in enumerate(test_queries):
            query = test_item['query']
            expected_answer = test_item.get('expected_answer', '')
            category = test_item.get('category', 'general')
            
            logger.info(f"Evaluating {i+1}/{len(test_queries)}: {query[:50]}...")
            
            # Evaluate single query
            eval_result = self.evaluate_single_query(
                query=query,
                expected_answer=expected_answer,
                category=category
            )
            
            # Skip if error occurred
            if 'error' in eval_result:
                logger.warning(f"Skipping query due to error: {eval_result['error']}")
                continue
            
            results['individual_results'].append(eval_result)
            
            # Collect metrics for aggregation
            metrics = eval_result['metrics']
            all_groundedness.append(metrics['groundedness'])
            all_relevance.append(metrics['relevance'])
            all_confidence.append(metrics['system_confidence'])
            all_processing_times.append(eval_result['processing_time'])
            
            # Collect quality metrics
            for metric_name in all_quality_metrics:
                if metric_name in metrics:
                    all_quality_metrics[metric_name].append(metrics[metric_name])
            
            # Category-wise metrics
            if category not in category_metrics:
                category_metrics[category] = {
                    'groundedness': [],
                    'relevance': [],
                    'confidence': [],
                    'processing_time': [],
                    'overall_quality': []
                }
            
            category_metrics[category]['groundedness'].append(metrics['groundedness'])
            category_metrics[category]['relevance'].append(metrics['relevance'])
            category_metrics[category]['confidence'].append(metrics['system_confidence'])
            category_metrics[category]['processing_time'].append(eval_result['processing_time'])
            if 'overall_quality' in metrics:
                category_metrics[category]['overall_quality'].append(metrics['overall_quality'])
        
        # Calculate aggregate metrics
        if all_groundedness:  # Only if we have successful evaluations
            results['aggregate_metrics'] = {
                'avg_groundedness': self._safe_mean(all_groundedness),
                'avg_relevance': self._safe_mean(all_relevance),
                'avg_confidence': self._safe_mean(all_confidence),
                'avg_processing_time': self._safe_mean(all_processing_times),
                'total_processing_time': sum(all_processing_times),
                'queries_per_second': len(all_processing_times) / sum(all_processing_times) if sum(all_processing_times) > 0 else 0,
                'successful_evaluations': len(all_groundedness),
                'success_rate': len(all_groundedness) / len(test_queries)
            }
            
            # Add quality metrics to aggregate
            for metric_name, values in all_quality_metrics.items():
                if values:
                    results['aggregate_metrics'][f'avg_{metric_name}'] = self._safe_mean(values)
        
        # Calculate category performance
        for category, metrics in category_metrics.items():
            results['category_performance'][category] = {
                'count': len(metrics['groundedness']),
                'avg_groundedness': self._safe_mean(metrics['groundedness']),
                'avg_relevance': self._safe_mean(metrics['relevance']),
                'avg_confidence': self._safe_mean(metrics['confidence']),
                'avg_processing_time': self._safe_mean(metrics['processing_time']),
                'avg_overall_quality': self._safe_mean(metrics['overall_quality']) if metrics['overall_quality'] else 0.0
            }
        
        # Performance analysis
        if all_processing_times:
            results['performance_analysis'] = self.performance_metric.calculate(all_processing_times)
        
        # Add to evaluation history
        self.evaluation_history.append(results)
        
        # Save results if requested
        if save_results:
            self._save_evaluation_results(results)
        
        logger.info(f"Comprehensive evaluation completed. Success rate: {results['aggregate_metrics'].get('success_rate', 0):.2%}")
        
        return results
    
    def compare_evaluations(self, 
                           evaluation_ids: List[str] = None) -> Dict[str, Any]:
        """
        Compare multiple evaluation results.
        
        Args:
            evaluation_ids: List of evaluation IDs to compare. If None, compares all
            
        Returns:
            Comparison analysis
        """
        if not self.evaluation_history:
            logger.warning("No evaluation history available for comparison")
            return {}
        
        evaluations_to_compare = self.evaluation_history
        if evaluation_ids:
            evaluations_to_compare = [
                eval_result for eval_result in self.evaluation_history
                if eval_result['evaluation_id'] in evaluation_ids
            ]
        
        if len(evaluations_to_compare) < 2:
            logger.warning("Need at least 2 evaluations for comparison")
            return {}
        
        comparison = {
            'comparison_timestamp': datetime.now().isoformat(),
            'evaluations_compared': len(evaluations_to_compare),
            'metric_trends': {},
            'performance_trends': {},
            'category_trends': {}
        }
        
        # Compare aggregate metrics
        metrics_to_compare = ['avg_groundedness', 'avg_relevance', 'avg_confidence', 'avg_processing_time']
        
        for metric in metrics_to_compare:
            values = [
                eval_result['aggregate_metrics'].get(metric, 0)
                for eval_result in evaluations_to_compare
                if 'aggregate_metrics' in eval_result
            ]
            
            if values:
                comparison['metric_trends'][metric] = {
                    'values': values,
                    'trend': 'improving' if values[-1] > values[0] else 'declining',
                    'change': values[-1] - values[0] if len(values) > 1 else 0,
                    'best_value': max(values),
                    'worst_value': min(values)
                }
        
        return comparison
    
    def generate_evaluation_report(self, 
                                  evaluation_results: Dict[str, Any],
                                  include_individual_results: bool = False) -> str:
        """
        Generate a detailed evaluation report.
        
        Args:
            evaluation_results: Results from comprehensive_evaluation
            include_individual_results: Whether to include individual query results
            
        Returns:
            Formatted evaluation report
        """
        report_lines = []
        
        # Header
        report_lines.append("# RAG System Evaluation Report")
        report_lines.append(f"Generated: {evaluation_results['timestamp']}")
        report_lines.append(f"Evaluation ID: {evaluation_results['evaluation_id']}")
        report_lines.append("")
        
        # Summary
        report_lines.append("## Executive Summary")
        agg_metrics = evaluation_results.get('aggregate_metrics', {})
        
        report_lines.append(f"- **Total Queries Evaluated**: {evaluation_results['total_queries']}")
        report_lines.append(f"- **Success Rate**: {agg_metrics.get('success_rate', 0):.1%}")
        report_lines.append(f"- **Average Groundedness**: {agg_metrics.get('avg_groundedness', 0):.3f}")
        report_lines.append(f"- **Average Relevance**: {agg_metrics.get('avg_relevance', 0):.3f}")
        report_lines.append(f"- **Average Confidence**: {agg_metrics.get('avg_confidence', 0):.3f}")
        report_lines.append(f"- **Average Processing Time**: {agg_metrics.get('avg_processing_time', 0):.3f}s")
        report_lines.append(f"- **Queries per Second**: {agg_metrics.get('queries_per_second', 0):.2f}")
        report_lines.append("")
        
        # Dataset Statistics
        dataset_stats = evaluation_results.get('dataset_statistics', {})
        if dataset_stats:
            report_lines.append("## Dataset Statistics")
            report_lines.append(f"- **Total Questions**: {dataset_stats.get('total_questions', 0)}")
            report_lines.append(f"- **Average Query Length**: {dataset_stats.get('avg_query_length', 0):.1f} words")
            
            # Categories
            categories = dataset_stats.get('categories', {})
            if categories:
                report_lines.append("- **Categories**:")
                for category, count in categories.items():
                    report_lines.append(f"  - {category}: {count}")
            
            # Languages
            languages = dataset_stats.get('languages_detected', {})
            if languages:
                report_lines.append("- **Languages**:")
                for lang, count in languages.items():
                    report_lines.append(f"  - {lang}: {count}")
            
            report_lines.append("")
        
        # Category Performance
        category_perf = evaluation_results.get('category_performance', {})
        if category_perf:
            report_lines.append("## Performance by Category")
            report_lines.append("")
            
            for category, metrics in category_perf.items():
                report_lines.append(f"### {category.title()}")
                report_lines.append(f"- **Count**: {metrics['count']}")
                report_lines.append(f"- **Groundedness**: {metrics['avg_groundedness']:.3f}")
                report_lines.append(f"- **Relevance**: {metrics['avg_relevance']:.3f}")
                report_lines.append(f"- **Confidence**: {metrics['avg_confidence']:.3f}")
                report_lines.append(f"- **Processing Time**: {metrics['avg_processing_time']:.3f}s")
                if metrics.get('avg_overall_quality', 0) > 0:
                    report_lines.append(f"- **Overall Quality**: {metrics['avg_overall_quality']:.3f}")
                report_lines.append("")
        
        # Performance Analysis
        perf_analysis = evaluation_results.get('performance_analysis', {})
        if perf_analysis:
            report_lines.append("## Performance Analysis")
            report_lines.append(f"- **Average Processing Time**: {perf_analysis.get('avg_processing_time', 0):.3f}s")
            report_lines.append(f"- **Median Processing Time**: {perf_analysis.get('median_processing_time', 0):.3f}s")
            report_lines.append(f"- **Min Processing Time**: {perf_analysis.get('min_processing_time', 0):.3f}s")
            report_lines.append(f"- **Max Processing Time**: {perf_analysis.get('max_processing_time', 0):.3f}s")
            report_lines.append(f"- **Standard Deviation**: {perf_analysis.get('std_processing_time', 0):.3f}s")
            report_lines.append("")
        
        # System Information
        system_info = evaluation_results.get('system_info', {})
        if system_info:
            report_lines.append("## System Information")
            
            vector_store = system_info.get('vector_store', {})
            if vector_store:
                report_lines.append(f"- **Total Documents**: {vector_store.get('total_documents', 0)}")
                report_lines.append(f"- **Collection Name**: {vector_store.get('collection_name', 'N/A')}")
            
            memory = system_info.get('memory', {})
            if memory:
                report_lines.append(f"- **Total Conversations**: {memory.get('total_conversations', 0)}")
                report_lines.append(f"- **Total Interactions**: {memory.get('total_interactions', 0)}")
            
            embedding_model = system_info.get('embedding_model', {})
            if embedding_model:
                report_lines.append(f"- **Embedding Model**: {embedding_model.get('model_name', 'N/A')}")
            
            report_lines.append("")
        
        # Individual Results (if requested)
        if include_individual_results:
            individual_results = evaluation_results.get('individual_results', [])
            if individual_results:
                report_lines.append("## Individual Query Results")
                report_lines.append("")
                
                for i, result in enumerate(individual_results[:10]):  # Show first 10
                    report_lines.append(f"### Query {i+1}")
                    report_lines.append(f"**Question**: {result['query']}")
                    report_lines.append(f"**Answer**: {result['answer'][:200]}{'...' if len(result['answer']) > 200 else ''}")
                    report_lines.append(f"**Category**: {result['category']}")
                    report_lines.append(f"**Language**: {result['query_language']}")
                    
                    metrics = result['metrics']
                    report_lines.append(f"**Metrics**:")
                    report_lines.append(f"- Groundedness: {metrics['groundedness']:.3f}")
                    report_lines.append(f"- Relevance: {metrics['relevance']:.3f}")
                    report_lines.append(f"- Confidence: {metrics['system_confidence']:.3f}")
                    
                    if result.get('expected_answer'):
                        report_lines.append(f"**Expected Answer**: {result['expected_answer']}")
                    
                    report_lines.append("")
                
                if len(individual_results) > 10:
                    report_lines.append(f"... and {len(individual_results) - 10} more results")
                    report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        
        if agg_metrics.get('avg_groundedness', 0) < 0.7:
            report_lines.append("- **Low Groundedness**: Consider improving context retrieval or answer generation")
        
        if agg_metrics.get('avg_relevance', 0) < 0.6:
            report_lines.append("- **Low Relevance**: Consider tuning embedding model or retrieval parameters")
        
        if agg_metrics.get('avg_processing_time', 0) > 2.0:
            report_lines.append("- **Slow Processing**: Consider optimizing retrieval or generation pipeline")
        
        if agg_metrics.get('success_rate', 0) < 0.9:
            report_lines.append("- **Low Success Rate**: Investigate and fix errors in the pipeline")
        
        report_lines.append("")
        report_lines.append("---")
        report_lines.append(f"Report generated by RAG Evaluator v1.0")
        
        return "\n".join(report_lines)
    
    def _safe_mean(self, values: List[float]) -> float:
        """Calculate mean safely, handling empty lists."""
        return sum(values) / len(values) if values else 0.0
    
    def _save_evaluation_results(self, results: Dict[str, Any]):
        """Save evaluation results to files."""
        eval_id = results['evaluation_id']
        
        # Save detailed results as JSON
        json_path = self.output_dir / f"{eval_id}_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Save summary report as markdown
        report = self.generate_evaluation_report(results, include_individual_results=False)
        report_path = self.output_dir / f"{eval_id}_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Save detailed report with individual results
        detailed_report = self.generate_evaluation_report(results, include_individual_results=True)
        detailed_path = self.output_dir / f"{eval_id}_detailed_report.md"
        with open(detailed_path, 'w', encoding='utf-8') as f:
            f.write(detailed_report)
        
        logger.info(f"Evaluation results saved:")
        logger.info(f"  - JSON: {json_path}")
        logger.info(f"  - Report: {report_path}")
        logger.info(f"  - Detailed Report: {detailed_path}")


# Example usage
if __name__ == "__main__":
    # This would be used with an actual RAG system
    print("RAG Evaluator module loaded successfully!")
    print("Usage: evaluator = RAGEvaluator(rag_system)")
    print("       results = evaluator.comprehensive_evaluation()")