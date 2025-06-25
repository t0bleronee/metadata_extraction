"""
Metadata Generator Module for Automated Document Analysis
Converts NLP analysis results into structured, standardized metadata formats
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Structured metadata for a document"""
    # Basic document information
    document_id: str
    filename: str
    file_size: int
    file_type: str
    processing_date: str
    
    # Content analysis
    title: Optional[str]
    summary: str
    language: str
    
    # Text statistics
    word_count: int
    sentence_count: int
    paragraph_count: int
    reading_time_minutes: int
    
    # Topics and keywords
    primary_topics: List[str]
    keywords: List[str]
    entities: Dict[str, List[str]]
    
    # Document characteristics
    document_type: str
    formality_level: str
    complexity_level: str
    sentiment: str
    
    # Technical metadata
    readability_score: float
    readability_level: str
    structure_score: float
    
    # Additional fields
    tags: List[str]
    categories: List[str]
    confidence_score: float

class MetadataGenerator:
    """
    Generates structured metadata from document processing and NLP analysis results
    """
    
    def __init__(self):
        """Initialize the metadata generator"""
        logger.info("Initializing Metadata Generator...")
        
        # Document type classification patterns
        self.document_type_patterns = {
            'research_paper': ['abstract', 'methodology', 'conclusion', 'references', 'hypothesis'],
            'business_report': ['executive summary', 'revenue', 'quarterly', 'analysis', 'recommendations'],
            'technical_document': ['system', 'architecture', 'implementation', 'configuration', 'api'],
            'legal_document': ['agreement', 'contract', 'terms', 'conditions', 'liability'],
            'manual': ['instructions', 'steps', 'procedure', 'guide', 'how to'],
            'presentation': ['slide', 'agenda', 'overview', 'introduction', 'thank you'],
            'article': ['article', 'author', 'published', 'journal', 'volume'],
            'proposal': ['proposal', 'budget', 'timeline', 'objectives', 'deliverables']
        }
        
        # Category mapping based on topics
        self.category_mapping = {
            'technology': ['Information Technology', 'Software Development', 'Digital Innovation'],
            'business': ['Business Strategy', 'Finance', 'Marketing', 'Management'],
            'research': ['Academic Research', 'Scientific Study', 'Data Analysis'],
            'medical': ['Healthcare', 'Medical Research', 'Clinical Studies'],
            'legal': ['Legal Documents', 'Contracts', 'Compliance'],
            'education': ['Educational Content', 'Training Materials', 'Academic'],
            'finance': ['Financial Analysis', 'Investment', 'Banking']
        }
        
        logger.info("✅ Metadata Generator initialized!")
    
    def generate_metadata(self, 
                         document_result: Dict, 
                         nlp_result: Dict,
                         custom_tags: List[str] = None) -> DocumentMetadata:
        """
        Generate comprehensive metadata from processing results
        
        Args:
            document_result: Output from DocumentProcessor
            nlp_result: Output from NLPAnalyzer
            custom_tags: Optional custom tags to add
            
        Returns:
            DocumentMetadata object with all extracted information
        """
        logger.info(f"Generating metadata for: {document_result.get('file_metadata', {}).get('filename', 'unknown')}")
        
        # Extract basic information
        file_meta = document_result.get('file_metadata', {})
        text_stats = document_result.get('text_statistics', {})
        
        # Generate unique document ID
        doc_id = str(uuid.uuid4())[:8]
        
        # Extract title (heuristic approach)
        title = self._extract_title(document_result.get('processed_text', ''))
        
        # Determine document type
        doc_type = self._classify_document_type(
            document_result.get('processed_text', ''), 
            nlp_result
        )
        
        # Extract and process topics
        topics = self._process_topics(nlp_result.get('topics', []))
        
        # Extract keywords
        keywords = self._process_keywords(nlp_result.get('keywords', []))
        
        # Process entities
        entities = self._process_entities(nlp_result.get('entities', {}))
        
        # Determine categories
        categories = self._determine_categories(topics, entities, keywords)
        
        # Generate tags
        tags = self._generate_tags(topics, keywords, doc_type, custom_tags or [])
        
        # Calculate scores
        confidence_score = self._calculate_confidence_score(nlp_result)
        structure_score = self._calculate_structure_score(nlp_result.get('document_structure', {}))
        
        # Determine complexity and formality
        complexity_level = self._determine_complexity_level(nlp_result)
        formality_level = self._determine_formality_level(nlp_result)
        
        # Create metadata object
        metadata = DocumentMetadata(
            # Basic information
            document_id=doc_id,
            filename=file_meta.get('filename', 'unknown'),
            file_size=file_meta.get('file_size', 0),
            file_type=file_meta.get('file_extension', '').upper().replace('.', ''),
            processing_date=datetime.now().isoformat(),
            
            # Content
            title=title,
            summary=nlp_result.get('summary', ''),
            language=text_stats.get('detected_language', 'unknown'),
            
            # Statistics
            word_count=text_stats.get('word_count', 0),
            sentence_count=text_stats.get('sentence_count', 0),
            paragraph_count=text_stats.get('paragraph_count', 0),
            reading_time_minutes=text_stats.get('estimated_reading_time_minutes', 0),
            
            # Analysis results
            primary_topics=topics[:5],  # Top 5 topics
            keywords=keywords[:10],     # Top 10 keywords
            entities=entities,
            
            # Classification
            document_type=doc_type,
            formality_level=formality_level,
            complexity_level=complexity_level,
            sentiment=nlp_result.get('sentiment', {}).get('sentiment', 'neutral'),
            
            # Technical scores
            readability_score=nlp_result.get('readability', {}).get('flesch_score', 0),
            readability_level=nlp_result.get('readability', {}).get('reading_level', 'unknown'),
            structure_score=structure_score,
            
            # Organization
            tags=tags,
            categories=categories,
            confidence_score=confidence_score
        )
        
        logger.info("✅ Metadata generation completed")
        return metadata
    
    def _extract_title(self, text: str) -> Optional[str]:
        """Extract document title using heuristics"""
        if not text:
            return None
        
        lines = text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line and len(line) > 5 and len(line) < 100:
                # Check if it looks like a title
                if (line.isupper() or 
                    line.count(' ') < 10 or 
                    any(char in line for char in [':', '=', '-']) or
                    not line.endswith('.')):
                    return line
        
        # Fallback: use first sentence if it's short enough
        sentences = text.split('.')
        if sentences and len(sentences[0]) < 100:
            return sentences[0].strip()
        
        return None
    
    def _classify_document_type(self, text: str, nlp_result: Dict) -> str:
        """Classify document type based on content patterns"""
        text_lower = text.lower()
        
        # Score each document type
        type_scores = {}
        for doc_type, patterns in self.document_type_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            if score > 0:
                type_scores[doc_type] = score
        
        # Also consider structure
        structure = nlp_result.get('document_structure', {})
        if structure.get('has_title') and structure.get('has_headings'):
            type_scores['report'] = type_scores.get('report', 0) + 2
        
        # Return highest scoring type or 'document' as default
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0].replace('_', ' ').title()
        
        return 'Document'
    
    def _process_topics(self, topics: List) -> List[str]:
        """Process and clean topic information"""
        processed_topics = []
        for topic in topics:
            if hasattr(topic, 'topic'):
                topic_name = topic.topic
            elif isinstance(topic, dict):
                topic_name = topic.get('topic', '')
            else:
                topic_name = str(topic)
            
            if topic_name and topic_name.lower() != 'general':
                processed_topics.append(topic_name.title())
        
        return processed_topics
    
    def _process_keywords(self, keywords: List) -> List[str]:
        """Process and clean keyword information"""
        processed_keywords = []
        for keyword in keywords:
            if isinstance(keyword, tuple):
                word = keyword[0]
            elif isinstance(keyword, dict):
                word = keyword.get('word', '')
            else:
                word = str(keyword)
            
            if word and len(word) > 3:
                processed_keywords.append(word.lower())
        
        return processed_keywords
    
    def _process_entities(self, entities: Dict) -> Dict[str, List[str]]:
        """Process and clean entity information"""
        processed_entities = {}
        
        for entity_type, entity_list in entities.items():
            clean_entities = []
            for entity in entity_list:
                if hasattr(entity, 'text'):
                    entity_text = entity.text
                elif isinstance(entity, dict):
                    entity_text = entity.get('text', '')
                else:
                    entity_text = str(entity)
                
                if entity_text:
                    clean_entities.append(entity_text)
            
            if clean_entities:
                processed_entities[entity_type.replace('_', ' ').title()] = clean_entities[:5]  # Limit to 5 per type
        
        return processed_entities
    
    def _determine_categories(self, topics: List[str], entities: Dict, keywords: List[str]) -> List[str]:
        """Determine document categories based on content analysis"""
        categories = set()
        
        # Category from topics
        for topic in topics:
            topic_lower = topic.lower()
            for domain, cats in self.category_mapping.items():
                if domain in topic_lower:
                    categories.update(cats[:2])  # Add first 2 categories
        
        # Category from entities
        if 'Email' in entities or 'Phone' in entities:
            categories.add('Contact Information')
        if 'Organization' in entities:
            categories.add('Organizational')
        if 'Date' in entities:
            categories.add('Time-Sensitive')
        
        # Category from keywords
        keyword_str = ' '.join(keywords).lower()
        if any(word in keyword_str for word in ['report', 'analysis', 'study']):
            categories.add('Analytical')
        if any(word in keyword_str for word in ['policy', 'procedure', 'guideline']):
            categories.add('Procedural')
        
        return list(categories)[:5]  # Limit to 5 categories
    
    def _generate_tags(self, topics: List[str], keywords: List[str], doc_type: str, custom_tags: List[str]) -> List[str]:
        """Generate relevant tags for the document"""
        tags = set()
        
        # Add topic-based tags
        tags.update([topic.lower().replace(' ', '_') for topic in topics])
        
        # Add document type tag
        tags.add(doc_type.lower().replace(' ', '_'))
        
        # Add high-frequency keyword tags
        tags.update(keywords[:5])
        
        # Add custom tags
        tags.update([tag.lower().replace(' ', '_') for tag in custom_tags])
        
        # Clean and return
        clean_tags = [tag for tag in tags if len(tag) > 2 and len(tag) < 20]
        return sorted(clean_tags)[:10]  # Limit to 10 tags
    
    def _calculate_confidence_score(self, nlp_result: Dict) -> float:
        """Calculate overall confidence score for the metadata"""
        scores = []
        
        # Topic confidence
        topics = nlp_result.get('topics', [])
        if topics:
            topic_confidences = []
            for topic in topics:
                if hasattr(topic, 'confidence'):
                    topic_confidences.append(topic.confidence)
                elif isinstance(topic, dict):
                    topic_confidences.append(topic.get('confidence', 0.5))
            if topic_confidences:
                scores.append(sum(topic_confidences) / len(topic_confidences))
        
        # Entity confidence (pattern-based entities have high confidence)
        entities = nlp_result.get('entities', {})
        if entities:
            scores.append(0.8)  # High confidence for pattern-based extraction
        
        # Readability confidence
        readability = nlp_result.get('readability', {})
        if readability.get('flesch_score', 0) > 0:
            scores.append(0.7)
        
        # Structure confidence
        structure = nlp_result.get('document_structure', {})
        structure_indicators = sum([
            structure.get('has_title', False),
            structure.get('has_headings', False),
            structure.get('sections', 0) > 0
        ])
        scores.append(structure_indicators / 3)
        
        return round(sum(scores) / len(scores) if scores else 0.5, 2)
    
    def _calculate_structure_score(self, structure: Dict) -> float:
        """Calculate document structure score"""
        score = 0
        
        # Points for structural elements
        if structure.get('has_title'):
            score += 0.2
        if structure.get('has_headings'):
            score += 0.3
        if structure.get('has_lists'):
            score += 0.2
        if structure.get('sections', 0) > 0:
            score += 0.2
        if structure.get('paragraphs', 0) > 1:
            score += 0.1
        
        return round(min(score, 1.0), 2)
    
    def _determine_complexity_level(self, nlp_result: Dict) -> str:
        """Determine document complexity level"""
        readability = nlp_result.get('readability', {})
        language_features = nlp_result.get('language_features', {})
        
        flesch_score = readability.get('flesch_score', 50)
        lexical_diversity = language_features.get('lexical_diversity', 0.5)
        complexity_indicators = len(language_features.get('complexity_indicators', []))
        
        # Calculate complexity score
        complexity_score = 0
        
        if flesch_score < 30:
            complexity_score += 3
        elif flesch_score < 50:
            complexity_score += 2
        elif flesch_score < 70:
            complexity_score += 1
        
        if lexical_diversity > 0.7:
            complexity_score += 2
        elif lexical_diversity > 0.5:
            complexity_score += 1
        
        complexity_score += complexity_indicators
        
        if complexity_score >= 5:
            return 'High'
        elif complexity_score >= 3:
            return 'Medium'
        else:
            return 'Low'
    
    def _determine_formality_level(self, nlp_result: Dict) -> str:
        """Determine document formality level"""
        language_features = nlp_result.get('language_features', {})
        formality_score = language_features.get('formality_score', 0.5)
        
        if formality_score > 0.7:
            return 'Formal'
        elif formality_score > 0.3:
            return 'Semi-formal'
        else:
            return 'Informal'
    
    def export_metadata(self, metadata: DocumentMetadata, format: str = 'json') -> str:
        """
        Export metadata in various formats
        
        Args:
            metadata: DocumentMetadata object
            format: Export format ('json', 'xml', 'yaml')
            
        Returns:
            Formatted metadata string
        """
        metadata_dict = asdict(metadata)
        
        if format.lower() == 'json':
            return json.dumps(metadata_dict, indent=2, ensure_ascii=False)
        
        elif format.lower() == 'xml':
            return self._dict_to_xml(metadata_dict, 'document_metadata')
        
        elif format.lower() == 'yaml':
            try:
                import yaml
                return yaml.dump(metadata_dict, default_flow_style=False)
            except ImportError:
                logger.warning("PyYAML not installed, falling back to JSON")
                return json.dumps(metadata_dict, indent=2)
        
        else:
            return json.dumps(metadata_dict, indent=2)
    
    def _dict_to_xml(self, data: Dict, root_name: str) -> str:
        """Convert dictionary to XML format"""
        def dict_to_xml_recursive(d, root):
            xml_str = f"<{root}>\n"
            for key, value in d.items():
                if isinstance(value, dict):
                    xml_str += f"  {dict_to_xml_recursive(value, key)}\n"
                elif isinstance(value, list):
                    xml_str += f"  <{key}>\n"
                    for item in value:
                        if isinstance(item, str):
                            xml_str += f"    <item>{item}</item>\n"
                        else:
                            xml_str += f"    <item>{str(item)}</item>\n"
                    xml_str += f"  </{key}>\n"
                else:
                    xml_str += f"  <{key}>{str(value)}</{key}>\n"
            xml_str += f"</{root}>"
            return xml_str
        
        return dict_to_xml_recursive(data, root_name)
    
    def save_metadata(self, metadata: DocumentMetadata, output_path: str, format: str = 'json'):
        """Save metadata to file"""
        formatted_data = self.export_metadata(metadata, format)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_data)
        
        logger.info(f"Metadata saved to: {output_path}")


# Example usage and testing
if __name__ == "__main__":
    # Test the metadata generator
    generator = MetadataGenerator()
    
    # Mock data for testing (replace with actual results from your modules)
    mock_document_result = {
        'file_metadata': {
            'filename': 'ai_research_paper.pdf',
            'file_size': 2048576,
            'file_extension': '.pdf'
        },
        'text_statistics': {
            'word_count': 3500,
            'sentence_count': 187,
            'paragraph_count': 45,
            'detected_language': 'en',
            'estimated_reading_time_minutes': 18
        },
        'processed_text': 'Artificial Intelligence in Modern Business Applications\n\nThis research paper examines...',
        'processing_status': 'success'
    }
    
    mock_nlp_result = {
        'topics': [
            {'topic': 'Technology', 'confidence': 0.9, 'keywords': ['ai', 'machine', 'learning']},
            {'topic': 'Business', 'confidence': 0.7, 'keywords': ['company', 'revenue', 'market']}
        ],
        'keywords': [('artificial', 25), ('intelligence', 20), ('business', 15), ('technology', 12)],
        'entities': {
            'email': [{'text': 'contact@research.com'}],
            'organization': [{'text': 'TechCorp Inc'}]
        },
        'summary': 'This paper explores AI applications in business environments and their impact on productivity.',
        'sentiment': {'sentiment': 'positive', 'score': 0.3},
        'readability': {'flesch_score': 45.2, 'reading_level': 'fairly_difficult'},
        'document_structure': {'has_title': True, 'has_headings': True, 'sections': 5},
        'language_features': {'lexical_diversity': 0.65, 'formality_score': 0.8, 'complexity_indicators': ['complex_vocabulary']}
    }
    
    # Generate metadata
    metadata = generator.generate_metadata(mock_document_result, mock_nlp_result, ['research', 'ai'])
    
    print("🏷️  Generated Metadata:")
    print("=" * 60)
    print(f"📄 Document: {metadata.filename}")
    print(f"🆔 ID: {metadata.document_id}")
    print(f"📝 Title: {metadata.title}")
    print(f"📊 Type: {metadata.document_type}")
    print(f"🏷️  Topics: {', '.join(metadata.primary_topics)}")
    print(f"🔤 Keywords: {', '.join(metadata.keywords[:5])}")
    print(f"📈 Confidence: {metadata.confidence_score}")
    print(f"🎯 Categories: {', '.join(metadata.categories)}")
    
    # Export to JSON
    json_output = generator.export_metadata(metadata, 'json')
    print(f"\n📤 JSON Export (first 300 chars):")
    print(json_output[:300] + "...")
    
    print("\n✅ Metadata generation test completed!")