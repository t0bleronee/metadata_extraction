"""
NLP Analyzer Module for Document Metadata Generation
Extracts semantic information, entities, topics, and summaries
"""

import re
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
import unicodedata

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EntityInfo:
    """Information about extracted entities"""
    text: str
    entity_type: str
    confidence: float = 1.0
    context: str = ""

@dataclass
class TopicInfo:
    """Information about extracted topics"""
    topic: str
    keywords: List[str]
    confidence: float
    sentences: List[str]

class NLPAnalyzer:
    """
    Advanced NLP analysis for document metadata generation
    Uses rule-based and statistical approaches for robust analysis
    """
    
    def __init__(self):
        """Initialize the NLP analyzer with patterns and vocabularies"""
        logger.info("Initializing NLP Analyzer...")
        
        # Entity recognition patterns
        self.entity_patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
            'url': re.compile(r'https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?'),
            'date': re.compile(r'\b(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b', re.IGNORECASE),
            'currency': re.compile(r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\b\d{1,3}(?:,\d{3})*(?:\.\d{2})? (?:USD|EUR|GBP|CAD)\b'),
            'percentage': re.compile(r'\b\d+(?:\.\d+)?%\b'),
            'organization': re.compile(r'\b(?:Inc|LLC|Corp|Ltd|Company|Organization|University|Institute|Department|Agency|Foundation|Association)\b', re.IGNORECASE),
        }
        
        # Common stop words for topic extraction
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
            'his', 'her', 'its', 'our', 'their', 'myself', 'yourself', 'himself', 'herself', 'itself',
            'ourselves', 'yourselves', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'where',
            'when', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'just'
        }
        
        # Technical/domain keywords that indicate topics
        self.domain_keywords = {
            'technology': ['software', 'hardware', 'computer', 'system', 'application', 'program', 'code', 'algorithm', 'data', 'database', 'network', 'server', 'cloud', 'api', 'framework', 'platform'],
            'business': ['revenue', 'profit', 'sales', 'marketing', 'customer', 'client', 'market', 'strategy', 'business', 'company', 'organization', 'management', 'finance', 'budget', 'cost'],
            'research': ['study', 'research', 'analysis', 'experiment', 'hypothesis', 'methodology', 'results', 'conclusion', 'findings', 'data', 'sample', 'statistical', 'significant'],
            'medical': ['patient', 'treatment', 'diagnosis', 'medical', 'health', 'clinical', 'therapy', 'disease', 'symptom', 'medicine', 'doctor', 'hospital', 'healthcare'],
            'legal': ['contract', 'agreement', 'legal', 'law', 'court', 'case', 'plaintiff', 'defendant', 'attorney', 'lawyer', 'litigation', 'settlement', 'regulation', 'compliance'],
            'education': ['student', 'teacher', 'education', 'learning', 'course', 'curriculum', 'school', 'university', 'academic', 'degree', 'diploma', 'graduate', 'undergraduate'],
            'finance': ['investment', 'portfolio', 'stock', 'bond', 'asset', 'liability', 'equity', 'dividend', 'interest', 'loan', 'credit', 'debt', 'financial', 'banking']
        }
        
        logger.info("✅ NLP Analyzer initialized!")
    
    def analyze_document(self, text: str, filename: str = "") -> Dict:
        """
        Perform comprehensive NLP analysis on document text
        
        Args:
            text: Document text to analyze
            filename: Optional filename for context
            
        Returns:
            Dict containing all analysis results
        """
        logger.info(f"Starting NLP analysis for: {filename or 'text input'}")
        
        # Clean and preprocess text
        clean_text = self._preprocess_text(text)
        sentences = self._split_sentences(clean_text)
        words = self._extract_words(clean_text)
        
        # Perform various analyses
        analysis_results = {
            'entities': self._extract_entities(text),
            'topics': self._extract_topics(clean_text, sentences, words),
            'summary': self._generate_summary(sentences),
            'keywords': self._extract_keywords(words),
            'document_structure': self._analyze_structure(text),
            'sentiment': self._analyze_sentiment(clean_text),
            'readability': self._analyze_readability(sentences, words),
            'language_features': self._analyze_language_features(text, words)
        }
        
        logger.info("✅ NLP analysis completed")
        return analysis_results
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for analysis"""
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove excessive whitespace but preserve paragraph structure
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using rule-based approach"""
        # Simple sentence splitting - handles most cases
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract and clean words from text"""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return [word for word in words if len(word) > 2]
    
    def _extract_entities(self, text: str) -> Dict[str, List[EntityInfo]]:
        """Extract named entities using pattern matching"""
        entities = defaultdict(list)
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                entity_text = match.group().strip()
                if entity_text:  # Avoid empty matches
                    # Get context (20 characters before and after)
                    start = max(0, match.start() - 20)
                    end = min(len(text), match.end() + 20)
                    context = text[start:end].replace('\n', ' ')
                    
                    entities[entity_type].append(EntityInfo(
                        text=entity_text,
                        entity_type=entity_type,
                        confidence=0.9,  # High confidence for pattern matches
                        context=context
                    ))
        
        # Remove duplicates
        for entity_type in entities:
            seen = set()
            unique_entities = []
            for entity in entities[entity_type]:
                if entity.text.lower() not in seen:
                    seen.add(entity.text.lower())
                    unique_entities.append(entity)
            entities[entity_type] = unique_entities
        
        return dict(entities)
    
    def _extract_topics(self, text: str, sentences: List[str], words: List[str]) -> List[TopicInfo]:
        """Extract main topics using keyword analysis and domain detection"""
        topics = []
        
        # Count word frequencies (excluding stop words)
        word_freq = Counter([word for word in words if word not in self.stop_words])
        
        # Identify domain-specific topics
        domain_scores = defaultdict(int)
        domain_keywords_found = defaultdict(list)
        
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword in word_freq:
                    domain_scores[domain] += word_freq[keyword]
                    domain_keywords_found[domain].append(keyword)
        
        # Create topic info for top domains
        for domain, score in sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
            if score > 2:  # Minimum threshold
                relevant_sentences = []
                for sentence in sentences[:10]:  # Check first 10 sentences
                    if any(keyword in sentence.lower() for keyword in domain_keywords_found[domain]):
                        relevant_sentences.append(sentence)
                
                topics.append(TopicInfo(
                    topic=domain.title(),
                    keywords=domain_keywords_found[domain][:5],
                    confidence=min(score / 10, 1.0),
                    sentences=relevant_sentences[:3]
                ))
        
        # Add general high-frequency topics
        top_words = [word for word, count in word_freq.most_common(10) if count > 2]
        if top_words:
            topics.append(TopicInfo(
                topic="General",
                keywords=top_words[:5],
                confidence=0.5,
                sentences=sentences[:2]
            ))
        
        return topics
    
    def _extract_keywords(self, words: List[str]) -> List[Tuple[str, int]]:
        """Extract important keywords with frequency"""
        word_freq = Counter([word for word in words if word not in self.stop_words])
        
        # Filter out very common and very rare words
        total_words = len(words)
        keywords = []
        
        for word, freq in word_freq.most_common(20):
            # Skip if too rare (< 0.1%) or too common (> 5%)
            percentage = freq / total_words
            if 0.001 <= percentage <= 0.05 and len(word) > 3:
                keywords.append((word, freq))
        
        return keywords
    
    def _generate_summary(self, sentences: List[str], max_sentences: int = 3) -> str:
        """Generate extractive summary using sentence scoring"""
        if not sentences:
            return ""
        
        if len(sentences) <= max_sentences:
            return " ".join(sentences)
        
        # Score sentences based on length and position
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = 0
            
            # Position score (earlier sentences are more important)
            position_score = 1.0 - (i / len(sentences))
            score += position_score * 0.3
            
            # Length score (medium length sentences preferred)
            word_count = len(sentence.split())
            if 10 <= word_count <= 30:
                score += 0.4
            elif 5 <= word_count < 10 or 30 < word_count <= 50:
                score += 0.2
            
            # Keyword score - sentences with important words
            important_words = ['important', 'significant', 'key', 'main', 'primary', 'conclusion', 'result']
            if any(word in sentence.lower() for word in important_words):
                score += 0.3
            
            sentence_scores.append((sentence, score))
        
        # Select top sentences
        top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:max_sentences]
        
        # Return in original order
        selected_indices = []
        for sentence, _ in top_sentences:
            try:
                idx = sentences.index(sentence)
                selected_indices.append(idx)
            except ValueError:
                continue
        
        selected_indices.sort()
        summary_sentences = [sentences[i] for i in selected_indices]
        
        return " ".join(summary_sentences)
    
    def _analyze_structure(self, text: str) -> Dict:
        """Analyze document structure"""
        structure = {
            'has_title': False,
            'has_headings': False,
            'has_lists': False,
            'has_tables': False,
            'sections': 0,
            'paragraphs': len(text.split('\n\n'))
        }
        
        # Check for title (first line in caps or with specific patterns)
        lines = text.split('\n')
        if lines:
            first_line = lines[0].strip()
            if (first_line.isupper() or 
                any(char in first_line for char in [':', '=', '-']) and len(first_line) < 100):
                structure['has_title'] = True
        
        # Check for headings (lines with specific patterns)
        heading_patterns = [
            r'^[A-Z][A-Za-z\s]+:$',  # "Section Name:"
            r'^#+\s',                 # Markdown headers
            r'^[IVX]+\.',            # Roman numerals
            r'^\d+\.',               # Numbered sections
        ]
        
        for line in lines:
            line = line.strip()
            if any(re.match(pattern, line) for pattern in heading_patterns):
                structure['has_headings'] = True
                structure['sections'] += 1
        
        # Check for lists
        if re.search(r'^\s*[-*•]\s', text, re.MULTILINE):
            structure['has_lists'] = True
        
        # Check for tables (simple heuristic)
        if '|' in text or '\t' in text:
            structure['has_tables'] = True
        
        return structure
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Basic sentiment analysis using word lists"""
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'outstanding',
            'positive', 'success', 'successful', 'achieve', 'achievement', 'improve', 'improvement',
            'benefit', 'advantage', 'effective', 'efficient', 'valuable', 'important', 'significant'
        }
        
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'poor', 'negative', 'fail', 'failure',
            'problem', 'issue', 'challenge', 'difficult', 'hard', 'impossible', 'wrong',
            'error', 'mistake', 'concern', 'risk', 'threat', 'danger', 'loss', 'decline'
        }
        
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            sentiment = "neutral"
            score = 0.0
        else:
            score = (positive_count - negative_count) / total_sentiment_words
            if score > 0.1:
                sentiment = "positive"
            elif score < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"
        
        return {
            'sentiment': sentiment,
            'score': score,
            'positive_words': positive_count,
            'negative_words': negative_count
        }
    
    def _analyze_readability(self, sentences: List[str], words: List[str]) -> Dict:
        """Calculate readability metrics"""
        if not sentences or not words:
            return {'flesch_score': 0, 'reading_level': 'unknown'}
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Count syllables (rough approximation)
        def count_syllables(word):
            vowels = 'aeiouy'
            syllables = 0
            prev_char_vowel = False
            
            for char in word.lower():
                if char in vowels:
                    if not prev_char_vowel:
                        syllables += 1
                    prev_char_vowel = True
                else:
                    prev_char_vowel = False
            
            return max(1, syllables)  # Every word has at least 1 syllable
        
        total_syllables = sum(count_syllables(word) for word in words)
        avg_syllables_per_word = total_syllables / len(words)
        
        # Flesch Reading Ease Score
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Determine reading level
        if flesch_score >= 90:
            reading_level = "very_easy"
        elif flesch_score >= 80:
            reading_level = "easy"
        elif flesch_score >= 70:
            reading_level = "fairly_easy"
        elif flesch_score >= 60:
            reading_level = "standard"
        elif flesch_score >= 50:
            reading_level = "fairly_difficult"
        elif flesch_score >= 30:
            reading_level = "difficult"
        else:
            reading_level = "very_difficult"
        
        return {
            'flesch_score': round(flesch_score, 1),
            'reading_level': reading_level,
            'avg_sentence_length': round(avg_sentence_length, 1),
            'avg_syllables_per_word': round(avg_syllables_per_word, 2)
        }
    
    def _analyze_language_features(self, text: str, words: List[str]) -> Dict:
        """Analyze various language features"""
        features = {
            'lexical_diversity': 0,
            'formality_score': 0,
            'complexity_indicators': []
        }
        
        # Lexical diversity (unique words / total words)
        if words:
            unique_words = set(words)
            features['lexical_diversity'] = round(len(unique_words) / len(words), 3)
        
        # Formality indicators
        formal_indicators = ['therefore', 'however', 'furthermore', 'moreover', 'consequently', 
                           'nevertheless', 'accordingly', 'thus', 'hence', 'whereas']
        informal_indicators = ['really', 'pretty', 'quite', 'very', 'totally', 'basically', 
                             'actually', 'literally', 'obviously', 'definitely']
        
        formal_count = sum(1 for word in words if word in formal_indicators)
        informal_count = sum(1 for word in words if word in informal_indicators)
        
        if formal_count + informal_count > 0:
            features['formality_score'] = formal_count / (formal_count + informal_count)
        
        # Complexity indicators
        if re.search(r'\b(?:complex|complicated|sophisticated|intricate)\b', text, re.IGNORECASE):
            features['complexity_indicators'].append('complex_vocabulary')
        
        if re.search(r'[;:]', text):
            features['complexity_indicators'].append('complex_punctuation')
        
        # Average word length
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            if avg_word_length > 5:
                features['complexity_indicators'].append('long_words')
        
        return features

# Example usage and testing
if __name__ == "__main__":
    # Test the analyzer
    analyzer = NLPAnalyzer()
    
    sample_text = """
    Artificial Intelligence and Machine Learning Report
    
    Introduction
    This report examines the current state of artificial intelligence and machine learning 
    technologies in business applications. Our research shows significant improvements in 
    automated decision-making systems.
    
    Key Findings
    - Machine learning algorithms have improved accuracy by 25%
    - Implementation costs have decreased by $50,000 annually
    - Customer satisfaction increased to 92%
    
    Methodology
    We analyzed data from 150 companies over 12 months. The study included both quantitative 
    metrics and qualitative feedback from users.
    
    Conclusion
    The results demonstrate that AI technologies provide substantial benefits for modern 
    businesses. We recommend continued investment in these systems.
    
    Contact: research@company.com
    Phone: (555) 123-4567
    """
    
    results = analyzer.analyze_document(sample_text, "ai_report.txt")
    
    print("🔍 NLP Analysis Results:")
    print("=" * 50)
    
    print("\n📊 Entities Found:")
    for entity_type, entities in results['entities'].items():
        if entities:
            print(f"  {entity_type.title()}: {[e.text for e in entities]}")
    
    print(f"\n🏷️ Topics ({len(results['topics'])}):")
    for topic in results['topics']:
        print(f"  - {topic.topic} (confidence: {topic.confidence:.2f})")
        print(f"    Keywords: {', '.join(topic.keywords[:3])}")
    
    print(f"\n📝 Summary:")
    print(f"  {results['summary']}")
    
    print(f"\n🔤 Top Keywords:")
    for keyword, freq in results['keywords'][:5]:
        print(f"  - {keyword}: {freq}")
    
    print(f"\n📖 Readability:")
    print(f"  - Reading level: {results['readability']['reading_level']}")
    print(f"  - Flesch score: {results['readability']['flesch_score']}")
    
    print(f"\n💭 Sentiment: {results['sentiment']['sentiment']} ({results['sentiment']['score']:.2f})")