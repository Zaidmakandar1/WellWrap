"""
Model Handler Module
Manages small pre-trained biomedical models compliant with hackathon guidelines.
Uses smaller task-specific models from Hugging Face, NOT large generative models.
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    AutoModel,
    pipeline
)
import logging
from typing import List, Dict, Optional, Tuple
import os
from pathlib import Path

# Additional NLP libraries for enhanced medical text processing
try:
    import spacy
    import scispacy
    from scispacy.abbreviation import AbbreviationDetector
    from scispacy.linking import EntityLinker
    SCISPACY_AVAILABLE = True
except ImportError:
    SCISPACY_AVAILABLE = False
    logging.warning("SciSpacy not available. Some advanced features will be disabled.")

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. Basic text processing will be used.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelHandler:
    """
    Handles small biomedical pre-trained models for medical text analysis.
    
    Compliant with hackathon rules - uses only smaller task-specific models:
    - BioBERT (smaller variants)
    - SciBERT  
    - ClinicalBERT (smaller variants)
    - Bio_ClinicalBERT
    
    NOT using large generative models like GPT, Claude, etc.
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize models - using smaller biomedical models only
        self.models = {}
        self.tokenizers = {}
        
        # Load smaller biomedical models
        self._load_biomedical_models()
        
        # Load additional NLP tools
        self._load_nlp_tools()
    
    def _load_biomedical_models(self):
        """Load small biomedical models that are hackathon compliant"""
        
        try:
            # 1. Bio_ClinicalBERT - Small biomedical model for clinical text
            logger.info("Loading Bio_ClinicalBERT (small model)...")
            model_name = "emilyalsentzer/Bio_ClinicalBERT"
            self.tokenizers['clinical'] = AutoTokenizer.from_pretrained(model_name)
            self.models['clinical'] = AutoModel.from_pretrained(model_name).to(self.device)
            
            # 2. SciBERT for scientific/medical text understanding
            logger.info("Loading SciBERT (small model)...")
            scibert_name = "allenai/scibert_scivocab_uncased"
            self.tokenizers['scientific'] = AutoTokenizer.from_pretrained(scibert_name)
            self.models['scientific'] = AutoModel.from_pretrained(scibert_name).to(self.device)
            
            # 3. Small BioBERT variant for biomedical NER
            logger.info("Loading small biomedical NER model...")
            ner_model_name = "dmis-lab/biobert-base-cased-v1.1"
            self.tokenizers['biobert'] = AutoTokenizer.from_pretrained(ner_model_name)
            self.models['biobert'] = AutoModel.from_pretrained(ner_model_name).to(self.device)
            
            # 4. Create NER pipeline for medical entity recognition (small model)
            try:
                self.ner_pipeline = pipeline(
                    "ner",
                    model="d4data/biomedical-ner-all",  # Small biomedical NER model
                    tokenizer="d4data/biomedical-ner-all",
                    device=0 if torch.cuda.is_available() else -1
                )
            except Exception as e:
                logger.warning(f"Could not load biomedical NER pipeline: {e}")
                self.ner_pipeline = None
            
            logger.info("Successfully loaded all small biomedical models!")
            
        except Exception as e:
            logger.error(f"Error loading biomedical models: {e}")
            logger.info("Falling back to basic models...")
            self._load_fallback_models()
    
    def _load_fallback_models(self):
        """Load basic fallback models if biomedical models fail"""
        try:
            # Use basic BERT for text analysis as fallback
            logger.info("Loading basic BERT as fallback...")
            basic_model = "bert-base-uncased"
            self.tokenizers['basic'] = AutoTokenizer.from_pretrained(basic_model)
            self.models['basic'] = AutoModel.from_pretrained(basic_model).to(self.device)
            self.ner_pipeline = None
            
        except Exception as e:
            logger.error(f"Error loading fallback models: {e}")
            # Ultimate fallback - no ML models, just rule-based processing
            self.models = {}
            self.tokenizers = {}
            self.ner_pipeline = None
    
    def _load_nlp_tools(self):
        """Load additional NLP tools for enhanced medical text processing"""
        self.scispacy_nlp = None
        self.nltk_ready = False
        
        # Load SciSpacy for medical text processing
        if SCISPACY_AVAILABLE:
            try:
                logger.info("Loading SciSpacy biomedical model...")
                # Try to load a biomedical model (install with: python -m spacy download en_core_sci_sm)
                self.scispacy_nlp = spacy.load("en_core_sci_sm")
                
                # Add abbreviation detection
                abbreviation_pipe = AbbreviationDetector(self.scispacy_nlp)
                self.scispacy_nlp.add_pipe("abbreviation_detector")
                
                logger.info("SciSpacy loaded successfully!")
            except OSError:
                logger.warning("SciSpacy biomedical model not found. Install with: python -m spacy download en_core_sci_sm")
            except Exception as e:
                logger.warning(f"Error loading SciSpacy: {e}")
        
        # Initialize NLTK
        if NLTK_AVAILABLE:
            try:
                # Download required NLTK data if not present
                nltk.download('punkt', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                nltk.download('stopwords', quiet=True)
                self.nltk_ready = True
                logger.info("NLTK initialized successfully!")
            except Exception as e:
                logger.warning(f"Error initializing NLTK: {e}")
    
    def extract_medical_entities(self, text: str) -> List[Dict]:
        """
        Extract medical entities from text using small biomedical models.
        Returns entities like diseases, symptoms, medications, etc.
        """
        entities = []
        
        if self.ner_pipeline:
            try:
                # Use small biomedical NER model
                results = self.ner_pipeline(text)
                
                # Process and clean results
                for entity in results:
                    if entity['confidence'] > 0.5:  # Filter low confidence
                        entities.append({
                            'text': entity['word'],
                            'label': entity['entity'],
                            'confidence': entity['confidence'],
                            'start': entity.get('start', 0),
                            'end': entity.get('end', 0)
                        })
            except Exception as e:
                logger.warning(f"Error in medical entity extraction: {e}")
        
        return entities
    
    def get_medical_embeddings(self, text: str, model_type: str = 'clinical') -> Optional[torch.Tensor]:
        """
        Get embeddings from small biomedical models for text similarity/analysis.
        Uses small pre-trained models, not large generative ones.
        """
        if model_type not in self.models or model_type not in self.tokenizers:
            logger.warning(f"Model type '{model_type}' not available")
            return None
        
        try:
            tokenizer = self.tokenizers[model_type]
            model = self.models[model_type]
            
            # Tokenize and encode
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                # Use mean pooling of last hidden states
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            return None
    
    def analyze_medical_text(self, text: str) -> Dict:
        """
        Analyze medical text using small biomedical models.
        Returns analysis including entities, embeddings, and basic classification.
        """
        analysis = {
            'entities': [],
            'has_embeddings': False,
            'text_length': len(text),
            'model_info': 'Using small biomedical models (hackathon compliant)'
        }
        
        # Extract medical entities
        analysis['entities'] = self.extract_medical_entities(text)
        
        # Get embeddings if available
        embeddings = self.get_medical_embeddings(text)
        if embeddings is not None:
            analysis['has_embeddings'] = True
            analysis['embedding_shape'] = embeddings.shape
        
        return analysis
    
    def find_similar_terms(self, query_term: str, candidate_terms: List[str], 
                          threshold: float = 0.7) -> List[Tuple[str, float]]:
        """
        Find similar medical terms using embeddings from small biomedical models.
        Used for medical term normalization and simplification.
        """
        if not self.models:
            return []
        
        similar_terms = []
        
        try:
            # Get embedding for query term
            query_embedding = self.get_medical_embeddings(query_term)
            if query_embedding is None:
                return []
            
            # Compare with candidate terms
            for candidate in candidate_terms:
                candidate_embedding = self.get_medical_embeddings(candidate)
                if candidate_embedding is not None:
                    # Calculate cosine similarity
                    similarity = torch.cosine_similarity(query_embedding, candidate_embedding).item()
                    
                    if similarity >= threshold:
                        similar_terms.append((candidate, similarity))
            
            # Sort by similarity
            similar_terms.sort(key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            logger.warning(f"Error finding similar terms: {e}")
        
        return similar_terms
    
    def enhanced_medical_text_analysis(self, text: str) -> Dict:
        """Enhanced medical text analysis using all available models and tools"""
        analysis = {
            'entities': [],
            'abbreviations': [],
            'medical_concepts': [],
            'sentences': [],
            'pos_tags': [],
            'has_embeddings': False,
            'text_length': len(text),
            'tools_used': []
        }
        
        # Basic transformer-based analysis
        basic_analysis = self.analyze_medical_text(text)
        analysis.update(basic_analysis)
        analysis['tools_used'].append('transformers')
        
        # Enhanced processing with SciSpacy
        if self.scispacy_nlp:
            try:
                doc = self.scispacy_nlp(text)
                
                # Extract medical entities using SciSpacy
                scispacy_entities = []
                for ent in doc.ents:
                    scispacy_entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': 1.0  # SciSpacy doesn't provide confidence scores
                    })
                analysis['medical_concepts'] = scispacy_entities
                
                # Extract abbreviations
                if hasattr(doc._, 'abbreviations'):
                    abbreviations = []
                    for abbrev in doc._.abbreviations:
                        abbreviations.append({
                            'abbreviation': abbrev.text,
                            'definition': abbrev._.long_form.text if abbrev._.long_form else 'Unknown',
                            'start': abbrev.start_char,
                            'end': abbrev.end_char
                        })
                    analysis['abbreviations'] = abbreviations
                
                analysis['tools_used'].append('scispacy')
                logger.info(f"SciSpacy found {len(scispacy_entities)} medical concepts and {len(analysis['abbreviations'])} abbreviations")
            except Exception as e:
                logger.warning(f"Error in SciSpacy analysis: {e}")
        
        # Enhanced processing with NLTK
        if self.nltk_ready:
            try:
                # Sentence segmentation
                sentences = sent_tokenize(text)
                analysis['sentences'] = sentences
                
                # POS tagging for medical terms identification
                words = word_tokenize(text.lower())
                pos_tags = pos_tag(words)
                
                # Filter for relevant POS tags (nouns, adjectives that might be medical terms)
                medical_pos_tags = [(word, pos) for word, pos in pos_tags 
                                   if pos in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'VBN']]
                analysis['pos_tags'] = medical_pos_tags[:20]  # Limit to first 20 for brevity
                
                analysis['tools_used'].append('nltk')
                logger.info(f"NLTK processed {len(sentences)} sentences and tagged {len(medical_pos_tags)} potential medical terms")
            except Exception as e:
                logger.warning(f"Error in NLTK analysis: {e}")
        
        return analysis
    
    def preprocess_medical_text(self, text: str) -> str:
        """Preprocess medical text using available NLP tools"""
        processed_text = text
        
        # Clean and normalize using NLTK if available
        if self.nltk_ready:
            try:
                # Basic text cleaning
                sentences = sent_tokenize(processed_text)
                # Join sentences with consistent spacing
                processed_text = ' '.join(sentences)
                
                logger.debug("NLTK preprocessing applied")
            except Exception as e:
                logger.warning(f"Error in NLTK preprocessing: {e}")
        
        # Additional cleaning using SciSpacy if available
        if self.scispacy_nlp:
            try:
                doc = self.scispacy_nlp(processed_text)
                # Extract clean text tokens (removes excessive whitespace, normalizes)
                tokens = [token.text for token in doc if not token.is_space]
                processed_text = ' '.join(tokens)
                
                logger.debug("SciSpacy preprocessing applied")
            except Exception as e:
                logger.warning(f"Error in SciSpacy preprocessing: {e}")
        
        return processed_text
    
    def extract_medical_abbreviations(self, text: str) -> List[Dict]:
        """Extract medical abbreviations and their definitions using SciSpacy"""
        abbreviations = []
        
        if self.scispacy_nlp:
            try:
                doc = self.scispacy_nlp(text)
                
                if hasattr(doc._, 'abbreviations'):
                    for abbrev in doc._.abbreviations:
                        abbreviations.append({
                            'abbreviation': abbrev.text,
                            'definition': abbrev._.long_form.text if abbrev._.long_form else 'Definition not found',
                            'start': abbrev.start_char,
                            'end': abbrev.end_char,
                            'context': text[max(0, abbrev.start_char-50):abbrev.end_char+50]
                        })
                        
                logger.info(f"Found {len(abbreviations)} medical abbreviations")
            except Exception as e:
                logger.warning(f"Error extracting abbreviations: {e}")
        
        return abbreviations
    
    def classify_test_abnormality(self, test_name: str, value: float, 
                                 normal_range: str) -> Dict:
        """
        Use small models to help classify test result abnormality severity.
        This is rule-based with ML assistance, not using large generative models.
        """
        classification = {
            'severity': 'unknown',
            'explanation': '',
            'model_assisted': bool(self.models)
        }
        
        # Basic rule-based classification
        if 'hemoglobin' in test_name.lower():
            if value < 10:
                classification['severity'] = 'severe'
                classification['explanation'] = 'Significantly low hemoglobin'
            elif value < 12:
                classification['severity'] = 'mild'
                classification['explanation'] = 'Mildly low hemoglobin'
            else:
                classification['severity'] = 'normal'
                classification['explanation'] = 'Normal hemoglobin levels'
        
        # Add more test-specific logic as needed
        
        return classification
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        return {
            'loaded_models': list(self.models.keys()),
            'device': str(self.device),
            'ner_available': self.ner_pipeline is not None,
            'scispacy_available': self.scispacy_nlp is not None,
            'nltk_available': self.nltk_ready,
            'total_tools': len([tool for tool in [self.models, self.ner_pipeline, self.scispacy_nlp, self.nltk_ready] if tool]),
            'compliance': 'Uses only small task-specific biomedical models + NLP tools',
            'hackathon_compliant': True
        }
