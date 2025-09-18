"""
Medical Simplification Module
Converts complex medical terms and test results into patient-friendly explanations.
Uses rule-based logic combined with small biomedical models.
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from report_processor import MedicalTestResult
from model_handler import ModelHandler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SimplificationResult:
    """Result of medical term simplification"""
    test_name: str
    value: float
    unit: str
    normal_range: str
    status: str
    explanation: str
    health_insight: str
    urgency_level: str  # 'low', 'medium', 'high'

class MedicalSimplifier:
    """
    Simplifies medical reports into patient-friendly language.
    Combines rule-based approach with small biomedical models for better accuracy.
    """
    
    def __init__(self, model_handler: ModelHandler):
        self.model_handler = model_handler
        self.medical_explanations = self._initialize_explanations()
        self.condition_mappings = self._initialize_condition_mappings()
        self.urgency_rules = self._initialize_urgency_rules()
    
    def _initialize_explanations(self) -> Dict[str, Dict]:
        """Initialize medical term explanations database"""
        return {
            # Blood tests
            'hemoglobin': {
                'simple_name': 'Hemoglobin (oxygen-carrying protein)',
                'normal_explanation': 'Your blood has a healthy amount of the protein that carries oxygen throughout your body.',
                'low_explanation': 'Your blood has less oxygen-carrying protein than normal. This might make you feel tired or weak.',
                'high_explanation': 'Your blood has more oxygen-carrying protein than normal. This is less common but may need attention.',
                'function': 'Carries oxygen from your lungs to the rest of your body',
                'common_causes_low': ['iron deficiency', 'blood loss', 'chronic disease'],
                'common_causes_high': ['dehydration', 'lung disease', 'smoking']
            },
            
            'hematocrit': {
                'simple_name': 'Hematocrit (percentage of red blood cells)',
                'normal_explanation': 'The percentage of red blood cells in your blood is normal.',
                'low_explanation': 'You have a lower percentage of red blood cells than normal, which may cause fatigue.',
                'high_explanation': 'You have a higher percentage of red blood cells than normal.',
                'function': 'Shows what percentage of your blood is made up of red blood cells',
                'common_causes_low': ['anemia', 'blood loss', 'malnutrition'],
                'common_causes_high': ['dehydration', 'heart disease', 'lung disease']
            },
            
            'white blood cells': {
                'simple_name': 'White Blood Cells (infection fighters)',
                'normal_explanation': 'Your infection-fighting cells are at normal levels.',
                'low_explanation': 'Your infection-fighting cells are lower than normal. Your body might have trouble fighting infections.',
                'high_explanation': 'Your infection-fighting cells are higher than normal. This might mean your body is fighting an infection or inflammation.',
                'function': 'Fight infections and protect your body from germs',
                'common_causes_low': ['viral infections', 'certain medications', 'bone marrow problems'],
                'common_causes_high': ['bacterial infections', 'stress', 'smoking', 'some medications']
            },
            
            'platelets': {
                'simple_name': 'Platelets (blood clotting cells)',
                'normal_explanation': 'Your blood clotting cells are at normal levels.',
                'low_explanation': 'Your blood clotting cells are lower than normal. You might bruise easily or have trouble stopping bleeding.',
                'high_explanation': 'Your blood clotting cells are higher than normal. This might increase clotting risk.',
                'function': 'Help your blood clot to stop bleeding when you get injured',
                'common_causes_low': ['certain medications', 'autoimmune conditions', 'viral infections'],
                'common_causes_high': ['inflammation', 'cancer', 'iron deficiency']
            },
            
            # Cholesterol tests
            'total cholesterol': {
                'simple_name': 'Total Cholesterol',
                'normal_explanation': 'Your overall cholesterol level is in the healthy range.',
                'low_explanation': 'Your cholesterol is low, which is generally good for heart health.',
                'high_explanation': 'Your cholesterol is higher than recommended. This may increase your risk of heart disease.',
                'function': 'A type of fat in your blood that can affect heart health',
                'common_causes_high': ['diet high in saturated fats', 'genetics', 'lack of exercise', 'obesity']
            },
            
            'ldl cholesterol': {
                'simple_name': 'LDL Cholesterol ("Bad" cholesterol)',
                'normal_explanation': 'Your "bad" cholesterol is at a healthy level.',
                'low_explanation': 'Your "bad" cholesterol is low, which is excellent for heart health.',
                'high_explanation': 'Your "bad" cholesterol is high. This can build up in your arteries and increase heart disease risk.',
                'function': 'The "bad" cholesterol that can clog your arteries',
                'common_causes_high': ['unhealthy diet', 'lack of exercise', 'genetics', 'obesity']
            },
            
            'hdl cholesterol': {
                'simple_name': 'HDL Cholesterol ("Good" cholesterol)',
                'normal_explanation': 'Your "good" cholesterol is at a healthy protective level.',
                'low_explanation': 'Your "good" cholesterol is low. Higher levels help protect against heart disease.',
                'high_explanation': 'Your "good" cholesterol is high, which is excellent for heart protection.',
                'function': 'The "good" cholesterol that helps remove bad cholesterol from your arteries',
                'common_causes_low': ['lack of exercise', 'smoking', 'obesity', 'diabetes']
            },
            
            'triglycerides': {
                'simple_name': 'Triglycerides (blood fats)',
                'normal_explanation': 'The fat levels in your blood are normal.',
                'low_explanation': 'Your blood fat levels are low, which is generally healthy.',
                'high_explanation': 'Your blood fat levels are high. This may increase your risk of heart disease.',
                'function': 'A type of fat in your blood that stores energy',
                'common_causes_high': ['high sugar diet', 'excess calories', 'alcohol', 'diabetes']
            },
            
            # Metabolic tests
            'glucose': {
                'simple_name': 'Blood Sugar (Glucose)',
                'normal_explanation': 'Your blood sugar level is in the normal range.',
                'low_explanation': 'Your blood sugar is lower than normal. This can cause dizziness, shakiness, or confusion.',
                'high_explanation': 'Your blood sugar is higher than normal. This might indicate diabetes or prediabetes.',
                'function': 'The main source of energy for your body\'s cells',
                'common_causes_high': ['diabetes', 'prediabetes', 'stress', 'certain medications'],
                'common_causes_low': ['too much diabetes medication', 'not eating enough', 'excessive exercise']
            },
            
            'creatinine': {
                'simple_name': 'Creatinine (kidney function marker)',
                'normal_explanation': 'Your kidneys are filtering waste normally.',
                'low_explanation': 'Your creatinine is low, which usually isn\'t a concern.',
                'high_explanation': 'Your creatinine is high, which might indicate your kidneys aren\'t filtering waste as well as they should.',
                'function': 'A waste product that shows how well your kidneys are working',
                'common_causes_high': ['kidney disease', 'dehydration', 'certain medications', 'muscle disorders']
            },
            
            # Liver tests
            'alt': {
                'simple_name': 'ALT (liver enzyme)',
                'normal_explanation': 'Your liver enzyme levels are normal.',
                'low_explanation': 'Your liver enzyme is low, which is usually not a concern.',
                'high_explanation': 'Your liver enzyme is elevated, which might indicate liver inflammation or damage.',
                'function': 'An enzyme that shows how well your liver is working',
                'common_causes_high': ['fatty liver', 'hepatitis', 'certain medications', 'alcohol use']
            },
            
            'ast': {
                'simple_name': 'AST (liver and muscle enzyme)',
                'normal_explanation': 'Your liver and muscle enzyme levels are normal.',
                'low_explanation': 'Your enzyme level is low, which is usually not concerning.',
                'high_explanation': 'Your enzyme is elevated, which might indicate liver or muscle problems.',
                'function': 'An enzyme found in your liver and muscles',
                'common_causes_high': ['liver disease', 'muscle damage', 'heart problems', 'certain medications']
            },
            
            # Thyroid
            'tsh': {
                'simple_name': 'TSH (thyroid stimulating hormone)',
                'normal_explanation': 'Your thyroid hormone levels are normal.',
                'low_explanation': 'Your TSH is low, which might mean your thyroid is overactive.',
                'high_explanation': 'Your TSH is high, which might mean your thyroid is underactive.',
                'function': 'Controls your thyroid gland, which regulates metabolism',
                'common_causes_high': ['underactive thyroid', 'certain medications'],
                'common_causes_low': ['overactive thyroid', 'thyroid medication']
            }
        }
    
    def _initialize_condition_mappings(self) -> Dict[str, str]:
        """Map medical conditions to simple explanations"""
        return {
            'anemia': 'low red blood cell count, which can make you feel tired',
            'hyperlipidemia': 'high cholesterol levels in your blood',
            'diabetes': 'high blood sugar levels',
            'hypothyroidism': 'underactive thyroid gland',
            'hyperthyroidism': 'overactive thyroid gland',
            'hepatitis': 'liver inflammation',
            'nephritis': 'kidney inflammation',
            'leukocytosis': 'high white blood cell count',
            'thrombocytopenia': 'low platelet count'
        }
    
    def _initialize_urgency_rules(self) -> Dict[str, Dict]:
        """Define urgency levels for different test results"""
        return {
            'hemoglobin': {
                'high_urgency': {'low': 7.0, 'high': 20.0},
                'medium_urgency': {'low': 10.0, 'high': 18.0},
                'low_urgency': {'low': 12.0, 'high': 16.0}
            },
            'glucose': {
                'high_urgency': {'low': 40.0, 'high': 400.0},
                'medium_urgency': {'low': 60.0, 'high': 250.0},
                'low_urgency': {'low': 70.0, 'high': 100.0}
            },
            'creatinine': {
                'high_urgency': {'low': 0.2, 'high': 5.0},
                'medium_urgency': {'low': 0.4, 'high': 2.0},
                'low_urgency': {'low': 0.6, 'high': 1.3}
            }
        }
    
    def simplify_report(self, test_results: List[MedicalTestResult]) -> List[Dict]:
        """
        Simplify a complete medical report into patient-friendly explanations.
        Returns list of simplified test explanations.
        """
        simplified_results = []
        
        for result in test_results:
            try:
                simplified = self._simplify_single_test(result)
                if simplified:
                    simplified_results.append(simplified)
            except Exception as e:
                logger.error(f"Error simplifying test {result.test_name}: {e}")
                continue
        
        return simplified_results
    
    def _simplify_single_test(self, result: MedicalTestResult) -> Optional[Dict]:
        """Simplify a single test result"""
        test_key = result.test_name.lower()
        
        # Get explanation template
        explanation_data = self.medical_explanations.get(test_key, {})
        
        if not explanation_data:
            # Try fuzzy matching with small models if available
            explanation_data = self._find_similar_explanation(test_key)
        
        # Determine status if not provided
        status = result.status or 'unknown'
        
        # Get appropriate explanation based on status
        if status == 'low':
            explanation = explanation_data.get('low_explanation', 
                f'Your {result.test_name.lower()} is below normal range.')
        elif status == 'high':
            explanation = explanation_data.get('high_explanation', 
                f'Your {result.test_name.lower()} is above normal range.')
        else:
            explanation = explanation_data.get('normal_explanation', 
                f'Your {result.test_name.lower()} is in the normal range.')
        
        # Generate health insight
        health_insight = self._generate_health_insight(result, explanation_data, status)
        
        # Determine urgency
        urgency = self._determine_urgency(result.test_name.lower(), result.value, status)
        
        return {
            'test_name': explanation_data.get('simple_name', result.test_name),
            'value': result.value,
            'unit': result.unit or '',
            'normal_range': result.normal_range or 'Not specified',
            'status': status,
            'explanation': explanation,
            'health_insight': health_insight,
            'urgency_level': urgency,
            'function': explanation_data.get('function', ''),
        }
    
    def _find_similar_explanation(self, test_key: str) -> Dict:
        """Use comprehensive model analysis to find similar test explanations"""
        if not self.model_handler.models:
            return {}
        
        try:
            # Get list of known tests
            known_tests = list(self.medical_explanations.keys())
            
            # Find similar terms using biomedical embeddings
            similar_tests = self.model_handler.find_similar_terms(
                test_key, known_tests, threshold=0.6
            )
            
            if similar_tests:
                best_match = similar_tests[0][0]  # Get the best match
                logger.info(f"Found similar test '{best_match}' for '{test_key}' (similarity: {similar_tests[0][1]:.3f})")
                return self.medical_explanations[best_match]
            
            # Try enhanced analysis if available
            if hasattr(self.model_handler, 'enhanced_medical_text_analysis'):
                enhanced_analysis = self.model_handler.enhanced_medical_text_analysis(test_key)
                logger.info(f"Enhanced analysis used tools: {enhanced_analysis.get('tools_used', [])}")
                
                # Look for medical concepts that might match our known tests
                medical_concepts = enhanced_analysis.get('medical_concepts', [])
                for concept in medical_concepts:
                    concept_text = concept['text'].lower()
                    for known_test in known_tests:
                        if known_test in concept_text or concept_text in known_test:
                            logger.info(f"Found concept match '{concept_text}' -> '{known_test}'")
                            return self.medical_explanations[known_test]
                            
        except Exception as e:
            logger.warning(f"Error in comprehensive similarity matching: {e}")
        
        return {}
    
    def _generate_health_insight(self, result: MedicalTestResult, 
                               explanation_data: Dict, status: str) -> str:
        """Generate additional health insights and recommendations"""
        insights = []
        
        test_name = result.test_name.lower()
        
        # Add specific insights based on test and status
        if status == 'high':
            causes = explanation_data.get('common_causes_high', [])
            if causes:
                insights.append(f"Common causes: {', '.join(causes[:2])}")
        elif status == 'low':
            causes = explanation_data.get('common_causes_low', [])
            if causes:
                insights.append(f"Common causes: {', '.join(causes[:2])}")
        
        # Add general recommendations
        recommendations = self._get_general_recommendations(test_name, status)
        if recommendations:
            insights.extend(recommendations)
        
        return '. '.join(insights) + '.' if insights else ''
    
    def _get_general_recommendations(self, test_name: str, status: str) -> List[str]:
        """Get general health recommendations based on test results"""
        recommendations = []
        
        if 'cholesterol' in test_name and status == 'high':
            recommendations.append("Consider heart-healthy diet and exercise")
        elif 'glucose' in test_name and status == 'high':
            recommendations.append("Monitor blood sugar and consider dietary changes")
        elif 'hemoglobin' in test_name and status == 'low':
            recommendations.append("Consider iron-rich foods or supplements")
        elif 'blood pressure' in test_name and status == 'high':
            recommendations.append("Reduce salt intake and increase physical activity")
        
        return recommendations
    
    def _determine_urgency(self, test_name: str, value: float, status: str) -> str:
        """Determine urgency level based on test results"""
        
        # Check specific urgency rules
        if test_name in self.urgency_rules:
            rules = self.urgency_rules[test_name]
            
            # Check high urgency first
            high_rules = rules.get('high_urgency', {})
            if (value < high_rules.get('low', float('-inf')) or 
                value > high_rules.get('high', float('inf'))):
                return 'high'
            
            # Check medium urgency
            medium_rules = rules.get('medium_urgency', {})
            if (value < medium_rules.get('low', float('-inf')) or 
                value > medium_rules.get('high', float('inf'))):
                return 'medium'
        
        # Default urgency based on status
        if status == 'normal':
            return 'low'
        elif status in ['high', 'low']:
            return 'medium'
        else:
            return 'low'
    
    def get_overall_health_summary(self, simplified_results: List[Dict]) -> Dict:
        """Generate an overall health summary from all test results"""
        summary = {
            'total_tests': len(simplified_results),
            'normal_count': 0,
            'abnormal_count': 0,
            'high_urgency_count': 0,
            'medium_urgency_count': 0,
            'key_concerns': [],
            'positive_findings': [],
            'overall_status': 'unknown'
        }
        
        for result in simplified_results:
            if result['status'] == 'normal':
                summary['normal_count'] += 1
                summary['positive_findings'].append(result['test_name'])
            else:
                summary['abnormal_count'] += 1
                
            if result['urgency_level'] == 'high':
                summary['high_urgency_count'] += 1
                summary['key_concerns'].append(result['test_name'])
            elif result['urgency_level'] == 'medium':
                summary['medium_urgency_count'] += 1
        
        # Determine overall status
        if summary['high_urgency_count'] > 0:
            summary['overall_status'] = 'needs_attention'
        elif summary['abnormal_count'] > summary['normal_count']:
            summary['overall_status'] = 'some_concerns'
        else:
            summary['overall_status'] = 'mostly_normal'
        
        return summary
