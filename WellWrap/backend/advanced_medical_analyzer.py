"""
Advanced Medical Analyzer - Lightweight Version
Integration layer between Flask backend and medical analysis
"""

import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)

class MedicalTestResult:
    """Medical test result data structure"""
    def __init__(self, test_name, value, unit=None, normal_range=None, status=None, severity_score=0.0):
        self.test_name = test_name
        self.value = value
        self.unit = unit
        self.normal_range = normal_range
        self.status = status
        self.severity_score = severity_score

@dataclass
class DiseaseRisk:
    """Disease risk assessment result"""
    disease_name: str
    risk_level: str
    confidence: float
    contributing_factors: List[str]
    description: str
    recommendations: List[str]

class AdvancedMedicalAnalyzer:
    """
    Lightweight Medical Analyzer for Flask Backend
    Provides basic medical report analysis without heavy ML dependencies
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized lightweight medical analyzer")
    
    def extract_text_from_pdf(self, file) -> str:
        """Extract text from uploaded PDF file using PyPDF2"""
        try:
            import PyPDF2
            from io import BytesIO
            
            # Read file content
            file_content = file.read()
            file.seek(0)  # Reset file pointer
            
            # Extract text using PyPDF2
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
        except ImportError:
            self.logger.error("PyPDF2 not available. Please install: pip install PyPDF2")
            return "PDF text extraction requires PyPDF2. Please install it."
        except Exception as e:
            self.logger.error(f"PDF extraction failed: {e}")
            return f"Error extracting PDF: {str(e)}"
    
    def extract_medical_data(self, text: str) -> List[MedicalTestResult]:
        """Extract medical test results from text using pattern matching"""
        results = []
        
        # Common test patterns (case insensitive)
        patterns = {
            'Hemoglobin': r'(?:hemoglobin|hgb|hb)[\s:]*(\d+\.?\d*)',
            'Hematocrit': r'(?:hematocrit|hct)[\s:]*(\d+\.?\d*)',
            'White Blood Cells': r'(?:wbc|white blood cell)[\s:]*(\d+\.?\d*)',
            'Platelets': r'(?:platelets|plt)[\s:]*(\d+\.?\d*)',
            'Glucose': r'(?:glucose|blood sugar)[\s:]*(\d+\.?\d*)',
            'Total Cholesterol': r'(?:total cholesterol|cholesterol)[\s:]*(\d+\.?\d*)',
            'LDL Cholesterol': r'(?:ldl cholesterol|ldl)[\s:]*(\d+\.?\d*)',
            'HDL Cholesterol': r'(?:hdl cholesterol|hdl)[\s:]*(\d+\.?\d*)',
            'Triglycerides': r'(?:triglycerides|trig)[\s:]*(\d+\.?\d*)',
            'Creatinine': r'(?:creatinine|creat)[\s:]*(\d+\.?\d*)',
        }
        
        text_lower = text.lower()
        
        for test_name, pattern in patterns.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                try:
                    value = float(matches[0])
                    status = self._determine_status(test_name, value)
                    severity = self._calculate_severity(test_name, value, status)
                    
                    results.append(MedicalTestResult(
                        test_name=test_name,
                        value=value,
                        unit=self._get_unit(test_name),
                        normal_range=self._get_normal_range(test_name),
                        status=status,
                        severity_score=severity
                    ))
                except ValueError:
                    continue
        
        return results
    
    def _determine_status(self, test_name: str, value: float) -> str:
        """Determine if test result is normal, high, or low"""
        ranges = {
            'Hemoglobin': (12.0, 16.0),
            'Hematocrit': (36.0, 46.0),
            'White Blood Cells': (4.0, 11.0),
            'Platelets': (150.0, 450.0),
            'Glucose': (70.0, 100.0),
            'Total Cholesterol': (0.0, 200.0),
            'LDL Cholesterol': (0.0, 100.0),
            'HDL Cholesterol': (40.0, 999.0),  # Higher is better
            'Triglycerides': (0.0, 150.0),
            'Creatinine': (0.6, 1.3),
        }
        
        if test_name in ranges:
            low, high = ranges[test_name]
            if value < low:
                return 'low'
            elif value > high:
                return 'high'
        
        return 'normal'
    
    def _calculate_severity(self, test_name: str, value: float, status: str) -> float:
        """Calculate severity score (0.0 to 1.0)"""
        if status == 'normal':
            return 0.0
        
        # Define critical thresholds
        critical_ranges = {
            'Hemoglobin': {'low': 8.0, 'high': 18.0},
            'Glucose': {'low': 50.0, 'high': 200.0},
            'Total Cholesterol': {'high': 300.0},
            'Triglycerides': {'high': 500.0},
        }
        
        if test_name in critical_ranges:
            thresholds = critical_ranges[test_name]
            if status == 'low' and 'low' in thresholds and value <= thresholds['low']:
                return 0.8
            elif status == 'high' and 'high' in thresholds and value >= thresholds['high']:
                return 0.8
        
        return 0.5  # Moderate severity for other abnormal results
    
    def _get_unit(self, test_name: str) -> str:
        """Get typical units for tests"""
        units = {
            'Hemoglobin': 'g/dL',
            'Hematocrit': '%',
            'White Blood Cells': 'K/uL',
            'Platelets': 'K/uL',
            'Glucose': 'mg/dL',
            'Total Cholesterol': 'mg/dL',
            'LDL Cholesterol': 'mg/dL',
            'HDL Cholesterol': 'mg/dL',
            'Triglycerides': 'mg/dL',
            'Creatinine': 'mg/dL',
        }
        return units.get(test_name, '')
    
    def _get_normal_range(self, test_name: str) -> str:
        """Get normal ranges for tests"""
        ranges = {
            'Hemoglobin': '12.0-16.0 g/dL',
            'Hematocrit': '36-46%',
            'White Blood Cells': '4.0-11.0 K/uL',
            'Platelets': '150-450 K/uL',
            'Glucose': '70-100 mg/dL',
            'Total Cholesterol': '<200 mg/dL',
            'LDL Cholesterol': '<100 mg/dL',
            'HDL Cholesterol': '>40 mg/dL',
            'Triglycerides': '<150 mg/dL',
            'Creatinine': '0.6-1.3 mg/dL',
        }
        return ranges.get(test_name, 'Varies')
    
    def detect_diseases(self, test_results: List[MedicalTestResult]) -> List[DiseaseRisk]:
        """Detect potential diseases based on test results"""
        diseases = []
        
        # Create a dictionary for easier lookup
        test_dict = {result.test_name.lower(): result for result in test_results}
        
        # Anemia detection
        if 'hemoglobin' in test_dict and test_dict['hemoglobin'].status == 'low':
            severity = "high" if test_dict['hemoglobin'].value < 8.0 else "medium"
            diseases.append(DiseaseRisk(
                disease_name="Iron Deficiency Anemia",
                risk_level=severity,
                confidence=0.8,
                contributing_factors=["Low hemoglobin levels"],
                description="A condition where your blood has fewer red blood cells or less hemoglobin than normal",
                recommendations=[
                    "Consult with your doctor immediately",
                    "Consider iron-rich diet (spinach, red meat, beans)",
                    "Get iron studies and further testing",
                    "Monitor for symptoms like fatigue and weakness"
                ]
            ))
        
        # Cardiovascular risk detection
        high_cholesterol = 'total cholesterol' in test_dict and test_dict['total cholesterol'].status == 'high'
        high_ldl = 'ldl cholesterol' in test_dict and test_dict['ldl cholesterol'].status == 'high'
        low_hdl = 'hdl cholesterol' in test_dict and test_dict['hdl cholesterol'].status == 'low'
        high_triglycerides = 'triglycerides' in test_dict and test_dict['triglycerides'].status == 'high'
        
        if high_cholesterol or high_ldl or low_hdl or high_triglycerides:
            risk_factors = []
            if high_cholesterol: risk_factors.append("High total cholesterol")
            if high_ldl: risk_factors.append("High LDL cholesterol")
            if low_hdl: risk_factors.append("Low HDL cholesterol")
            if high_triglycerides: risk_factors.append("High triglycerides")
            
            risk_level = "high" if len(risk_factors) >= 2 else "medium"
            
            diseases.append(DiseaseRisk(
                disease_name="Cardiovascular Disease Risk",
                risk_level=risk_level,
                confidence=0.9,
                contributing_factors=risk_factors,
                description="Elevated lipid levels increase risk of heart disease and stroke",
                recommendations=[
                    "Follow heart-healthy diet (Mediterranean or DASH diet)",
                    "Increase physical activity (150 min/week moderate exercise)",
                    "Consider statin therapy (consult doctor)",
                    "Schedule cardiology consultation",
                    "Monitor blood pressure regularly"
                ]
            ))
        
        # Diabetes risk detection
        if 'glucose' in test_dict and test_dict['glucose'].status == 'high':
            glucose_value = test_dict['glucose'].value
            if glucose_value >= 126:
                risk_level = "high"
                disease_name = "Diabetes"
            elif glucose_value >= 100:
                risk_level = "medium"
                disease_name = "Prediabetes"
            else:
                risk_level = "low"
                disease_name = "Glucose Intolerance"
            
            diseases.append(DiseaseRisk(
                disease_name=disease_name,
                risk_level=risk_level,
                confidence=0.85,
                contributing_factors=["Elevated fasting glucose"],
                description="High blood sugar levels that may indicate diabetes or prediabetes",
                recommendations=[
                    "Monitor blood sugar regularly",
                    "Follow diabetic-friendly diet",
                    "Maintain healthy weight",
                    "Regular exercise",
                    "Consult endocrinologist"
                ]
            ))
        
        # Kidney function concerns
        if 'creatinine' in test_dict and test_dict['creatinine'].status == 'high':
            diseases.append(DiseaseRisk(
                disease_name="Kidney Function Impairment",
                risk_level="medium",
                confidence=0.7,
                contributing_factors=["Elevated creatinine"],
                description="Kidney function may be impaired based on elevated waste product levels",
                recommendations=[
                    "Stay well hydrated",
                    "Monitor kidney function regularly",
                    "Limit protein intake if advised",
                    "Consult nephrologist if levels remain high"
                ]
            ))
        
        return diseases
    
    def generate_health_summary(self, test_results: List[MedicalTestResult], 
                               disease_risks: List[DiseaseRisk]) -> Dict[str, Any]:
        """Generate comprehensive health summary"""
        total_tests = len(test_results)
        normal_tests = sum(1 for t in test_results if t.status == 'normal')
        abnormal_tests = total_tests - normal_tests
        
        # Calculate health score (0-100)
        health_score = 100
        
        # Deduct points for abnormal results
        if total_tests > 0:
            abnormal_ratio = abnormal_tests / total_tests
            health_score -= abnormal_ratio * 30
        
        # Deduct points for disease risks
        for risk in disease_risks:
            if risk.risk_level == 'high':
                health_score -= 25
            elif risk.risk_level == 'medium':
                health_score -= 15
            else:
                health_score -= 5
        
        health_score = max(0, min(100, int(health_score)))
        
        # Determine overall status
        if health_score >= 80:
            overall_status = "Good"
            status_color = "success"
        elif health_score >= 60:
            overall_status = "Fair"
            status_color = "warning"
        else:
            overall_status = "Needs Attention"
            status_color = "danger"
        
        return {
            'health_score': health_score,
            'overall_status': overall_status,
            'status_color': status_color,
            'total_tests': total_tests,
            'normal_tests': normal_tests,
            'abnormal_tests': abnormal_tests,
            'disease_risks_count': len(disease_risks),
            'high_risk_diseases': [d.disease_name for d in disease_risks if d.risk_level == 'high'],
            'medium_risk_diseases': [d.disease_name for d in disease_risks if d.risk_level == 'medium'],
            'recommendations': self._generate_general_recommendations(test_results, disease_risks),
            'summary': f"Based on {total_tests} test results, your health score is {health_score}/100 ({overall_status})",
            'next_steps': self._generate_next_steps(disease_risks)
        }
    
    def _generate_general_recommendations(self, test_results: List[MedicalTestResult], 
                                        disease_risks: List[DiseaseRisk]) -> List[str]:
        """Generate general health recommendations"""
        recommendations = set()  # Use set to avoid duplicates
        
        # Add recommendations from disease risks
        for risk in disease_risks:
            recommendations.update(risk.recommendations[:2])  # Add first 2 recommendations
        
        # Add general health recommendations
        recommendations.update([
            "Maintain a balanced diet rich in fruits and vegetables",
            "Stay physically active with regular exercise",
            "Get adequate sleep (7-9 hours per night)",
            "Schedule regular check-ups with your healthcare provider"
        ])
        
        return list(recommendations)[:8]  # Limit to 8 recommendations
    
    def _generate_next_steps(self, disease_risks: List[DiseaseRisk]) -> List[str]:
        """Generate immediate next steps based on risk assessment"""
        next_steps = []
        
        high_risks = [r for r in disease_risks if r.risk_level == 'high']
        medium_risks = [r for r in disease_risks if r.risk_level == 'medium']
        
        if high_risks:
            next_steps.append("🚨 Schedule immediate appointment with your doctor")
            next_steps.append("📋 Discuss high-risk findings and treatment options")
        
        if medium_risks:
            next_steps.append("📅 Schedule follow-up appointment within 2-4 weeks")
            next_steps.append("📊 Consider additional testing as recommended")
        
        if not high_risks and not medium_risks:
            next_steps.append("✅ Continue current health maintenance routine")
            next_steps.append("📅 Schedule routine check-up in 6-12 months")
        
        next_steps.append("📱 Monitor symptoms and contact doctor if concerns arise")
        
        return next_steps