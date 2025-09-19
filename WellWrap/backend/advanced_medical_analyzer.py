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
        """Extract text from uploaded PDF or image files.

        - PDFs: use PyPDF2 text extraction
        - Images (PNG/JPG): use Tesseract OCR via pytesseract
        """
        from io import BytesIO
        
        # Read file bytes and reset stream for downstream consumers
        file_bytes = file.read()
        file.seek(0)

        if not file_bytes:
            return ""

        # Simple magic header checks
        is_pdf = file_bytes[:5] == b"%PDF-"
        is_png = file_bytes[:8] == b"\x89PNG\r\n\x1a\n"
        is_jpg = file_bytes[:3] == b"\xff\xd8\xff"

        if is_pdf:
            # PDF path
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
                text_chunks = []
                for page in pdf_reader.pages:
                    try:
                        page_text = page.extract_text() or ""
                    except Exception:
                        page_text = ""
                    text_chunks.append(page_text)
                text = "\n".join([t for t in text_chunks if t])
                return text.strip()
            except ImportError:
                self.logger.error("PyPDF2 not available. Please install: pip install PyPDF2")
                return "PDF text extraction requires PyPDF2. Please install it."
            except Exception as e:
                self.logger.error(f"PDF extraction failed: {e}")
                # Fall through to try OCR on rendered image of first page (not implemented in lightweight version)
                return f"Error extracting PDF: {str(e)}"

        if is_png or is_jpg:
            # Image OCR path
            try:
                from PIL import Image
                import pytesseract
            except ImportError as e:
                missing = 'Pillow' if 'PIL' in str(e) else 'pytesseract'
                self.logger.error(f"{missing} not available. Install with: pip install Pillow pytesseract")
                return "Image text extraction requires Pillow and pytesseract. Please install them."

            try:
                image = Image.open(BytesIO(file_bytes))
                # Basic preprocessing can help OCR; keep minimal for speed
                if image.mode not in ("L", "RGB"):
                    image = image.convert("RGB")
                ocr_text = pytesseract.image_to_string(image)
                return (ocr_text or "").strip()
            except pytesseract.pytesseract.TesseractNotFoundError:
                msg = (
                    "Tesseract OCR engine not found. Install Tesseract and ensure it is on PATH. "
                    "Windows: download from UB Mannheim build and add tesseract.exe to PATH."
                )
                self.logger.error(msg)
                return msg
            except Exception as e:
                self.logger.error(f"Image OCR failed: {e}")
                return f"Error extracting image text: {str(e)}"

        # Unknown format; try PDF first then OCR as a fallback
        try:
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
            text = "\n".join([(p.extract_text() or "") for p in pdf_reader.pages])
            return text.strip()
        except Exception:
            try:
                from PIL import Image
                import pytesseract
                img = Image.open(BytesIO(file_bytes))
                return (pytesseract.image_to_string(img) or "").strip()
            except Exception as e:
                self.logger.error(f"Generic extraction failed: {e}")
                return "Unsupported file format or unreadable content."
    
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
    
    def _generate_action_plan(self, test_results: List[MedicalTestResult], 
                              disease_risks: List[DiseaseRisk]) -> List[Dict[str, Any]]:
        """Translate findings into concrete actions grouped by priority."""
        actions: List[Dict[str, Any]] = []

        def add_action(priority: str, title: str, details: str, when: str, who: str, links: Optional[List[Dict[str, str]]] = None):
            actions.append({
                'priority': priority,  # immediate, soon, routine
                'title': title,
                'details': details,
                'when': when,
                'who': who,
                'links': links or []
            })

        # Map abnormal tests to actions
        for tr in test_results:
            if tr.status == 'high':
                if tr.test_name in ['Total Cholesterol', 'LDL Cholesterol', 'Triglycerides']:
                    add_action(
                        priority='soon',
                        title='Improve lipid profile',
                        details='Adopt heart-healthy diet, increase activity; discuss statin therapy if applicable.',
                        when='Within 2-4 weeks',
                        who='Primary care / cardiology'
                    )
                if tr.test_name == 'Glucose':
                    add_action(
                        priority='soon',
                        title='Address elevated blood sugar',
                        details='Reduce refined carbs, monitor fasting glucose; evaluate for diabetes if persistent.',
                        when='Within 1-2 weeks',
                        who='Primary care / endocrinology'
                    )
                if tr.test_name == 'Creatinine':
                    add_action(
                        priority='soon',
                        title='Assess kidney function',
                        details='Repeat kidney panel, review hydration, medications; consider nephrology referral.',
                        when='Within 2-4 weeks',
                        who='Primary care / nephrology'
                    )
            elif tr.status == 'low':
                if tr.test_name == 'Hemoglobin':
                    add_action(
                        priority='soon',
                        title='Evaluate anemia',
                        details='Order iron studies; consider iron-rich diet and supplementation per clinician.',
                        when='Within 1-2 weeks',
                        who='Primary care / hematology'
                    )

        # Map disease risks to actions
        for risk in disease_risks:
            if risk.disease_name == 'Cardiovascular Disease Risk':
                add_action(
                    priority='soon' if risk.risk_level in ['high', 'medium'] else 'routine',
                    title='Reduce cardiovascular risk',
                    details='Manage lipids, BP, weight; stop smoking; consider statin; 150 min/week exercise.',
                    when='Start this week',
                    who='Primary care / cardiology'
                )
            if risk.disease_name in ['Diabetes', 'Prediabetes']:
                add_action(
                    priority='immediate' if risk.disease_name == 'Diabetes' else 'soon',
                    title='Glycemic management plan',
                    details='Dietary changes, glucose monitoring; consider medication and educator referral.',
                    when='Within 1 week' if risk.disease_name == 'Diabetes' else 'Within 2-4 weeks',
                    who='Primary care / endocrinology'
                )
            if 'Kidney' in risk.disease_name:
                add_action(
                    priority='soon',
                    title='Kidney protection steps',
                    details='Optimize hydration, review nephrotoxic meds; consider eGFR/urine ACR testing.',
                    when='Within 2-4 weeks',
                    who='Primary care / nephrology'
                )

        # De-duplicate by title keeping highest priority
        priority_rank = {'immediate': 0, 'soon': 1, 'routine': 2}
        dedup = {}
        for a in actions:
            t = a['title']
            if t not in dedup or priority_rank[a['priority']] < priority_rank[dedup[t]['priority']]:
                dedup[t] = a
        actions = sorted(dedup.values(), key=lambda x: priority_rank.get(x['priority'], 3))
        return actions

    def generate_health_summary(self, test_results: List[MedicalTestResult], 
                               disease_risks: List[DiseaseRisk]) -> Dict[str, Any]:
        """Generate comprehensive health summary with consequences if ignored"""
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
        
        high_risk_items = [d for d in disease_risks if d.risk_level == 'high']
        medium_risk_items = [d for d in disease_risks if d.risk_level == 'medium']
        
        # Build narrative summary
        key_findings = []
        for tr in test_results:
            if tr.status != 'normal':
                key_findings.append(f"{tr.test_name}: {tr.value} {tr.unit or ''} (expected {tr.normal_range}, {tr.status})")
        for dr in disease_risks:
            key_findings.append(f"Risk: {dr.disease_name} ({dr.risk_level})")
        
        consequences = []
        for dr in high_risk_items:
            if 'cardiovascular' in dr.disease_name.lower():
                consequences.append("Increased likelihood of heart attack or stroke if untreated")
            if 'diabetes' in dr.disease_name.lower():
                consequences.append("Risk of nerve damage, kidney disease, and vision loss if uncontrolled")
            if 'anemia' in dr.disease_name.lower():
                consequences.append("Severe fatigue, shortness of breath, and potential cardiac strain if untreated")
            if 'kidney' in dr.disease_name.lower():
                consequences.append("Progression to chronic kidney disease if not addressed")
        if medium_risk_items and not consequences:
            consequences.append("Condition may worsen over time and increase long-term complications")
        if not consequences and abnormal_tests > 0:
            consequences.append("Abnormal results may progress without follow-up care")

        action_plan = self._generate_action_plan(test_results, disease_risks)
        
        narrative = {
            'summary_text': (
                f"We analyzed {total_tests} tests. Your health score is {health_score}/100 "
                f"({overall_status}). "
                + ("Key findings include: " + "; ".join(key_findings) + ". " if key_findings else "")
                + ("If recommended actions are not taken, potential consequences include: "
                   + "; ".join(consequences) + "." if consequences else "")
            ).strip()
        }
        
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
            'next_steps': self._generate_next_steps(disease_risks),
            'narrative': narrative,
            'key_findings': key_findings,
            'consequences_if_ignored': consequences,
            'action_plan': action_plan,
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