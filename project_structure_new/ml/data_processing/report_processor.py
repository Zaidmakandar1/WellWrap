"""
Report Processing Module
Handles PDF text extraction and medical data parsing from reports.
"""

import PyPDF2
import pdfplumber
import re
import logging
from typing import List, Dict, Optional, Union
import io
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MedicalTestResult:
    """Data class for medical test results"""
    test_name: str
    value: Union[str, float]
    unit: Optional[str] = None
    normal_range: Optional[str] = None
    status: Optional[str] = None  # 'normal', 'high', 'low'

class ReportProcessor:
    """Processes medical reports from various formats"""
    
    def __init__(self, model_handler=None):
        self.medical_patterns = self._initialize_patterns()
        self.model_handler = model_handler  # Optional: for enhanced text processing
    
    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """Initialize regex patterns for common medical tests"""
        return {
            # CBC patterns
            'hemoglobin': [
                r'hemoglobin[:\s]+(\d+\.?\d*)\s*([mg/dLg%]+)?',
                r'hgb[:\s]+(\d+\.?\d*)\s*([mg/dLg%]+)?',
                r'hb[:\s]+(\d+\.?\d*)\s*([mg/dLg%]+)?'
            ],
            'hematocrit': [
                r'hematocrit[:\s]+(\d+\.?\d*)\s*([%]+)?',
                r'hct[:\s]+(\d+\.?\d*)\s*([%]+)?'
            ],
            'white_blood_cells': [
                r'white blood cell[s]?[:\s]+(\d+\.?\d*)\s*([K/uLx10³/µL]+)?',
                r'wbc[:\s]+(\d+\.?\d*)\s*([K/uLx10³/µL]+)?'
            ],
            'platelets': [
                r'platelet[s]?[:\s]+(\d+\.?\d*)\s*([K/uLx10³/µL]+)?',
                r'plt[:\s]+(\d+\.?\d*)\s*([K/uLx10³/µL]+)?'
            ],
            
            # Lipid profile patterns
            'total_cholesterol': [
                r'total cholesterol[:\s]+(\d+\.?\d*)\s*([mg/dL]+)?',
                r'cholesterol, total[:\s]+(\d+\.?\d*)\s*([mg/dL]+)?'
            ],
            'ldl_cholesterol': [
                r'ldl cholesterol[:\s]+(\d+\.?\d*)\s*([mg/dL]+)?',
                r'ldl[:\s]+(\d+\.?\d*)\s*([mg/dL]+)?'
            ],
            'hdl_cholesterol': [
                r'hdl cholesterol[:\s]+(\d+\.?\d*)\s*([mg/dL]+)?',
                r'hdl[:\s]+(\d+\.?\d*)\s*([mg/dL]+)?'
            ],
            'triglycerides': [
                r'triglyceride[s]?[:\s]+(\d+\.?\d*)\s*([mg/dL]+)?'
            ],
            
            # Basic metabolic panel
            'glucose': [
                r'glucose[:\s]+(\d+\.?\d*)\s*([mg/dL]+)?',
                r'blood sugar[:\s]+(\d+\.?\d*)\s*([mg/dL]+)?'
            ],
            'creatinine': [
                r'creatinine[:\s]+(\d+\.?\d*)\s*([mg/dL]+)?'
            ],
            'blood_urea_nitrogen': [
                r'blood urea nitrogen[:\s]+(\d+\.?\d*)\s*([mg/dL]+)?',
                r'bun[:\s]+(\d+\.?\d*)\s*([mg/dL]+)?'
            ],
            
            # Liver function
            'alt': [
                r'alt[:\s]+(\d+\.?\d*)\s*([U/L]+)?',
                r'alanine aminotransferase[:\s]+(\d+\.?\d*)\s*([U/L]+)?'
            ],
            'ast': [
                r'ast[:\s]+(\d+\.?\d*)\s*([U/L]+)?',
                r'aspartate aminotransferase[:\s]+(\d+\.?\d*)\s*([U/L]+)?'
            ],
            
            # Thyroid function
            'tsh': [
                r'tsh[:\s]+(\d+\.?\d*)\s*([mIU/LµIU/mL]+)?',
                r'thyroid stimulating hormone[:\s]+(\d+\.?\d*)\s*([mIU/LµIU/mL]+)?'
            ]
        }
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            # Try with pdfplumber first (better for formatted documents)
            with pdfplumber.open(io.BytesIO(pdf_file.getvalue())) as pdf:
                text_parts = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                
                if text_parts:
                    return '\n'.join(text_parts)
            
            # Fallback to PyPDF2
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.getvalue()))
            text_parts = []
            
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            return '\n'.join(text_parts)
            
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ""
    
    def parse_medical_data(self, text: str) -> List[MedicalTestResult]:
        """Parse medical test results from text with enhanced NLP preprocessing"""
        results = []
        
        # Preprocess text using enhanced NLP tools if available
        if self.model_handler:
            try:
                processed_text = self.model_handler.preprocess_medical_text(text)
                logger.info("Applied enhanced NLP preprocessing")
            except Exception as e:
                logger.warning(f"Enhanced preprocessing failed, using original text: {e}")
                processed_text = text
        else:
            processed_text = text
        
        text_lower = processed_text.lower()
        
        for test_name, patterns in self.medical_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                
                for match in matches:
                    try:
                        value = float(match.group(1))
                        unit = match.group(2) if match.lastindex > 1 else None
                        
                        # Extract normal range if present in nearby text
                        normal_range = self._extract_normal_range(text, match.start(), match.end())
                        
                        # Determine status
                        status = self._determine_status(test_name, value, normal_range)
                        
                        result = MedicalTestResult(
                            test_name=test_name.replace('_', ' ').title(),
                            value=value,
                            unit=unit,
                            normal_range=normal_range,
                            status=status
                        )
                        
                        results.append(result)
                        break  # Found a match for this test, move to next test
                        
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error parsing {test_name}: {e}")
                        continue
        
        # Remove duplicates (keep first occurrence)
        seen_tests = set()
        unique_results = []
        for result in results:
            if result.test_name not in seen_tests:
                unique_results.append(result)
                seen_tests.add(result.test_name)
        
        return unique_results
    
    def _extract_normal_range(self, text: str, start_pos: int, end_pos: int) -> Optional[str]:
        """Extract normal range from text near the test result"""
        # Look for normal range patterns in the vicinity of the match
        context_start = max(0, start_pos - 100)
        context_end = min(len(text), end_pos + 100)
        context = text[context_start:context_end]
        
        # Common patterns for normal ranges
        range_patterns = [
            r'normal[:\s]*([\d.-]+\s*[-–]\s*[\d.-]+[^\n]*)',
            r'reference[:\s]*([\d.-]+\s*[-–]\s*[\d.-]+[^\n]*)',
            r'\(([\d.-]+\s*[-–]\s*[\d.-]+[^)]*)',
            r'range[:\s]*([\d.-]+\s*[-–]\s*[\d.-]+[^\n]*)',
        ]
        
        for pattern in range_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _determine_status(self, test_name: str, value: float, normal_range: Optional[str]) -> Optional[str]:
        """Determine if test result is normal, high, or low"""
        if not normal_range:
            return self._get_default_status(test_name, value)
        
        # Extract numeric range from normal_range string
        range_match = re.search(r'(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)', normal_range)
        if range_match:
            try:
                min_val = float(range_match.group(1))
                max_val = float(range_match.group(2))
                
                if value < min_val:
                    return 'low'
                elif value > max_val:
                    return 'high'
                else:
                    return 'normal'
            except ValueError:
                pass
        
        return self._get_default_status(test_name, value)
    
    def _get_default_status(self, test_name: str, value: float) -> Optional[str]:
        """Get default status based on common normal ranges"""
        # Default ranges for common tests (approximate values)
        default_ranges = {
            'hemoglobin': (12.0, 16.0),
            'hematocrit': (36.0, 44.0),
            'white_blood_cells': (4.0, 11.0),
            'platelets': (150.0, 450.0),
            'total_cholesterol': (0, 200.0),
            'ldl_cholesterol': (0, 100.0),
            'hdl_cholesterol': (40.0, 100.0),
            'triglycerides': (0, 150.0),
            'glucose': (70.0, 100.0),
            'creatinine': (0.6, 1.3),
            'blood_urea_nitrogen': (7.0, 20.0),
            'alt': (7.0, 56.0),
            'ast': (10.0, 40.0),
            'tsh': (0.4, 4.0)
        }
        
        if test_name.lower().replace(' ', '_') in default_ranges:
            min_val, max_val = default_ranges[test_name.lower().replace(' ', '_')]
            
            if value < min_val:
                return 'low'
            elif value > max_val:
                return 'high'
            else:
                return 'normal'
        
        return None
    
    def get_test_summary(self, results: List[MedicalTestResult]) -> Dict[str, int]:
        """Get summary of test results by status"""
        summary = {'normal': 0, 'high': 0, 'low': 0, 'unknown': 0}
        
        for result in results:
            status = result.status or 'unknown'
            summary[status] += 1
        
        return summary
