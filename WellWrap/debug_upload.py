#!/usr/bin/env python3
"""
Debug script to help identify upload issues
"""

import sys
import traceback
from pathlib import Path
from io import BytesIO

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def debug_pdf_processing():
    """Debug PDF processing with detailed logging"""
    print("🔍 Debug: PDF Processing Test")
    print("=" * 50)
    
    try:
        from backend.advanced_medical_analyzer import AdvancedMedicalAnalyzer
        
        print("✅ Successfully imported AdvancedMedicalAnalyzer")
        
        # Initialize analyzer
        analyzer = AdvancedMedicalAnalyzer()
        print("✅ Successfully initialized analyzer")
        
        # Test with sample medical text (simulating extracted PDF content)
        sample_texts = [
            # Simple case
            "Hemoglobin: 12.5 g/dL\nGlucose: 95 mg/dL",
            
            # Complex case with multiple tests
            """
            LABORATORY RESULTS
            Complete Blood Count:
            Hemoglobin: 11.2 g/dL (Low)
            Hematocrit: 33.8%
            White Blood Cells: 7.2 K/uL
            Platelets: 285 K/uL
            
            Lipid Profile:
            Total Cholesterol: 245 mg/dL (High)
            LDL: 165 mg/dL (High)
            HDL: 38 mg/dL (Low)
            Triglycerides: 320 mg/dL (High)
            """,
            
            # Edge case - minimal text
            "Test: Normal",
            
            # Edge case - no medical data
            "This is just regular text without medical values",
            
            # Edge case - garbled text (OCR artifacts)
            "Hemoglobin: l2.5 g/dL\nGlucose: O95 mg/dL"
        ]
        
        for i, text in enumerate(sample_texts, 1):
            print(f"\n--- Test Case {i} ---")
            print(f"Input text: {text[:100]}...")
            
            try:
                # Test medical data extraction
                test_results = analyzer.extract_medical_data(text)
                print(f"✅ Extracted {len(test_results)} medical results")
                
                # Test disease detection
                disease_risks = analyzer.detect_diseases(test_results)
                print(f"✅ Detected {len(disease_risks)} disease risks")
                
                # Test health summary
                health_summary = analyzer.generate_health_summary(test_results, disease_risks)
                print(f"✅ Generated health summary: {health_summary.get('health_score', 'N/A')}/100")
                
                # Test JSON serialization
                serializable_data = analyzer.convert_to_json_serializable(test_results, disease_risks)
                print(f"✅ JSON serialization successful")
                
                # Test complete analysis
                complete_analysis = {
                    **health_summary,
                    'test_results': serializable_data['test_results'],
                    'disease_risks': serializable_data['disease_risks']
                }
                
                import json
                json_str = json.dumps(complete_analysis)
                print(f"✅ Complete analysis JSON: {len(json_str)} characters")
                
            except Exception as e:
                print(f"❌ Error in test case {i}: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"❌ Critical error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()
        return False

def debug_file_types():
    """Debug file type detection"""
    print("\n🔍 Debug: File Type Detection")
    print("=" * 50)
    
    try:
        from run_app import determine_file_type
        
        test_files = [
            'report.pdf',
            'scan.jpg',
            'image.png',
            'data.txt',
            'document.docx',
            'unknown.xyz'
        ]
        
        for filename in test_files:
            file_type = determine_file_type(filename)
            print(f"✅ {filename} -> {file_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ File type detection error: {str(e)}")
        traceback.print_exc()
        return False

def debug_common_issues():
    """Debug common issues that might cause upload failures"""
    print("\n🔍 Debug: Common Issues Check")
    print("=" * 50)
    
    issues_found = []
    
    # Check imports
    try:
        import PyPDF2
        print("✅ PyPDF2 available")
    except ImportError:
        issues_found.append("PyPDF2 not available")
        print("❌ PyPDF2 not available")
    
    try:
        import pytesseract
        print("✅ pytesseract available")
    except ImportError:
        issues_found.append("pytesseract not available")
        print("❌ pytesseract not available")
    
    try:
        from PIL import Image
        print("✅ PIL available")
    except ImportError:
        issues_found.append("PIL not available")
        print("❌ PIL not available")
    
    try:
        import cv2
        print("✅ OpenCV available")
    except ImportError:
        issues_found.append("OpenCV not available")
        print("❌ OpenCV not available")
    
    # Check database
    try:
        from run_app import db, MedicalReport
        print("✅ Database models available")
    except Exception as e:
        issues_found.append(f"Database issue: {str(e)}")
        print(f"❌ Database issue: {str(e)}")
    
    if issues_found:
        print(f"\n⚠️  Found {len(issues_found)} potential issues:")
        for issue in issues_found:
            print(f"   - {issue}")
    else:
        print("\n✅ No obvious issues found")
    
    return len(issues_found) == 0

if __name__ == "__main__":
    print("🚀 WellWrap Upload Debug Tool")
    print("=" * 60)
    
    success = True
    
    if not debug_pdf_processing():
        success = False
    
    if not debug_file_types():
        success = False
    
    if not debug_common_issues():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All debug tests passed!")
        print("If you're still getting errors, please share the specific error message.")
    else:
        print("❌ Some issues found. Check the output above for details.")
    
    print("\n💡 To get more specific help:")
    print("   1. Share the exact error message you're seeing")
    print("   2. Mention which PDF is causing the issue")
    print("   3. Check the console output when uploading")