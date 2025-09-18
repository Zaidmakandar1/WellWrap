#!/usr/bin/env python3
"""
Database Initialization Script
Creates all necessary tables and sets up demo data for the Medical Report Simplifier
"""

import os
import sys
from datetime import datetime, date
import json

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def init_database():
    """Initialize the database with all tables and demo data"""
    
    print("🏥 Medical Report Simplifier - Database Initialization")
    print("=" * 60)
    
    try:
        # Import Flask app and models
        from run_app import app, db, User, MedicalReport, bcrypt
        
        with app.app_context():
            print("📂 Creating database tables...")
            
            # Drop all tables first (for clean setup)
            db.drop_all()
            print("   ✅ Dropped existing tables")
            
            # Create all tables
            db.create_all()
            print("   ✅ Created all tables")
            
            # Verify tables were created
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            print(f"   📋 Tables created: {', '.join(tables)}")
            
            # Create demo users
            print("\n👥 Creating demo users...")
            
            # Admin user
            admin_user = User(
                username='admin',
                email='admin@healthcare.com',
                password_hash=bcrypt.generate_password_hash('admin123').decode('utf-8'),
                first_name='System',
                last_name='Administrator',
                date_of_birth=date(1980, 1, 1),
                phone='555-0001',
                gender='Other',
                role='admin'
            )
            db.session.add(admin_user)
            print("   ✅ Created admin user (admin/admin123)")
            
            # Demo patient
            demo_patient = User(
                username='demo_patient',
                email='patient@demo.com',
                password_hash=bcrypt.generate_password_hash('demo123').decode('utf-8'),
                first_name='John',
                last_name='Doe',
                date_of_birth=date(1985, 5, 15),
                phone='555-0002',
                gender='Male',
                role='patient'
            )
            db.session.add(demo_patient)
            print("   ✅ Created demo patient (demo_patient/demo123)")
            
            # Test doctor
            test_doctor = User(
                username='dr_smith',
                email='doctor@healthcare.com',
                password_hash=bcrypt.generate_password_hash('doctor123').decode('utf-8'),
                first_name='Sarah',
                last_name='Smith',
                date_of_birth=date(1975, 8, 20),
                phone='555-0003',
                gender='Female',
                role='doctor'
            )
            db.session.add(test_doctor)
            print("   ✅ Created test doctor (dr_smith/doctor123)")
            
            # Commit users first
            db.session.commit()
            print("   💾 Saved users to database")
            
            # Create sample medical reports
            print("\n📋 Creating sample medical reports...")
            
            # Sample CBC report
            cbc_analysis = {
                'extracted_values': [
                    {'test_name': 'Hemoglobin', 'value': 8.5, 'unit': 'g/dL', 'status': 'low', 'normal_range': '12.0-15.5'},
                    {'test_name': 'Hematocrit', 'value': 25.2, 'unit': '%', 'status': 'low', 'normal_range': '36.0-44.0'},
                    {'test_name': 'White Blood Cells', 'value': 12.8, 'unit': 'K/uL', 'status': 'high', 'normal_range': '4.0-11.0'},
                    {'test_name': 'Platelets', 'value': 180, 'unit': 'K/uL', 'status': 'normal', 'normal_range': '150-450'}
                ],
                'health_score': 45,
                'summary': 'Results suggest iron deficiency anemia with possible infection or inflammation.',
                'recommendations': [
                    'Consult with primary care physician immediately',
                    'Iron studies recommended',
                    'Monitor for symptoms of fatigue and weakness',
                    'Follow-up blood work in 2-4 weeks'
                ],
                'risk_factors': ['Iron deficiency anemia', 'Possible infection']
            }
            
            cbc_report = MedicalReport(
                user_id=demo_patient.id,
                filename='sample_cbc_report.txt',
                original_filename='CBC_Report_John_Doe.txt',
                file_path='data/samples/sample_cbc_report.txt',
                extracted_text="""COMPLETE BLOOD COUNT (CBC) REPORT
=====================================

Patient: John Doe
Date: September 18, 2025
Lab: City Medical Laboratory

TEST RESULTS:
-------------

Hemoglobin: 8.5 g/dL (Normal: 12.0-15.5 g/dL) *LOW*
Hematocrit: 25.2% (Normal: 36.0-44.0%) *LOW*
White Blood Cells: 12.8 K/uL (Normal: 4.0-11.0 K/uL) *HIGH*
Platelets: 180 K/uL (Normal: 150-450 K/uL) Normal

Red Blood Cell Count: 3.2 million/uL (Normal: 4.5-5.5 million/uL) *LOW*
Mean Corpuscular Volume: 68 fL (Normal: 80-100 fL) *LOW*
Mean Corpuscular Hemoglobin: 22 pg (Normal: 27-33 pg) *LOW*

CLINICAL NOTES:
---------------
Results suggest iron deficiency anemia with possible infection or inflammation.
Recommend iron studies and follow-up with primary care physician.""",
                analysis_results=json.dumps(cbc_analysis),
                health_score=45,
                upload_date=datetime.now(),
                report_date=date(2025, 9, 18),
                report_type='Blood Test',
                status='analyzed'
            )
            db.session.add(cbc_report)
            print("   ✅ Created sample CBC report")
            
            # Sample Lipid Profile report
            lipid_analysis = {
                'extracted_values': [
                    {'test_name': 'Total Cholesterol', 'value': 245, 'unit': 'mg/dL', 'status': 'high', 'normal_range': '<200'},
                    {'test_name': 'LDL Cholesterol', 'value': 165, 'unit': 'mg/dL', 'status': 'high', 'normal_range': '<100'},
                    {'test_name': 'HDL Cholesterol', 'value': 35, 'unit': 'mg/dL', 'status': 'low', 'normal_range': '>40'},
                    {'test_name': 'Triglycerides', 'value': 280, 'unit': 'mg/dL', 'status': 'high', 'normal_range': '<150'}
                ],
                'health_score': 35,
                'summary': 'High risk for cardiovascular disease based on multiple abnormal lipid values.',
                'recommendations': [
                    'Lifestyle modifications including diet and exercise',
                    'Consider statin therapy',
                    'Follow-up in 6-8 weeks',
                    'Consultation with cardiologist recommended'
                ],
                'risk_factors': ['Cardiovascular disease risk', 'Hyperlipidemia']
            }
            
            lipid_report = MedicalReport(
                user_id=demo_patient.id,
                filename='sample_lipid_report.txt',
                original_filename='Lipid_Profile_John_Doe.txt',
                file_path='data/samples/sample_lipid_report.txt',
                extracted_text="""LIPID PROFILE REPORT
====================

Patient: Jane Smith
Date: September 18, 2025
Lab: City Medical Laboratory
Fasting Status: 12 hours fasting

LIPID PANEL RESULTS:
-------------------

Total Cholesterol: 245 mg/dL (Desirable: <200 mg/dL) *HIGH*
LDL Cholesterol: 165 mg/dL (Optimal: <100 mg/dL) *HIGH*
HDL Cholesterol: 35 mg/dL (Desirable: >40 mg/dL for men, >50 mg/dL for women) *LOW*
Triglycerides: 280 mg/dL (Normal: <150 mg/dL) *HIGH*

RECOMMENDATIONS:
----------------
- Lifestyle modifications including diet and exercise
- Consider statin therapy
- Follow-up in 6-8 weeks
- Consultation with cardiologist recommended""",
                analysis_results=json.dumps(lipid_analysis),
                health_score=35,
                upload_date=datetime.now(),
                report_date=date(2025, 9, 18),
                report_type='Lipid Panel',
                status='analyzed'
            )
            db.session.add(lipid_report)
            print("   ✅ Created sample Lipid Profile report")
            
            # Commit all changes
            db.session.commit()
            print("   💾 Saved sample reports to database")
            
            # Verify database setup
            print("\n🔍 Verifying database setup...")
            user_count = User.query.count()
            report_count = MedicalReport.query.count()
            
            print(f"   👥 Users created: {user_count}")
            print(f"   📋 Reports created: {report_count}")
            
            # Test database queries
            print("\n🧪 Testing database queries...")
            demo_user = User.query.filter_by(username='demo_patient').first()
            if demo_user:
                user_reports = MedicalReport.query.filter_by(user_id=demo_user.id).count()
                print(f"   ✅ Demo user found with {user_reports} reports")
            else:
                print("   ❌ Demo user not found")
                
            print("\n" + "=" * 60)
            print("✅ Database initialization completed successfully!")
            print("\n📚 Login Credentials:")
            print("   👤 Admin: admin / admin123")
            print("   🏥 Patient: demo_patient / demo123")
            print("   👨‍⚕️ Doctor: dr_smith / doctor123")
            print("\n🌐 Start the application with: python run_app.py")
            
            return True
            
    except Exception as e:
        print(f"\n❌ Database initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def reset_database():
    """Reset the database by dropping and recreating all tables"""
    print("⚠️  Resetting database - all data will be lost!")
    response = input("Are you sure? (y/N): ").strip().lower()
    
    if response == 'y':
        return init_database()
    else:
        print("Database reset cancelled.")
        return False

def check_database():
    """Check database status and show statistics"""
    try:
        from run_app import app, db, User, MedicalReport
        
        with app.app_context():
            print("🔍 Database Status Check")
            print("=" * 30)
            
            # Check if tables exist
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            
            if not tables:
                print("❌ No tables found in database")
                print("💡 Run: python init_database.py --init")
                return False
            
            print(f"📋 Tables: {', '.join(tables)}")
            
            # Check data
            user_count = User.query.count()
            report_count = MedicalReport.query.count()
            
            print(f"👥 Users: {user_count}")
            print(f"📋 Reports: {report_count}")
            
            if user_count == 0:
                print("⚠️  No users found - database may need initialization")
            
            return True
            
    except Exception as e:
        print(f"❌ Database check failed: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Medical Report Simplifier Database Management')
    parser.add_argument('--init', action='store_true', help='Initialize database with demo data')
    parser.add_argument('--reset', action='store_true', help='Reset database (WARNING: deletes all data)')
    parser.add_argument('--check', action='store_true', help='Check database status')
    
    args = parser.parse_args()
    
    if args.init:
        init_database()
    elif args.reset:
        reset_database()
    elif args.check:
        check_database()
    else:
        # Default action - initialize if no tables exist, otherwise check status
        try:
            from run_app import app, db
            with app.app_context():
                from sqlalchemy import inspect
                inspector = inspect(db.engine)
                tables = inspector.get_table_names()
                
                if not tables:
                    print("No database tables found. Initializing...")
                    init_database()
                else:
                    check_database()
        except Exception as e:
            print(f"Error: {e}")
            print("Attempting to initialize database...")
            init_database()