#!/usr/bin/env python3
"""
Medical Report Simplifier - Comprehensive Healthcare Application
Features: User Auth, Dashboard, File Upload, Patient History, Medical Analysis
"""

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import os
import json
import uuid
from pathlib import Path
import sqlite3

# Add: safe JSON encoding utilities
try:
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None  # Fallback if numpy not installed

# Use the real analyzer implementation
from advanced_medical_analyzer import AdvancedMedicalAnalyzer


def _safe_default_serializer(obj):
    """Serialize objects not natively supported by json.
    Handles numpy types, sets, bytes, datetime, and falls back to str.
    """
    # numpy types
    if _np is not None:
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.bool_,)):
            return bool(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    # Python sets
    if isinstance(obj, set):
        return list(obj)
    # bytes
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8")
        except Exception:
            return obj.hex()
    # datetime/date
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass
    # dataclasses
    try:
        from dataclasses import asdict, is_dataclass
        if is_dataclass(obj):
            return asdict(obj)
    except Exception:
        pass
    # Fallback string conversion
    return str(obj)


def safe_json_dumps(data, **kwargs):
    """json.dumps with a default serializer that handles common non-JSON types."""
    if "default" not in kwargs:
        kwargs["default"] = _safe_default_serializer
    # Ensure ASCII disabled for symbols/emojis used in UI
    kwargs.setdefault("ensure_ascii", False)
    return json.dumps(data, **kwargs)

# Import our medical analyzer from the new ML structure
# from ml.data_processing.report_processor import ReportProcessor
# from ml.data_processing.simplifier import MedicalReportSimplifier

# Temporary placeholder until ML services are properly integrated
# class AdvancedMedicalAnalyzer:
#     def extract_text_from_pdf(self, file):
#         return "Sample extracted text - ML integration pending"
    
#     def extract_medical_data(self, text):
#         return []
    
#     def detect_diseases(self, test_results):
#         return []
    
#     def generate_health_summary(self, test_results, disease_risks):
#         return {"health_score": 75, "summary": "Analysis pending ML integration"}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///healthcare.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# Initialize medical analyzer
medical_analyzer = AdvancedMedicalAnalyzer()

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    date_of_birth = db.Column(db.Date)
    phone = db.Column(db.String(20))
    gender = db.Column(db.String(10))
    role = db.Column(db.String(20), default='patient')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    medical_reports = db.relationship('MedicalReport', backref='patient', lazy=True, cascade='all, delete-orphan')
    appointments = db.relationship('Appointment', backref='patient', lazy=True, cascade='all, delete-orphan')
    health_metrics = db.relationship('HealthMetric', backref='patient', lazy=True, cascade='all, delete-orphan')

class MedicalReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    extracted_text = db.Column(db.Text)
    analysis_results = db.Column(db.Text)  # JSON string
    test_results = db.Column(db.Text)  # JSON string
    disease_risks = db.Column(db.Text)  # JSON string
    health_score = db.Column(db.Integer)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    report_date = db.Column(db.Date)
    report_type = db.Column(db.String(50))
    status = db.Column(db.String(20), default='pending')  # pending, analyzed, reviewed

class TestResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(db.Integer, db.ForeignKey('medical_report.id'), nullable=False)
    test_name = db.Column(db.String(100), nullable=False)
    value = db.Column(db.Float, nullable=False)
    unit = db.Column(db.String(20))
    normal_range = db.Column(db.String(50))
    status = db.Column(db.String(20))  # normal, high, low, critical
    severity_score = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    doctor_name = db.Column(db.String(100), nullable=False)
    appointment_date = db.Column(db.DateTime, nullable=False)
    appointment_type = db.Column(db.String(50))
    status = db.Column(db.String(20), default='scheduled')  # scheduled, completed, cancelled
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class HealthMetric(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    metric_type = db.Column(db.String(50), nullable=False)  # blood_pressure, weight, glucose, etc.
    value = db.Column(db.String(100), nullable=False)
    unit = db.Column(db.String(20))
    recorded_date = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text)

class Medication(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    medication_name = db.Column(db.String(100), nullable=False)
    dosage = db.Column(db.String(50))
    frequency = db.Column(db.String(50))
    start_date = db.Column(db.Date)
    end_date = db.Column(db.Date)
    prescribing_doctor = db.Column(db.String(100))
    is_active = db.Column(db.Boolean, default=True)
    notes = db.Column(db.Text)

# Routes

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        date_of_birth = datetime.strptime(request.form['date_of_birth'], '%Y-%m-%d').date()
        phone = request.form.get('phone', '')
        gender = request.form.get('gender', '')
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose a different one.', 'danger')
            return render_template('auth/register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered. Please use a different email.', 'danger')
            return render_template('auth/register.html')
        
        # Create new user
        password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            first_name=first_name,
            last_name=last_name,
            date_of_birth=date_of_birth,
            phone=phone,
            gender=gender
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('auth/register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and bcrypt.check_password_hash(user.password_hash, password):
            if user.is_active:
                login_user(user, remember=True)
                user.last_login = datetime.utcnow()
                db.session.commit()
                
                next_page = request.args.get('next')
                flash(f'Welcome back, {user.first_name}!', 'success')
                return redirect(next_page) if next_page else redirect(url_for('dashboard'))
            else:
                flash('Your account has been deactivated. Please contact support.', 'danger')
        else:
            flash('Invalid username or password.', 'danger')
    
    return render_template('auth/login.html')

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Patient dashboard"""
    # Get recent medical reports
    recent_reports = MedicalReport.query.filter_by(user_id=current_user.id)\
                                      .order_by(MedicalReport.upload_date.desc()).limit(5).all()
    
    # Get upcoming appointments
    upcoming_appointments = Appointment.query.filter_by(user_id=current_user.id)\
                                           .filter(Appointment.appointment_date > datetime.utcnow())\
                                           .order_by(Appointment.appointment_date).limit(3).all()
    
    # Get recent health metrics
    recent_metrics = HealthMetric.query.filter_by(user_id=current_user.id)\
                                     .order_by(HealthMetric.recorded_date.desc()).limit(10).all()
    
    # Calculate health statistics
    total_reports = MedicalReport.query.filter_by(user_id=current_user.id).count()
    avg_health_score = db.session.query(db.func.avg(MedicalReport.health_score))\
                                .filter(MedicalReport.user_id == current_user.id)\
                                .filter(MedicalReport.health_score.isnot(None)).scalar()
    
    active_medications = Medication.query.filter_by(user_id=current_user.id, is_active=True).count()
    
    return render_template('dashboard/index.html',
                         recent_reports=recent_reports,
                         upcoming_appointments=upcoming_appointments,
                         recent_metrics=recent_metrics,
                         total_reports=total_reports,
                         avg_health_score=int(avg_health_score) if avg_health_score else 0,
                         active_medications=active_medications)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_report():
    """Upload medical reports"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected.', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected.', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Extract text from PDF
            try:
                extracted_text = medical_analyzer.extract_text_from_pdf(file)
                if not extracted_text:
                    flash('Could not extract text from the uploaded file.', 'danger')
                    return redirect(request.url)
                
                # Analyze the medical report
                test_results = medical_analyzer.extract_medical_data(extracted_text)
                disease_risks = medical_analyzer.detect_diseases(test_results)
                health_summary = medical_analyzer.generate_health_summary(test_results, disease_risks)
                
                # Create medical report record
                report = MedicalReport(
                    user_id=current_user.id,
                    filename=unique_filename,
                    original_filename=filename,
                    file_path=file_path,
                    extracted_text=extracted_text,
                    analysis_results=safe_json_dumps(health_summary),
                    test_results=safe_json_dumps([{
                        'test_name': r.test_name,
                        'value': r.value,
                        'unit': r.unit,
                        'normal_range': r.normal_range,
                        'status': r.status,
                        'severity_score': r.severity_score
                    } for r in test_results]),
                    disease_risks=safe_json_dumps([{
                        'disease_name': d.disease_name,
                        'risk_level': d.risk_level,
                        'confidence': d.confidence,
                        'contributing_factors': d.contributing_factors,
                        'description': d.description,
                        'recommendations': d.recommendations
                    } for d in disease_risks]),
                    health_score=health_summary.get('health_score', 0),
                    report_date=datetime.now().date(),
                    report_type=determine_report_type(filename),
                    status='analyzed'
                )
                
                db.session.add(report)
                db.session.commit()
                
                # Save individual test results
                for test_result in test_results:
                    test_record = TestResult(
                        report_id=report.id,
                        test_name=test_result.test_name,
                        value=test_result.value,
                        unit=test_result.unit,
                        normal_range=test_result.normal_range,
                        status=test_result.status,
                        severity_score=test_result.severity_score
                    )
                    db.session.add(test_record)
                
                db.session.commit()
                
                flash('Medical report uploaded and analyzed successfully!', 'success')
                return redirect(url_for('view_report', report_id=report.id))
                
            except Exception as e:
                flash(f'Error processing the file: {str(e)}', 'danger')
                # Clean up the uploaded file
                if os.path.exists(file_path):
                    os.remove(file_path)
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload PDF, PNG, JPG, or JPEG files only.', 'danger')
    
    return render_template('upload/index.html')

@app.route('/history')
@login_required
def patient_history():
    """View patient history"""
    # Get all medical reports for the current user
    reports = MedicalReport.query.filter_by(user_id=current_user.id)\
                                 .order_by(MedicalReport.upload_date.desc()).all()
    
    # Get health trend data for charts
    trend_data = get_health_trend_data(current_user.id)
    
    return render_template('history/index.html', reports=reports, trend_data=trend_data)

@app.route('/report/<int:report_id>')
@login_required
def view_report(report_id):
    """View detailed report analysis"""
    report = MedicalReport.query.filter_by(id=report_id, user_id=current_user.id).first_or_404()
    
    # Parse JSON data
    test_results = json.loads(report.test_results) if report.test_results else []
    disease_risks = json.loads(report.disease_risks) if report.disease_risks else []
    analysis_results = json.loads(report.analysis_results) if report.analysis_results else {}
    
    return render_template('reports/detail.html',
                         report=report,
                         test_results=test_results,
                         disease_risks=disease_risks,
                         analysis_results=analysis_results)

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """User profile management"""
    if request.method == 'POST':
        current_user.first_name = request.form['first_name']
        current_user.last_name = request.form['last_name']
        current_user.email = request.form['email']
        current_user.phone = request.form.get('phone', '')
        current_user.gender = request.form.get('gender', '')
        
        # Update password if provided
        if request.form.get('new_password'):
            if bcrypt.check_password_hash(current_user.password_hash, request.form['current_password']):
                current_user.password_hash = bcrypt.generate_password_hash(request.form['new_password']).decode('utf-8')
                flash('Password updated successfully!', 'success')
            else:
                flash('Current password is incorrect.', 'danger')
                return render_template('profile/index.html')
        
        db.session.commit()
        flash('Profile updated successfully!', 'success')
    
    return render_template('profile/index.html')

@app.route('/api/health-trends')
@login_required
def api_health_trends():
    """API endpoint for health trends data"""
    return jsonify(get_health_trend_data(current_user.id))

# Helper Functions

def allowed_file(filename):
    """Check if file type is allowed"""
    ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def determine_report_type(filename):
    """Determine report type from filename"""
    filename_lower = filename.lower()
    if any(word in filename_lower for word in ['cbc', 'blood', 'hematology']):
        return 'Blood Test'
    elif any(word in filename_lower for word in ['lipid', 'cholesterol']):
        return 'Lipid Panel'
    elif any(word in filename_lower for word in ['metabolic', 'chemistry']):
        return 'Metabolic Panel'
    elif any(word in filename_lower for word in ['thyroid', 'tsh']):
        return 'Thyroid Function'
    else:
        return 'General'

def get_health_trend_data(user_id):
    """Get health trend data for charts"""
    reports = MedicalReport.query.filter_by(user_id=user_id)\
                                 .filter(MedicalReport.health_score.isnot(None))\
                                 .order_by(MedicalReport.report_date).all()
    
    trend_data = {
        'dates': [report.report_date.strftime('%Y-%m-%d') for report in reports],
        'health_scores': [report.health_score for report in reports],
        'report_types': [report.report_type for report in reports]
    }
    
    # Get specific test trends
    test_trends = {}
    for report in reports:
        if report.test_results:
            tests = json.loads(report.test_results)
            for test in tests:
                test_name = test['test_name']
                if test_name not in test_trends:
                    test_trends[test_name] = {'dates': [], 'values': [], 'units': test.get('unit', '')}
                test_trends[test_name]['dates'].append(report.report_date.strftime('%Y-%m-%d'))
                test_trends[test_name]['values'].append(test['value'])
    
    trend_data['test_trends'] = test_trends
    
    return trend_data

# Initialize database
def init_db():
    """Initialize database tables"""
    with app.app_context():
        db.create_all()
        
        # Create default admin user if not exists
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin_user = User(
                username='admin',
                email='admin@healthcare.com',
                password_hash=bcrypt.generate_password_hash('admin123').decode('utf-8'),
                first_name='System',
                last_name='Administrator',
                role='admin'
            )
            db.session.add(admin_user)
            db.session.commit()
            print("Default admin user created: admin/admin123")

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
