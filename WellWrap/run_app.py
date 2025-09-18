#!/usr/bin/env python3
"""
Simple Medical Report Simplifier App Runner
Runs the Flask application with minimal dependencies
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

app = Flask(__name__, template_folder='frontend/templates', static_folder='frontend/static')
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

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

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

class MedicalReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    extracted_text = db.Column(db.Text)
    analysis_results = db.Column(db.Text)  # JSON string
    health_score = db.Column(db.Integer)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    report_date = db.Column(db.Date)
    report_type = db.Column(db.String(50))
    status = db.Column(db.String(20), default='pending')

# Simple medical analyzer
def simple_analyze_report(text):
    """Analyze medical report text and extract meaningful data"""
    if not text or not text.strip():
        return {
            'extracted_values': [],
            'health_score': None,
            'summary': 'No text found in the report for analysis.',
            'recommendations': ['Please ensure the report contains readable text']
        }
    
    analysis = {
        'extracted_values': [],
        'health_score': None,
        'summary': '',
        'recommendations': []
    }
    
    # Enhanced pattern matching for medical values
    import re
    patterns = {
        'hemoglobin': r'(?:hemoglobin|hgb|hb)[\s:]*([0-9]+\.?[0-9]*)\s*(?:g/dl|g\/dl|gm\/dl|g%)?',
        'glucose': r'(?:glucose|blood sugar|fasting glucose)[\s:]*([0-9]+\.?[0-9]*)\s*(?:mg/dl|mg\/dl|mmol/l)?',
        'cholesterol': r'(?:cholesterol|total cholesterol)[\s:]*([0-9]+\.?[0-9]*)\s*(?:mg/dl|mg\/dl|mmol/l)?',
        'blood_pressure_systolic': r'(?:bp|blood pressure)[\s:]*([0-9]+)\s*/\s*[0-9]+',
        'blood_pressure_diastolic': r'(?:bp|blood pressure)[\s:]*[0-9]+\s*/\s*([0-9]+)',
        'white_blood_cells': r'(?:wbc|white blood cells?)[\s:]*([0-9]+\.?[0-9]*)\s*(?:k/ul|k\/ul|x10\^3/ul)?',
        'red_blood_cells': r'(?:rbc|red blood cells?)[\s:]*([0-9]+\.?[0-9]*)\s*(?:m/ul|m\/ul|x10\^6/ul)?',
        'platelets': r'(?:platelets?|plt)[\s:]*([0-9]+\.?[0-9]*)\s*(?:k/ul|k\/ul|x10\^3/ul)?'
    }
    
    text_lower = text.lower()
    extracted_count = 0
    
    # Reference ranges for health scoring
    normal_ranges = {
        'hemoglobin': (12.0, 16.0),
        'glucose': (70, 100),
        'cholesterol': (0, 200),
        'blood_pressure_systolic': (90, 120),
        'blood_pressure_diastolic': (60, 80),
        'white_blood_cells': (4.0, 11.0),
        'red_blood_cells': (4.5, 5.9),
        'platelets': (150, 450)
    }
    
    for test_name, pattern in patterns.items():
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        if matches:
            try:
                value = float(matches[0])
                
                # Determine status based on normal ranges
                status = 'unknown'
                if test_name in normal_ranges:
                    min_val, max_val = normal_ranges[test_name]
                    if min_val <= value <= max_val:
                        status = 'normal'
                    else:
                        status = 'abnormal'
                
                analysis['extracted_values'].append({
                    'test': test_name.replace('_', ' ').title(),
                    'value': f"{value}",
                    'status': status
                })
                extracted_count += 1
            except (ValueError, IndexError):
                continue
    
    # Generate health score based on extracted values
    if extracted_count > 0:
        normal_count = sum(1 for val in analysis['extracted_values'] if val['status'] == 'normal')
        health_score = int((normal_count / extracted_count) * 100) if extracted_count > 0 else 50
        analysis['health_score'] = max(min(health_score, 100), 0)
    else:
        analysis['health_score'] = None
    
    # Generate summary
    if extracted_count > 0:
        analysis['summary'] = f"Successfully extracted {extracted_count} medical values from the report."
        
        normal_count = sum(1 for val in analysis['extracted_values'] if val['status'] == 'normal')
        abnormal_count = sum(1 for val in analysis['extracted_values'] if val['status'] == 'abnormal')
        
        if abnormal_count > 0:
            analysis['summary'] += f" {abnormal_count} values appear outside normal range."
            analysis['recommendations'].append('Consult with your healthcare provider about abnormal values')
        
        if normal_count > 0:
            analysis['recommendations'].append('Continue monitoring your health metrics')
            
        analysis['recommendations'].append('Regular medical check-ups are recommended')
    else:
        analysis['summary'] = 'No recognizable medical values found in the report text.'
        analysis['recommendations'] = [
            'Ensure the report contains clear medical test results',
            'Consider uploading a different format or clearer scan',
            'Consult with healthcare provider for proper interpretation'
        ]
    
    return analysis

def extract_text_from_pdf(file):
    """Extract text from PDF file with proper error handling"""
    try:
        import PyPDF2
        from io import BytesIO
        
        # Read file content
        file.seek(0)  # Ensure we're at the beginning
        file_content = file.read()
        
        if not file_content:
            return None, "PDF file appears to be empty"
        
        # Create PDF reader
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
        except Exception as e:
            return None, f"Unable to read PDF file: {str(e)}"
        
        # Check if PDF has pages
        if len(pdf_reader.pages) == 0:
            return None, "PDF file contains no pages"
        
        # Extract text from all pages
        text_parts = []
        for i, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text.strip():
                    text_parts.append(page_text.strip())
            except Exception as e:
                print(f"Warning: Could not extract text from page {i+1}: {e}")
                continue
        
        extracted_text = "\n\n".join(text_parts).strip()
        
        if not extracted_text:
            return None, "No readable text found in the PDF. The PDF might be image-based or encrypted."
        
        return extracted_text, None
        
    except ImportError:
        return None, "PDF text extraction requires PyPDF2 library. Please install it or upload as text file."
    except Exception as e:
        return None, f"Unexpected error extracting PDF: {str(e)}"

def extract_text_from_pdf_file(pdf_file):
    """Extract text from an already opened PDF file object"""
    try:
        import PyPDF2
        
        # Create PDF reader from file object
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
        except Exception as e:
            return None, f"Unable to read PDF file: {str(e)}"
        
        # Check if PDF has pages
        if len(pdf_reader.pages) == 0:
            return None, "PDF file contains no pages"
        
        # Extract text from all pages
        text_parts = []
        for i, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text.strip():
                    text_parts.append(page_text.strip())
            except Exception as e:
                print(f"Warning: Could not extract text from page {i+1}: {e}")
                continue
        
        extracted_text = "\n\n".join(text_parts).strip()
        
        if not extracted_text:
            return None, "No readable text found in the PDF. The PDF might be image-based or encrypted."
        
        return extracted_text, None
        
    except ImportError:
        return None, "PDF text extraction requires PyPDF2 library. Please install it or upload as text file."
    except Exception as e:
        return None, f"Unexpected error extracting PDF: {str(e)}"

# Routes
@app.route('/')
def home():
    """Home page"""
    try:
        return render_template('index.html')
    except Exception as e:
        # Fallback if template rendering fails
        return f"""
        <html>
        <head><title>Healthcare Portal</title></head>
        <body>
            <h1>🏥 Medical Report Simplifier</h1>
            <p>Welcome to the Healthcare Portal!</p>
            <p><a href="/register">Register</a> | <a href="/login">Login</a></p>
            <hr>
            <p><strong>Template Error:</strong> {str(e)}</p>
        </body>
        </html>
        """

@app.route('/test')
def test():
    """Simple test route"""
    return {"status": "ok", "message": "Flask app is working!", "users": User.query.count()}

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose a different one.', 'danger')
            return render_template('auth/register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered. Please use a different email.', 'danger')
            return render_template('auth/register.html')
        
        # Handle date of birth
        date_of_birth_str = request.form.get('date_of_birth')
        if date_of_birth_str:
            try:
                date_of_birth = datetime.strptime(date_of_birth_str, '%Y-%m-%d').date()
            except ValueError:
                date_of_birth = datetime.now().date()
        else:
            date_of_birth = datetime.now().date()
        
        # Create new user
        password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            first_name=first_name,
            last_name=last_name,
            date_of_birth=date_of_birth,
            phone=request.form.get('phone', ''),
            gender=request.form.get('gender', '')
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
    
    # Get upcoming appointments (empty list for now since Appointment model may not be fully implemented)
    upcoming_appointments = []
    
    # Get recent health metrics (empty list for now since HealthMetric model may not be fully implemented)
    recent_metrics = []
    
    # Calculate health statistics
    total_reports = MedicalReport.query.filter_by(user_id=current_user.id).count()
    
    # Calculate average health score from actual reports
    health_scores = [r.health_score for r in recent_reports if r.health_score is not None]
    avg_health_score = sum(health_scores) / len(health_scores) if health_scores else 0
    
    # Ensure we have an integer value for template comparisons
    avg_health_score = int(avg_health_score) if avg_health_score else 0
    
    # Active medications count (0 for now since Medication model may not be fully implemented)
    active_medications = 0
    
    return render_template('dashboard/index.html',
                         recent_reports=recent_reports,
                         upcoming_appointments=upcoming_appointments,
                         recent_metrics=recent_metrics,
                         total_reports=total_reports,
                         avg_health_score=avg_health_score,
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
        
        if file:
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Extract text based on file type
            extracted_text = None
            extraction_error = None
            
            if filename.lower().endswith('.pdf'):
                # For PDF files, read from saved file path
                try:
                    with open(file_path, 'rb') as pdf_file:
                        extracted_text, extraction_error = extract_text_from_pdf_file(pdf_file)
                except Exception as e:
                    extraction_error = f"Error accessing saved PDF file: {str(e)}"
                    extracted_text = None
            else:
                # For text files, read content
                try:
                    file.seek(0)  # Reset file pointer
                    file_content = file.read()
                    
                    # Try different encodings
                    encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
                    for encoding in encodings:
                        try:
                            extracted_text = file_content.decode(encoding, errors='ignore').strip()
                            if extracted_text:
                                break
                        except UnicodeDecodeError:
                            continue
                    
                    if not extracted_text:
                        extraction_error = "Could not decode text file content"
                        
                except Exception as e:
                    extraction_error = f"Error reading text file: {str(e)}"
            
            # Handle extraction errors
            if extraction_error or not extracted_text or not extracted_text.strip():
                error_message = extraction_error or 'Could not extract readable text from the uploaded file.'
                flash(error_message, 'danger')
                return redirect(request.url)
            
            # Analyze the report
            analysis = simple_analyze_report(extracted_text)
            
            # Create medical report record
            report = MedicalReport(
                user_id=current_user.id,
                filename=unique_filename,
                original_filename=filename,
                file_path=file_path,
                extracted_text=extracted_text,
                analysis_results=json.dumps(analysis),
                health_score=analysis['health_score'],
                report_date=datetime.now().date(),
                report_type='General',
                status='analyzed'
            )
            
            db.session.add(report)
            db.session.commit()
            
            flash('Medical report uploaded and analyzed successfully!', 'success')
            return redirect(url_for('view_report', report_id=report.id))
    
    return render_template('upload/index.html')

@app.route('/report/<int:report_id>')
@login_required
def view_report(report_id):
    """View detailed report analysis"""
    report = MedicalReport.query.filter_by(id=report_id, user_id=current_user.id).first_or_404()
    
    analysis_results = json.loads(report.analysis_results) if report.analysis_results else {}
    
    return render_template('reports/detail.html',
                         report=report,
                         analysis_results=analysis_results)

@app.route('/history')
@login_required
def patient_history():
    """View patient history"""
    reports = MedicalReport.query.filter_by(user_id=current_user.id)\
                                 .order_by(MedicalReport.upload_date.desc()).all()
    
    return render_template('history/index.html', reports=reports)

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """User profile management"""
    if request.method == 'POST':
        current_user.first_name = request.form.get('first_name', current_user.first_name)
        current_user.last_name = request.form.get('last_name', current_user.last_name)
        current_user.email = request.form.get('email', current_user.email)
        current_user.phone = request.form.get('phone', current_user.phone)
        current_user.gender = request.form.get('gender', current_user.gender)
        
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

def allowed_file(filename):
    """Check if file type is allowed"""
    ALLOWED_EXTENSIONS = {'pdf', 'txt', 'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize database
def init_db():
    """Initialize database tables"""
    with app.app_context():
        db.create_all()
        
        # Create demo user if not exists
        demo_user = User.query.filter_by(username='demo_patient').first()
        if not demo_user:
            demo_user = User(
                username='demo_patient',
                email='demo@healthcare.com',
                password_hash=bcrypt.generate_password_hash('demo123').decode('utf-8'),
                first_name='Demo',
                last_name='Patient',
                date_of_birth=datetime.now().date()
            )
            db.session.add(demo_user)
            db.session.commit()
            print("Demo user created: demo_patient/demo123")

if __name__ == '__main__':
    print("🏥 Starting WellWrap...")
    print("📂 Initializing database...")
    init_db()
    
    # Check database status by querying users
    db_path = 'healthcare.db'
    try:
        with app.app_context():
            user_count = User.query.count()
            print(f"✅ Database operational: {os.path.abspath(db_path)}")
            print(f"👥 Users in database: {user_count}")
            if user_count > 0:
                print(f"🔐 Existing users can login directly")
    except Exception as e:
        print(f"⚠️ Database error: {e}")
    
    print("📚 Web Application: http://localhost:5000")
    print("👤 Demo Login: demo_patient / demo123")
    print("📋 Upload medical reports and get AI analysis!")
    print("💾 All user data persists between sessions")
    print("Press Ctrl+C to stop")
    app.run(debug=True, host='0.0.0.0', port=5000)
