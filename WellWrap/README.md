# 🏥 WellWrap

An AI-powered healthcare application that transforms complex medical reports into easy-to-understand health insights. Built with Flask, featuring medical data analysis, health scoring, and personalized recommendations.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v2.3+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 🌟 Features

### 🔬 **Medical Analysis**
- **PDF Text Extraction** - Extract text from medical reports
- **Pattern Recognition** - Identify medical values and test results
- **Health Scoring** - 0-100 scale health assessment
- **Risk Detection** - Identify potential health risks
- **Personalized Recommendations** - Tailored health advice

### 📊 **Dashboard & Analytics**
- **Interactive Dashboard** - Real-time health metrics
- **Report History** - Track health progress over time
- **Visual Charts** - Health trends and analytics
- **Export Features** - Download reports and data

### 🔐 **User Management**
- **Secure Authentication** - User registration and login
- **Profile Management** - Personal health information
- **Data Privacy** - HIPAA-compliant design patterns
- **Multi-user Support** - Individual user accounts

### 🤖 **AI-Powered Analysis**
- **Blood Test Analysis** - CBC, lipid panels, metabolic panels
- **Disease Risk Assessment** - Anemia, cardiovascular, diabetes
- **Health Recommendations** - Evidence-based suggestions
- **Trend Analysis** - Historical health data insights

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/wellwrap.git
   cd wellwrap
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize the database**
   ```bash
   python init_database.py --init
   ```

4. **Run the application**
   ```bash
   python run_app.py
   ```

5. **Access the application**
   - Open your browser and go to: http://localhost:5000
   - Register a new account or use demo credentials

### Alternative: Enhanced Version
For a simplified version with core features:
```bash
python enhanced_app.py
```

## 📋 Usage

### 1. **Register & Login**
- Create a new account with your personal information
- Secure login with encrypted passwords

### 2. **Upload Medical Reports**
- Support for PDF files and text documents
- Automatic text extraction and processing
- Secure file storage

### 3. **View Analysis Results**
- Health score calculation (0-100)
- Detailed test result breakdown
- Risk factor identification
- Personalized recommendations

### 4. **Track Health Progress**
- Historical report comparison
- Health trend visualization
- Progress monitoring

## 🏗️ Project Structure

```
wellwrap/
├── 📁 backend/                 # Backend API and business logic
│   ├── 📁 api/                # API routes and endpoints
│   ├── 📁 models/             # Database models
│   ├── 📁 services/           # Business logic services
│   └── app.py                 # Main Flask application
├── 📁 frontend/               # Frontend templates and assets
│   ├── 📁 templates/          # Jinja2 HTML templates
│   ├── 📁 static/             # CSS, JavaScript, images
│   └── 📁 components/         # Reusable UI components
├── 📁 ml/                     # Machine Learning components
│   ├── 📁 models/             # ML model definitions
│   ├── 📁 services/           # ML analysis services
│   ├── 📁 data_processing/    # Data preprocessing
│   └── 📁 streamlit_app/      # Streamlit ML interface
├── 📁 data/                   # Data storage
│   ├── 📁 samples/            # Sample medical reports
│   └── 📁 processed/          # Processed data
├── 📁 tests/                  # Test suites
├── 📁 docs/                   # Documentation
└── 📁 scripts/                # Utility scripts
```

## 🧪 Testing

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Test Specific Components
```bash
# Test database functionality
python test_upload_functionality.py

# Test medical analysis
python test_streamlit_fix.py

# Test Flask app
python test_flask_minimal.py --diagnose
```

### Sample Data Testing
Use the provided sample files in `data/samples/`:
- `sample_cbc_report.txt` - Complete Blood Count report
- `sample_lipid_report.txt` - Lipid profile report

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the root directory:
```env
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///healthcare.db
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=16777216
DEBUG=True
```

### Database Configuration
The application uses SQLite by default. For production, configure PostgreSQL:
```python
SQLALCHEMY_DATABASE_URI = 'postgresql://user:password@localhost/healthcare'
```

## 📊 Medical Analysis Capabilities

### Supported Test Types
- **Complete Blood Count (CBC)**
  - Hemoglobin, Hematocrit, WBC, Platelets
  - Anemia detection and classification
  - Infection indicators

- **Lipid Profiles**
  - Total Cholesterol, LDL, HDL, Triglycerides
  - Cardiovascular risk assessment
  - Treatment recommendations

- **Metabolic Panels**
  - Glucose, Creatinine, Liver function
  - Diabetes risk evaluation
  - Kidney function assessment

### Health Scoring Algorithm
- **Excellent (85-100)**: All values within normal range
- **Good (70-84)**: Minor abnormalities, low risk
- **Fair (55-69)**: Some concerning values, moderate risk
- **Needs Attention (<55)**: Multiple abnormalities, high risk

## 🛡️ Security & Privacy

### Data Protection
- **Encrypted Storage** - All sensitive data encrypted at rest
- **Secure Transmission** - HTTPS for all communications
- **Access Control** - User-based data isolation
- **Audit Logging** - Comprehensive activity tracking

### HIPAA Compliance
- **Data Minimization** - Only collect necessary information
- **User Consent** - Clear privacy policies
- **Data Retention** - Configurable retention policies
- **Breach Prevention** - Security best practices

## 🚀 Deployment

### Local Development
```bash
python run_app.py
```

### Production Deployment
```bash
# Using Gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 run_app:app

# Using Docker
docker build -t medical-report-simplifier .
docker run -p 8000:8000 medical-report-simplifier
```

### Cloud Deployment
- **Heroku**: Ready for Heroku deployment
- **AWS**: EC2, ECS, or Lambda deployment
- **Google Cloud**: App Engine or Cloud Run
- **Azure**: App Service deployment

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `pytest`
5. Submit a pull request

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Write tests for new features

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Medical Community** - For providing insights into healthcare needs
- **Open Source Libraries** - Flask, SQLAlchemy, PyTorch, and others
- **Contributors** - Everyone who has contributed to this project

## 📞 Support

- **Documentation**: Check the `/docs` directory
- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Join our GitHub Discussions
- **Email**: support@medical-report-simplifier.com

## 🗺️ Roadmap

### Version 2.0 (Planned)
- [ ] Advanced ML models for disease prediction
- [ ] Integration with wearable devices
- [ ] Telemedicine consultation features
- [ ] Mobile application (React Native)
- [ ] Multi-language support

### Version 1.5 (In Progress)
- [x] Enhanced medical analysis algorithms
- [x] Improved user interface
- [x] Better error handling and logging
- [ ] API documentation with Swagger
- [ ] Advanced reporting features

---

**⚠️ Medical Disclaimer**: This application is for informational purposes only and should not replace professional medical advice. Always consult with qualified healthcare providers for medical decisions.

**Built with ❤️ for better healthcare outcomes**