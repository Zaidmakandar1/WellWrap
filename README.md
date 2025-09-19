🏥 WellWrap

An AI-powered healthcare application that transforms complex medical reports into easy-to-understand health insights. Built with Flask, featuring medical data analysis, health scoring, and personalized recommendations.








🌟 Features
🔬 Medical Analysis

📄 PDF Text Extraction from medical reports

🧠 Pattern Recognition for lab values and test metrics

🎯 Health Scoring (0–100 scale)

🚩 Risk Detection (anemia, cardiovascular, diabetes)

💡 Personalized Recommendations

📊 Dashboard & Analytics

📈 Interactive health dashboards

🕒 Report history and progress tracking

📉 Visual health trend charts

📤 Report and data exports

🔐 User Management

🔑 Secure authentication

👤 User profile management

🛡️ HIPAA-compliant data design

👥 Multi-user support

🤖 AI-Powered Analysis

🩸 Blood test breakdowns (CBC, Lipid, Metabolic)

🧬 Disease risk prediction

📋 Health advice based on evidence

📊 Trend analysis over time

🚀 Quick Start
Prerequisites

Python 3.8 or higher

pip (Python package installer)

Git

Installation
# Clone the repository
git clone https://github.com/yourusername/wellwrap.git
cd wellwrap

# Install dependencies
pip install -r requirements.txt

# Initialize the database
python init_database.py --init

# Run the application
python start_app.py


Visit: http://localhost:5000

📋 Usage
1. Register & Login

Create an account and securely log in

2. Upload Medical Reports

Upload PDFs or text files

Automatic extraction and secure storage

3. View Analysis Results

Get a health score (0–100)

View risk flags and health insights

Receive recommendations

4. Track Progress

See historical comparisons

Monitor trends and changes over time

🏗️ Project Structure
wellwrap/
├── backend/           # Flask backend API
│   ├── api/           # API routes
│   ├── models/        # Database schemas
│   ├── services/      # Business logic
│   └── app.py
├── frontend/          # UI templates & assets
│   ├── templates/
│   ├── static/
│   └── components/
├── ml/                # ML analysis logic
│   ├── models/
│   ├── services/
│   ├── data_processing/
│   └── streamlit_app/
├── data/              # Sample & processed data
├── tests/             # Unit and integration tests
├── docs/              # Documentation
└── scripts/           # Helper and init scripts

📊 Medical Analysis Capabilities
Supported Test Types

CBC (Complete Blood Count)

Hemoglobin, WBC, Platelets

Detects anemia and infections

Lipid Profile

Cholesterol, LDL/HDL, Triglycerides

Assesses heart health

Metabolic Panel

Glucose, Creatinine, Liver function

Evaluates diabetes and kidney function

Health Score Scale
Score Range	Interpretation
85–100	✅ Excellent
70–84	👍 Good
55–69	⚠️ Fair
< 55	🚨 Needs Attention
🛡️ Security & Privacy
Data Protection

🔐 Encrypted data storage

🔒 HTTPS communication

👤 User-level data isolation

📝 Activity logs for audits

HIPAA-Compliant Practices

📉 Minimal data collection

📃 User consent required

📁 Customizable data retention

🔍 Breach detection & prevention

Version 1.5 (In Progress)

 Advanced analysis logic

 Improved UI/UX

 Better logging

 Swagger API docs

 Richer report generation
