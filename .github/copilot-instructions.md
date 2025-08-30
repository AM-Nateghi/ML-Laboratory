# ML Laboratory Repository Instructions

**ALWAYS follow these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Repository Overview

ML Laboratory contains 5 machine learning projects:
1. **Sales Prediction** - Complete FastAPI web application with ML models (MAIN PROJECT)
2. **Spam Email Detection** - CSV dataset only
3. **Machine Failure Prediction** - CSV dataset only  
4. **Movie Recommendation System** - CSV datasets only
5. **Corona Virus Detection** - CSV dataset and old notebook

## Working Effectively

### Environment Setup
Run these commands to set up the environment. NEVER CANCEL builds - they take time to complete:

```bash
# Create conda environment and install dependencies
conda install -c conda-forge fastapi uvicorn pydantic pandas scikit-learn joblib kneed numpy scipy -y
# Installation takes 5-10 minutes. NEVER CANCEL. Set timeout to 15+ minutes.
```

**CRITICAL**: pip install has network limitations in this environment. ALWAYS use conda for package installation.

### Build and Train Models
```bash
cd "1.Sales Prediction"

# Run training pipeline to create models
/usr/share/miniconda/bin/python train.py
# Training takes 2 minutes 15 seconds. NEVER CANCEL. Set timeout to 5+ minutes.

# Organize model files  
mkdir -p saved_models
mv *.pkl saved_models/

# Test model performance
/usr/share/miniconda/bin/python test_train.py  
# Testing takes 20 seconds. NEVER CANCEL. Set timeout to 1+ minute.
```

### Run the Sales Prediction Web Application
```bash
cd "1.Sales Prediction"

# Start FastAPI server
/usr/share/miniconda/bin/python main.py
# Server starts immediately and runs on http://localhost:2007
# Access web UI at: http://localhost:2007/
# API endpoint: http://localhost:2007/predict (POST)
```

## Validation Requirements

### ALWAYS manually validate changes to Sales Prediction project:

1. **Build validation**: Run training pipeline and verify model files are created
2. **API validation**: Start server and test prediction endpoint:
   ```bash
   curl -X POST "http://localhost:2007/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Customers": 800,
       "Promo": true,
       "StateHoliday": false,
       "SchoolHoliday": false,
       "StoreType": 1,
       "Assortment": 1,
       "CompetitionDistance": 500,
       "CompetitionOpenSinceMonth": 1,
       "CompetitionOpenSinceYear": 2010,
       "HasCompetition": true,
       "Promo2": false,
       "Promo2SinceWeek": 0,
       "Promo2SinceYear": 0,
       "PromoInterval": 0,
       "month": 6,
       "year": 15
     }'
   ```
   Expected response: `{"success":true,"cluster":1,"prediction":...}`

3. **Performance validation**: Verify R² scores are above 0.9 in test output

## Critical Build Information

### Timing Expectations (NEVER CANCEL)
- **Conda installation**: 5-10 minutes. Set timeout to 15+ minutes.
- **Model training**: 2 minutes 15 seconds. Set timeout to 5+ minutes.  
- **Model testing**: 20 seconds. Set timeout to 1+ minute.
- **Server startup**: Immediate (< 5 seconds)

### Python Environment
- **ALWAYS use**: `/usr/share/miniconda/bin/python` for Sales Prediction project
- **System python3**: Missing required packages, not suitable for ML tasks
- **Package manager**: Use conda only, pip has network timeouts

### Known Issues and Workarounds
- **hdbscan import error**: Already commented out in train.py, ignore if encountered
- **pip network timeouts**: Use conda instead: `conda install -c conda-forge <package>`
- **ModuleNotFoundError**: Use conda python: `/usr/share/miniconda/bin/python`

## Project Structure and Navigation

### Sales Prediction (1.Sales Prediction/)
- `main.py` - FastAPI web application and prediction API
- `train.py` - ML training pipeline (clustering + regression models)
- `test_train.py` - Model performance validation
- `train.ipynb` - Jupyter notebook for exploration
- `static/` - Web UI files (HTML, CSS, JavaScript)
- `saved_models/` - Trained model files (created by train.py)
- `RoS_train.csv`, `RoS_test.csv` - Training and test datasets

### Other Projects
- Projects 2-5 contain only CSV datasets and links to Kaggle sources
- No executable code or build requirements
- Used for data analysis and experimentation only

## Common Development Tasks

### Adding New Features to Sales Prediction
1. Run training pipeline first to ensure models exist
2. Make code changes to main.py or related files
3. Test API endpoint with sample data
4. Verify web UI still loads at http://localhost:2007
5. Check model performance hasn't degraded

### Debugging Model Issues
1. Check `saved_models/` directory exists with .pkl files
2. Verify training completed without errors
3. Test model loading in Python: `import joblib; joblib.load('saved_models/sales_kmeans_models.pkl')`
4. Run test_train.py to check R² scores

### Data Processing Changes
1. Modify train.py for new preprocessing steps
2. Re-run complete training pipeline (2+ minutes)
3. Validate with test_train.py
4. Test API predictions with new data format

## Frequently Used Commands

```bash
# Check conda environment and packages
conda list | grep -E "(pandas|sklearn|fastapi)"

# Quick model retrain (from Sales Prediction directory)
/usr/share/miniconda/bin/python train.py && mkdir -p saved_models && mv *.pkl saved_models/

# Start development server with auto-reload
cd "1.Sales Prediction" && /usr/share/miniconda/bin/python main.py

# Test model performance
cd "1.Sales Prediction" && /usr/share/miniconda/bin/python test_train.py

# Check server health
curl -s http://localhost:2007/ | head -5
```

## Repository File Inventory

### Root Directory
```
.git/
.gitignore
1.Sales Prediction/     # Main project
2.Spam Email Detection/ # Dataset only
3.Machine Failure Prediction/ # Dataset only  
4.Movie Recommendation System/ # Dataset only
5.Corona Virus Detection/ # Dataset + old notebook
```

### 1.Sales Prediction/ Contents
```
main.py              # FastAPI app
train.py             # ML training pipeline
test_train.py        # Model validation
train.ipynb          # Jupyter notebook
static/              # Web UI assets
  ├── sales_prediction_ui.html
  ├── script.js
  ├── style.css
  └── index.html
RoS_train.csv        # Training data (9.7MB)
RoS_test.csv         # Test data (9.9MB)
link.txt             # Kaggle dataset source
saved_models/        # Generated model files
  ├── sales_kmeans_models.pkl (393MB)
  └── sales_spectral_models.pkl (363MB)
```

## Dependencies Summary

### Core ML Stack (conda-forge)
- pandas, numpy, scipy
- scikit-learn, joblib
- kneed (for elbow method)

### Web Framework (conda-forge)  
- fastapi, uvicorn
- pydantic (validation)

### Development Tools
- Python 3.13 (conda)
- curl (API testing)
- time (performance measurement)