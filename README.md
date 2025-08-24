# Process Capability & Stability Analysis Web Application

A comprehensive Streamlit-based web application for statistical process control, capability analysis, and stability monitoring.

## Features

### 📊 Core Analysis Capabilities
- **Data Upload & Validation**: Support for CSV/Excel files with automatic data validation
- **Basic Statistics**: Descriptive statistics, normality testing, and outlier detection
- **Distribution Fitting**: Compare multiple distributions (Normal, Lognormal, Weibull, Gamma, Logistic)
- **Process Capability Indices**: Cp, Cpk, Pp, Ppk with confidence intervals
- **Tolerance Intervals**: SAS METHOD=3 implementation for one-sided and two-sided intervals
- **%OOS Calculation**: Multiple methods (fitted distribution, empirical, tolerance interval)
- **Control Charts**: I-MR charts with stability analysis

### 🎯 Key Algorithms
- **METHOD=3 Tolerance Intervals**: Exact one-sided and Howe's approximation for two-sided
- **Distribution-based Capability**: Support for non-normal distributions using quantile methods
- **Coverage Scanning**: Automated scan to find maximum coverage within specifications
- **Statistical Tests**: Anderson-Darling normality test, Kolmogorov-Smirnov goodness-of-fit

## Installation

1. **Clone or download the project**
   ```bash
   cd /Users/jiamingxu/Downloads/all_stats
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app_new.py
   ```

## Usage

### Step-by-Step Analysis Workflow

1. **Upload Data**: Upload your CSV or Excel file with numeric data
2. **Basic Statistics**: Review descriptive statistics and normality test results
3. **Distribution Comparison**: Compare multiple distribution fits and select the best one
4. **Capability Indices**: Calculate Cp, Cpk, Pp, Ppk with confidence intervals
5. **Tolerance Intervals**: Calculate tolerance intervals using METHOD=3
6. **%OOS Results**: Compare different %OOS calculation methods
7. **Control Charts**: View I-MR charts for process stability
8. **Export Results**: Download comprehensive reports

### Data Format Requirements
- At least one numeric column for analysis
- Optional: timestamp, batch ID, grouping columns
- Supported formats: CSV, Excel (.xlsx, .xls)

### Specification Limits
- **LSL**: Lower Specification Limit
- **USL**: Upper Specification Limit  
- **Target**: Target value (optional)
- Support for both one-sided and two-sided specifications

## Technical Architecture

### Project Structure
```
all_stats/
├── app_new.py             # Main Streamlit application
├── requirements.txt       # Python dependencies
├── test_app.py           # Test script
├── services/             # Core calculation modules
│   ├── capability.py     # Capability indices calculation
│   ├── distributions.py  # Distribution fitting and comparison
│   ├── statistics.py     # Basic statistical functions
│   ├── tolerance.py      # Tolerance interval algorithms
│   └── oos.py           # %OOS calculation methods
├── utils/                # Utility functions
│   └── validators.py     # Data validation and checks
├── assets/               # Static assets
│   └── style.css         # Custom CSS styles
└── tests/                # Test directory
```

### Key Algorithms Implemented

#### Tolerance Intervals (METHOD=3)
- **One-sided**: Exact solution using non-central t-distribution
- **Two-sided**: Howe's approximation and exact numerical solution
- **Coverage Scanning**: Automated scan to find maximum p* within specifications

#### Distribution Fitting
- Maximum likelihood estimation for multiple distributions
- AIC/BIC comparison for model selection
- Goodness-of-fit testing (Kolmogorov-Smirnov)

#### Capability Indices
- **Normal distribution**: Traditional formulas with analytic confidence intervals
- **Non-normal distributions**: Quantile-based approach using fitted distributions
- **Confidence Intervals**: Analytic methods for normal, bootstrap for non-normal

## Example Usage

### Test the Application
```bash
python test_app.py  # Note: Update test to use app_new.py
```

### Run with Sample Data
1. The application includes built-in validation for sample data
2. Upload a CSV file with a numeric column
3. Enter specification limits (LSL/USL)
4. Follow the step-by-step analysis workflow

### Sample Data Format
```csv
value,timestamp,batch
10.2,2024-01-01,A
10.5,2024-01-01,A
9.8,2024-01-02,B
10.1,2024-01-02,B
...
```

## Validation & Testing

The application includes comprehensive validation:
- Data quality checks (missing values, outliers, constant columns)
- Specification limit validation
- Statistical assumption checking
- Error handling and user feedback

## Performance Considerations

- **Caching**: Streamlit caching for expensive computations
- **Large datasets**: Automatic sampling for visualization
- **Numerical stability**: Robust statistical algorithms
- **Memory efficiency**: Pandas-based data processing

## Browser Compatibility

- Modern browsers with JavaScript enabled
- Recommended: Chrome, Firefox, Safari, Edge
- Mobile-responsive design

## License & Attribution

This application implements standard statistical methods for process capability analysis. The SAS METHOD=3 implementation follows established statistical literature.

## Support

For issues or questions, please check the application documentation or consult statistical references for methodology details.