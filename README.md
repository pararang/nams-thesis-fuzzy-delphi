# Fuzzy Delphi and MCDM Analysis Tool

A comprehensive Multi-Criteria Decision Making (MCDM) tool that integrates Fuzzy Delphi Method (FDM), DEMATEL, and House of Quality (HOQ) methodologies for advanced data analysis in research applications.

## Overview

This project was specifically designed to assist my ❤️ wife in processing data for her thesis research on the implementation of Sustainability Principles to support the performance of Small and Medium Enterprises in Makassar. The tool has evolved beyond its original FDM-TFN scope to include advanced MCDM methodologies including DEMATEL for causal relationship analysis and HOQ for quality planning. This comprehensive tool streamlines research analysis, offering both a convenient web interface and a robust command-line script.

For deeper insights into the thesis's methodologies or subject matter, please reach out to nurastrimufthias@gmail.com.

## What is FDM, DEMATEL and HOQ?

### Fuzzy Delphi Method (FDM)
The Fuzzy Delphi Method is a structured communication technique that uses rounds of anonymous questionnaires to reach consensus among experts. In this tool, it's specifically implemented with Triangular Fuzzy Numbers (TFN) to handle uncertainty in expert opinions and provide more realistic results.

### DEMATEL (Decision Making Trial and Evaluation Laboratory)
DEMATEL is an advanced MCDM method used to identify and analyze causal relationships between factors in complex systems. It generates a causal diagram showing which factors are "cause" factors (influencing others) and which are "effect" factors (being influenced by others).

### House of Quality (HOQ)
HOQ is a matrix-based tool used to translate customer requirements into technical characteristics. It's a key component of Quality Function Deployment (QFD) that helps prioritize product development efforts based on customer needs.

## Features

### Fuzzy Delphi Method (FDM)
- Conversion of Likert scale scores (1-4) to Triangular Fuzzy Numbers (TFN)
- Calculation of mean TFN for each indicator
- Computation of consensus distances for each respondent
- Defuzzification of TFN values using the formula: A = (l + 2m + 2u) / 4
- Summary statistics and visualization
- Export options for calculation details and summary results

### DEMATEL Analysis
- Upload direct relation matrix CSV files
- Compute normalized matrix (T) and total relation matrix (T*)
- Calculate prominence (D+R) and net influence (D-R) vectors
- Identify cause and effect factors
- Generate causal diagrams
- Download analysis results

### HOQ Analysis
- Standalone House of Quality analysis
- Integrated HOQ-DEMATEL analysis
- Customer requirements to technical characteristics mapping
- Relationship matrix with customizable scales
- Technical correlation matrix (optional)
- Visual comparison of baseline vs. DEMATEL-adjusted importance

## Quick Start

### Option 1: Docker (Recommended)
```bash
# Clone the repository
git clone https://github.com/pararang/nams-thesis.git
cd nams-thesis

# Run in development mode with live reloading
make dev
```

### Option 2: Direct Python
```bash
# Clone the repository
git clone https://github.com/pararang/nams-thesis.git
cd nams-thesis

# Install dependencies
pip install -r requirements.txt

# Run the web application
streamlit run app.py
```

The application will be available at http://localhost:8501

## Installation

```bash
# Clone the repository
git clone https://github.com/pararang/nams-thesis.git
cd nams-thesis

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.6+
- pandas
- numpy
- streamlit (for UI version)
- xlsxwriter (for Excel export)
- pyDEMATEL (for DEMATEL analysis)
- matplotlib (for visualization)

## Requirements

- Python 3.6+
- pandas
- numpy
- streamlit (for UI version)
- xlsxwriter (for Excel export)

## Usage

The application provides three main tabs in the web interface:

### 1. Fuzzy Delphi TFN
Process survey data using Fuzzy Delphi Method with Triangular Fuzzy Numbers:


1. Upload a CSV file with survey data
2. View the raw data and TFN conversion results
3. See detailed calculations for each indicator
4. Download calculation details and summary results

### 2. DEMATEL Analysis
Analyze causal relationships between factors:

1. Upload a direct relation matrix CSV file
2. Compute normalized matrix (T) and total relation matrix (T*)
3. View cause-effect analysis results
4. Download matrices and results

### 3. HOQ-DEMATEL Integration
Integrate House of Quality with DEMATEL analysis:

1. Choose between standalone HOQ or integrated HOQ-DEMATEL analysis
2. Input customer requirements and technical characteristics
3. Define relationship matrix and importance weights
4. In integrated mode, add DEMATEL factors and relationships
5. View results with visualizations

### Docker Development

For a containerized development environment:

```bash
# Build and run with live code reloading
make dev

# Other available commands
make build    # Build the Docker image
make run      # Run the container (production mode)
make stop     # Stop the running container
make clean    # Remove the Docker image
```

The application will be available at http://localhost:8501 with live code reloading enabled.

### Command-Line Script Version

For batch processing or automation, use the script version:

```bash
python fuzzy_script.py input_data.csv
```

The script will:
1. Process the input CSV file
2. Display calculation steps in the terminal
3. Generate a summary CSV file with results

### Input Data Formats

#### FDM-TFN Input Data Format
The input CSV file should have the following format:
- First column: Respondent code or name
- Subsequent columns: Indicator scores (1-4)
- First row: Column headers (indicator names)

Example:
```
Respondent,Indicator1,Indicator2,Indicator3
R1,3,4,2
R2,4,3,3
R3,2,4,4
```

#### DEMATEL Input Data Format
The direct relation matrix CSV should be a square matrix:
- Both rows and columns represent factors/indicators
- Values indicate the degree of influence from row factor to column factor
- First row and column contain factor names

Example:
```
,Factor1,Factor2,Factor3
Factor1,0,2,1
Factor2,1,0,3
Factor3,2,1,0
```

### TFN Mapping

The tool uses the following mapping from Likert scale to TFN:
- 1 → (0, 0, 0.25)
- 2 → (0, 0.25, 0.5)
- 3 → (0.25, 0.5, 0.75)
- 4 → (0.5, 0.75, 1.0)

## Output

The tool generates:
1. Detailed calculation tables for each indicator
2. Mean TFN values (L, M, U) for each indicator
3. Consensus distances for each respondent
4. Defuzzified values for each indicator
5. Summary table with all results

## Features Comparison

| Feature                     | FDM-TFN       | DEMATEL        | HOQ              | HOQ-DEMATEL Integration     |
| --------------------------- | ------------- | -------------- | ---------------- | --------------------------- |
| Causal Analysis             | ❌             | ✅              | ❌                | ✅                           |
| Expert Consensus            | ✅             | ❌              | ❌                | ❌                           |
| Customer Requirements Focus | ❌             | ❌              | ✅                | ✅                           |
| Technical Characteristics   | ❌             | ❌              | ✅                | ✅                           |
| Relationship Matrix         | ❌             | ✅              | ✅                | ✅                           |
| Visualization               | Limited       | Causal Diagram | Bar Charts       | Causal Diagram + Bar Charts |
| Data Format                 | Survey Scores | Square Matrix  | Matrix + Weights | Matrix + Weights + DEMATEL  |

## Project Structure

```
nams-thesis/
├── app.py                 # Main Streamlit application
├── fdm_tfn.py             # Fuzzy Delphi Method with TFN implementation
├── dematel.py             # DEMATEL analysis implementation
├── hoq_integration.py     # HOQ and integrated HOQ-DEMATEL implementation
├── fuzzy_script.py        # Command-line script version
├── sample_input.csv       # Example input data file
├── Dockerfile             # Docker configuration
├── requirements.txt       # Python dependencies
├── Makefile               # Docker build and run commands
└── README.md              # This file
```

## Methodology

The tool combines multiple Multi-Criteria Decision Making (MCDM) techniques:

### Fuzzy Delphi Method
1. Experts provide scores on Likert scale (1-4)
2. Convert scores to Triangular Fuzzy Numbers (TFN)
3. Calculate mean TFN values for each indicator
4. Compute consensus distances to identify agreement among experts
5. Defuzzify final values for decision making

### DEMATEL
1. Create direct relation matrix based on expert evaluations
2. Normalize the direct relation matrix
3. Compute total relation matrix
4. Calculate prominence (D+R) and net influence (D-R) vectors
5. Identify cause and effect factors through causal diagram

### House of Quality Integration
1. Map customer requirements to technical characteristics
2. Define relationship matrix between requirements and characteristics
3. Calculate technical importance based on customer importance and relationships
4. Apply DEMATEL results to adjust importance weights

## Troubleshooting

### Common Issues

#### 1. Installation Issues
- If getting an error with pyDEMATEL installation, try:
  ```bash
  pip install --upgrade pip
  pip install pyDEMATEL
  ```

#### 2. Docker Issues
- If Docker build fails, ensure Docker Desktop is running
- For permission issues, try running with sudo (Linux)

#### 3. UI Issues
- If getting matplotlib deprecation warnings, they are non-critical
- For "too many values to unpack" errors, ensure input matrices are properly formatted

#### 4. DEMATEL Analysis
- Ensure your direct relation matrix is square (same number of rows and columns)
- Matrix values should be non-negative
- Diagonal elements are typically zeros

#### 5. HOQ Integration
- Ensure the number of DEMATEL factors matches either customer requirements or technical characteristics
- Relationship strengths should be between 0-9 scale

## TODO

- Enable dynamic TFN mapping value via UI
- Enable dynamic formula for defuzzified values calculation
- Add more MCDM methods (ANP, TOPSIS, VIKOR)
- Implement data validation and error recovery
- Add automated tests for all modules
- Improve visualization options for all methods
- Add export options for various formats (PDF, LaTeX)
- Implement user authentication for multi-user scenarios
- Add real-time collaboration features
- Add advanced statistical analysis features

## Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT
