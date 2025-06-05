# Fuzzy Delphi TFN Calculator

A specialized data processing tool designed to help with Fuzzy Delphi Method calculations for thesis research data analysis.

## Overview

This project was specifically created to help my wife process data for her thesis research abouut the implementation of Sustainability Principles to support the performance of Small and Medium Enterprises in Makassar. Her research utilizes the Fuzzy Delphi Method with Triangular Fuzzy Numbers (TFN) to analyze survey data from SMEs in Makassar regarding sustainability practices. The tool provides both a user-friendly web interface and a command-line script for processing this specialized survey data.

For correspondence regarding the detail of the thesis topic and its methodology/approach, please contact her at nurastrimufthias@gmail.com.

## Features

- Conversion of Likert scale scores (1-4) to Triangular Fuzzy Numbers (TFN)
- Calculation of mean TFN for each indicator
- Computation of consensus distances for each respondent
- Defuzzification of TFN values using the formula: A = (l + 2m + 2u) / 4
- Summary statistics and visualization
- Export options for calculation details and summary results

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

## Usage

### Web Interface (UI Version)

The UI version provides a graphical interface for interactive data processing using Streamlit:

```bash
streamlit run app.py
```

This will launch the application in your default web browser where you can:

1. Upload a CSV file with survey data
2. View the raw data and TFN conversion results
3. See detailed calculations for each indicator
4. Download calculation details and summary results

### Command-Line Script Version

For batch processing or automation, use the script version:

```bash
python fuzzy_script.py input_data.csv
```

The script will:
1. Process the input CSV file
2. Display calculation steps in the terminal
3. Generate a summary CSV file with results

### Input Data Format

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

## TODO

- Enable dynamic TFN mapping value via UI
- Enable dynamic formula for defuzzified values calculation

## License

MIT
