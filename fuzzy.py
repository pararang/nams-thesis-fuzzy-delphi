import csv
import sys
import numpy as np

# Mapping Likert scale to TFN [l, m, u]
def likert_to_tfn(score):
    mapping = {
        1: (1, 1, 2),
        2: (1, 2, 3),
        3: (2, 3, 4),
        4: (3, 4, 4)
    }
    return mapping[int(score)]

# Read the CSV and structure the data
def read_csv(filename):
    indicators = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
        num_indicators = len(rows[0]) - 1  # exclude respondent code
        for i in range(1, num_indicators + 1):
            col = [row[i] for row in rows]
            indicators.append(col)
    return indicators

# Aggregate TFNs for one indicator
def aggregate_tfns(tfns):
    l = np.mean([t[0] for t in tfns])
    m = np.mean([t[1] for t in tfns])
    u = np.mean([t[2] for t in tfns])
    return (l, m, u)

def defuzzify_tfn(tfn):
    return (tfn[0] + tfn[1] + tfn[2]) / 3

def main():
    if len(sys.argv) < 2:
        print("Usage: python fuzzy_delphi_from_csv.py <input.csv>")
        sys.exit(1)

    filename = sys.argv[1]
    indicators = read_csv(filename)
    result = []
    for idx, scores in enumerate(indicators, 1):
        tfns = [likert_to_tfn(score) for score in scores]
        agg_tfn = aggregate_tfns(tfns)
        crisp = defuzzify_tfn(agg_tfn)
        print(f"Indicator {idx}: Aggregated TFN = {agg_tfn}, Defuzzified = {crisp:.3f}")
        result.append((agg_tfn, crisp))

if __name__ == "__main__":
    main()
