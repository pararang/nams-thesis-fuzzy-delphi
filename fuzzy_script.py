import pandas as pd
import numpy as np

# 1. Define the TFN mapping based on the spreadsheet sample (Image 2)
def skor_to_tfn(skor):
    mapping = {
        1: (0,    0,    0.25),
        2: (0,    0.25, 0.5 ),
        3: (0.25, 0.5,  0.75),
        4: (0.5,  0.75, 1.0 )
    }
    return mapping[int(skor)]

# 2. Convert scores to TFNs and compute mean TFN for an indicator
def indicator_tfns(scores):
    tfns = [skor_to_tfn(s) for s in scores]
    l = np.mean([t[0] for t in tfns])
    m = np.mean([t[1] for t in tfns])
    u = np.mean([t[2] for t in tfns])
    return tfns, (l, m, u)

# 3. Consensus distance d for each respondent (distance from mean TFN)
def consensus_distances(tfns, mean_tfn):
    dists = []
    for tfn in tfns:
        d = np.sqrt(((tfn[0] - mean_tfn[0])**2 + (tfn[1] - mean_tfn[1])**2 + (tfn[2] - mean_tfn[2])**2) / 3)
        dists.append(d)
    return dists

# 4. Defuzzification formula as per spreadsheet
def defuzzify_tfn(mean_tfn):
    # A = (l + 2*m + u) / 4
    return (mean_tfn[0] + 2 * mean_tfn[1] + mean_tfn[2]) / 4

def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python fuzzy_delphi_from_csv_spreadsheet_sample.py <input.csv>")
        sys.exit(1)

    csv_file = sys.argv[1]
    df = pd.read_csv(csv_file)

    # Assume first column is respondent, the rest are indicators
    indicator_names = df.columns[1:]
    results = []

    print("Fuzzy Delphi Calculation (Spreadsheet Sample TFN mapping)\n")
    for indikator in indicator_names:
        print(f"\n--- Indicator {indikator} ---")
        scores = df[indikator].tolist()
        respondents = df.iloc[:,0].tolist()

        # Step 2: Convert and average
        tfns, mean_tfn = indicator_tfns(scores)
        print("Respondent & TFN:")
        for r, s, tfn in zip(respondents, scores, tfns):
            print(f"  {r}: Score={s} → TFN={tfn}")
        print(f"Mean TFN: {mean_tfn}")

        # Step 3: Consensus d
        dists = consensus_distances(tfns, mean_tfn)
        for r, d in zip(respondents, dists):
            print(f"  Consensus distance {r}: {d:.4f}")

        # Step 4: Defuzzification
        crisp = defuzzify_tfn(mean_tfn)
        print(f"Defuzzification (A): {crisp:.4f}")

        results.append({
            "Indicator": indikator,
            "Mean_TFN": mean_tfn,
            "Defuzzified": crisp,
            "Consensus_d": dists
        })

    # Optionally, save summary to CSV
    summary = pd.DataFrame({
        "Indicator": [r["Indicator"] for r in results],
        "Mean_TFN_L": [r["Mean_TFN"][0] for r in results],
        "Mean_TFN_M": [r["Mean_TFN"][1] for r in results],
        "Mean_TFN_U": [r["Mean_TFN"][2] for r in results],
        "Defuzzified": [r["Defuzzified"] for r in results]
    })
    summary.to_csv("fuzzy_delphi_summary.csv", index=False)
    print("\nSummary saved to fuzzy_delphi_summary.csv")

if __name__ == "__main__":
    main()
