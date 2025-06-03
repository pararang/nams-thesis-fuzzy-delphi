import streamlit as st
import pandas as pd
import numpy as np

st.title("Fuzzy Delphi TFN Calculator")

st.markdown("""
**Instructions:**
1. Upload a CSV file with the first column as respondent code or name, and the next columns as indicator scores (1-4). First row as table headers.
2. The TFN mapping and calculation steps follow the spreadsheet sample in your reference.
""")

# Score to TFN mapping as in your spreadsheet sample
def skor_to_tfn(skor):
    mapping = {
        1: (0,    0,    0.25),
        2: (0,    0.25, 0.5 ),
        3: (0.25, 0.5,  0.75),
        4: (0.5,  0.75, 1.0 )
    }
    return mapping[int(skor)]

def indicator_tfns(scores):
    tfns = [skor_to_tfn(s) for s in scores]
    l = np.mean([t[0] for t in tfns])
    m = np.mean([t[1] for t in tfns])
    u = np.mean([t[2] for t in tfns])
    return tfns, (l, m, u)

def consensus_distances(tfns, mean_tfn):
    dists = []
    for tfn in tfns:
        d = np.sqrt(((tfn[0] - mean_tfn[0])**2 + (tfn[1] - mean_tfn[1])**2 + (tfn[2] - mean_tfn[2])**2) / 3)
        dists.append(d)
    return dists

def defuzzify_tfn(mean_tfn):
    return (mean_tfn[0] + 2 * mean_tfn[1] + mean_tfn[2]) / 4

uploaded_file = st.file_uploader("Upload your input CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data")
    st.dataframe(df)

    indicator_names = df.columns[1:]
    respondents = df.iloc[:, 0].tolist()

    results = []

    st.write("## Results and Calculation Steps")
    for indikator in indicator_names:
        st.write(f"---\n### Indicator: {indikator}")
        scores = df[indikator].tolist()
        tfns, mean_tfn = indicator_tfns(scores)
        dists = consensus_distances(tfns, mean_tfn)
        crisp = defuzzify_tfn(mean_tfn)

        # Show TFN conversion for each respondent
        tfn_table = pd.DataFrame({
            "Respondent": respondents,
            "Score": scores,
            "L": [t[0] for t in tfns],
            "M": [t[1] for t in tfns],
            "U": [t[2] for t in tfns],
            "Consensus d": dists
        })
        st.write("#### TFN Conversion & Consensus Distance")
        st.dataframe(tfn_table)

        # Show mean TFN and defuzzified value
        st.write(f"**Mean TFN (L, M, U):** {tuple(round(x,4) for x in mean_tfn)}")
        st.write(f"**Defuzzified Value:** {crisp:.4f}")

        results.append({
            "Indicator": indikator,
            "Mean_TFN_L": mean_tfn[0],
            "Mean_TFN_M": mean_tfn[1],
            "Mean_TFN_U": mean_tfn[2],
            "Defuzzified": crisp
        })

    # Show summary table
    st.write("---")
    st.write("## Summary Table")
    summary_df = pd.DataFrame(results)
    st.dataframe(summary_df)
    st.download_button(
        label="Download Summary as CSV",
        data=summary_df.to_csv(index=False),
        file_name="fuzzy_delphi_summary.csv",
        mime="text/csv"
    )
