import streamlit as st
import pandas as pd
import numpy as np
import datetime
from io import BytesIO



st.title("Fuzzy Delphi TFN Calculator")

st.markdown("""
**Instructions:**
1. Upload a CSV file with the first column as respondent code or name, and the next columns as indicator scores (1-4). First row as table headers. See the example here: https://github.com/pararang/nams-thesis/blob/main/sample_input.csv
2. Skor tfn is mapped as follows:
- 1 → (0, 0, 0.25)
- 2 → (0, 0.25, 0.5)
- 3 → (0.25, 0.5, 0.75)
- 4 → (0.5, 0.75, 1.0)
3. The application will calculate the mean TFN for each indicator, consensus distances, and defuzzified values.
4. Defuzzified values are calculated using the formula:
**A = (l + 2m + 2u) / 4**
5. Results will be displayed and can be downloaded as a CSV file.
""")

# Score to TFN mapping as in your spreadsheet sample
def scor_to_tfn(skor):
    mapping = {
        1: (0,    0,    0.25),
        2: (0,    0.25, 0.5 ),
        3: (0.25, 0.5,  0.75),
        4: (0.5,  0.75, 1.0 )
    }
    
    try:
        skor_int = int(skor)
        if skor_int not in mapping:
            raise ValueError(f"Score {skor} is out of bounds (must be 1-4)")
        return mapping[skor_int]
    except Exception as e:
        st.warning(f"Invalid score '{skor}': {e}")
        return (np.nan, np.nan, np.nan)

def indicator_tfns(scores):
    tfns = [scor_to_tfn(s) for s in scores]
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
    return (mean_tfn[0] + (2 * mean_tfn[1]) + (mean_tfn[2] * 2)) / 4

uploaded_file = st.file_uploader("Upload your input CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data")
    st.dataframe(df)

    indicator_names = df.columns[1:]
    respondents = df.iloc[:, 0].tolist()

    results = []
    calculation_sheets = {}

    st.write("## Results and Calculation Steps")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
        calculation_sheets[indikator] = tfn_table
        # st.download_button(
        #     label="Download as CSV",
        #     data=tfn_table.to_csv(index=False),
        #     file_name= f"fuzzy_delp_tfn_{timestamp}_{indikator}.csv",
        #     mime="text/csv"
        # )

        # buffer = BytesIO()
        # writer = pd.ExcelWriter(buffer, engine='xlsxwriter')
        
        # st.download_button(
        #     label="Download as Excel",
        #     data=tfn_table.to_excel(writer, sheet_name=indikator, index=False),
        #     file_name= f"fuzzy_delp_tfn_{timestamp}_{indikator}.xlsx",
        #     mime="application/vnd.ms-excel"
        # )

        # buffer = BytesIO()
        # with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        #     tfn_table.to_excel(writer, sheet_name=indikator, index=False)

        # st.download_button(
        #     label="Download Excel", 
        #     data=buffer, #buffer.getvalue(), 
        #     file_name=f"fuzzy_delp_tfn_{timestamp}_{indikator}.xlsx", 
        #     mime="application/vnd.ms-excel"
        # )


        # st.download_button(
        #     label="Download Excel", 
        #     data=buffer.getvalue(), 
        #     file_name=f"fuzzy_delp_tfn2_{timestamp}_{indikator}.xlsx", 
        #     mime="application/vnd.ms-excel"
        # )


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
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        for indikator, table in calculation_sheets.items():
            table.to_excel(writer, sheet_name=indikator, index=False)
        
        writer.close()

        st.download_button(
            label="Download All Calculation in Excel", 
            data=buffer, #buffer.getvalue(), 
            file_name=f"fuzzy_delp_tfn_calculation_{timestamp}.xlsx", 
            mime="application/vnd.ms-excel"
        )

    st.write("---")
    st.write("## Summary Table")
    summary_df = pd.DataFrame(results)
    st.dataframe(summary_df)
    st.download_button(
        label="Download Summary as CSV",
        data=summary_df.to_csv(index=False),
        file_name=f"fuzzy_delp_tfn_summary_{timestamp}.csv",
        mime="text/csv"
    )
