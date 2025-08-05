import streamlit as st
import pandas as pd
import numpy as np
import datetime
from io import BytesIO



# Floating QR code donation button (bottom left)
st.markdown(
    """
    <style>
    .donate-qr-fixed {
        position: fixed;
        left: 20px;
        bottom: 20px;
        z-index: 9999;
        background: rgba(255,255,255,0.95);
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        padding: 10px 10px 2px 10px;
        text-align: center;
        width: 160px;
    }
    .donate-qr-fixed img {
        width: 120px;
        height: 120px;
        margin-bottom: 4px;
    }
    .donate-qr-fixed .donate-label {
        font-size: 0.95em;
        color: #333;
        font-weight: 500;
        margin-bottom: 2px;
    }
    </style>
    <div class=\"donate-qr-fixed\">
        <div class=\"donate-label\">Buy me a coffee</div>
        <a href=\"https://saweria.co/pararang\" target=\"_blank\" rel=\"noopener\">
            <img src=\"https://api.qrserver.com/v1/create-qr-code/?size=640x640&data=https://saweria.co/pararang\" alt=\"Donate QR Code\" />
        </a>
        <div style=\"font-size:0.6em;color:#888;\">https://saweria.co/pararang</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Fuzzy Delphi TFN Calculator")

st.markdown("""
**Instructions:**
1. Upload a CSV file with the first column as respondent code or name, and the next columns as indicator scores (1-4). First row as table headers. See the example [here](https://github.com/pararang/nams-thesis/blob/main/sample_input.csv).
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

    skor_int = int(skor)
    if skor_int not in mapping:
        raise ValueError(f"Score {skor} is out of bounds (must be 1-4)")
    return mapping[skor_int]

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
    st.write("## Raw Data")
    st.dataframe(df)

    indicator_names = df.columns[1:]
    respondents = df.iloc[:, 0].tolist()

    results = []
    calculation_sheets = {}

    st.write("## TFN Conversion & Consensus Distances Calculation per Indicator")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    err_bags = []
    for indikator in indicator_names:
        scores = df[indikator].tolist()
        try:
            tfns, mean_tfn = indicator_tfns(scores)
            dists = consensus_distances(tfns, mean_tfn)
            crisp = defuzzify_tfn(mean_tfn)
        except Exception as e:
            err_bags.append(f"Error processing indicator '{indikator}': {e}")
            continue
        
        # always suudzon, if there is an error, assuming there is other error on other indicators
        # so, collect all errors and show them at the end then user can fix them all at once istead of one by one each time user find error
        if err_bags:
            continue

        st.write(f"---\n### {indikator}")
        # Show TFN conversion for each respondent
        tfn_table = pd.DataFrame({
            "Respondent": respondents,
            "Score": scores,
            "L": [t[0] for t in tfns],
            "M": [t[1] for t in tfns],
            "U": [t[2] for t in tfns],
            "Consensus d": dists
        })
        st.dataframe(tfn_table)
        calculation_sheets[indikator] = tfn_table

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

    if err_bags:
        st.write("### Errors")
        for err in err_bags:
            st.error(err)
    else:
        st.write("---")
        st.write("## Summary")
        summary_df = pd.DataFrame(results)
        st.dataframe(summary_df)

        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            for indikator, table in calculation_sheets.items():
                table.to_excel(writer, sheet_name=indikator, index=False)
        
        st.write("---")
            
        st.download_button(
            label="Download Results in Single Excel File", 
            data=buffer, #buffer.getvalue(), 
            file_name=f"fuzzy_delp_tfn_calculation_{timestamp}.xlsx", 
            mime="application/vnd.ms-excel",
            # type="primary",
            icon=":material/download:",
            use_container_width=True,
        )
    
