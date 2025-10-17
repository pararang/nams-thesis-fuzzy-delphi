import streamlit as st
import pandas as pd
import datetime
from io import BytesIO
from fdm_tfn import (fdm_form)
from dematel import (dematel_form)


def main():
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

    fdmTab, dematelTab = st.tabs(["Fuzzy Delphi TFN", "Dematel Analysis"])
    with fdmTab:
        fdm_form()
    with dematelTab:
        dematel_form()
            
if __name__ == "__main__":
    main()