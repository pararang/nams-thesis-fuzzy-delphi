"""
DEMATEL (Decision Making Trial and Evaluation Laboratory) computation module.

This module contains the core mathematical functions for DEMATEL analysis,
including matrix normalization and total relation matrix computation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import streamlit as st
import matplotlib.pyplot as plt
from pyDEMATEL.DEMATELSolver import DEMATELSolver


def dematel_form():
    """Main application function."""
    st.title("DEMATEL Analyzer")
    st.markdown("Upload a direct relation matrix CSV file to compute the total relation matrix.")

    # File upload section
    uploaded_file = st.file_uploader(
        "Upload direct relation matrix (comma-delimited CSV)",
        type=["csv"],
        help="CSV file with comma delimiter, first row/column as labels"
    )

    if uploaded_file is not None:
        try:
            # Load and display input matrix
            df = pd.read_csv(uploaded_file, index_col=0)
            st.subheader("Input Matrix (D)")
            st.dataframe(df)

            # Validate input
            is_valid, error_msg = validate_input_matrix(df)
            if not is_valid:
                st.error(f"Invalid input matrix: {error_msg}")
                return

            # Computation section
            if st.button("Compute DEMATEL", type="primary"):
                with st.spinner("Computing DEMATEL matrices..."):
                    # Convert to numpy array
                    D = df.values.astype(float)

                     # Use pyDEMATEL solver
                    solver = DEMATELSolver()
                    solver.setMatrix([D])
                    solver.addExpert("user")
                    solver.setFactors(list(df.index))
                    solver.setNumberOfExperts(1)
                    solver.setNumberOfFactors(len(df))

                    solver.step1()  # Compute direct influence matrix
                    solver.step2()  # Normalize
                    solver.step3()  # Compute total influence matrix
                    solver.step4()  # Compute relation and prominence

                    # Get matrices
                    T = solver.getNormalizedDirectInfluenceMatrix()
                    T_star = solver.getTotalInfluenceMatrix()

                    # Get vectors
                    prominence_result = solver.getProminence()
                    relation_result = solver.getRalation()
                    
                    # Handle the unpacking more safely
                    if isinstance(prominence_result, (list, tuple)) and len(prominence_result) >= 2:
                        D_vector, R_vector = prominence_result[0], prominence_result[1]
                    else:
                        # Fallback if only one value is returned
                        D_vector = R_vector = prominence_result if isinstance(prominence_result, (list, np.ndarray)) else [prominence_result]
                    
                    if isinstance(relation_result, (list, tuple)) and len(relation_result) >= 2:
                        D_plus_R, D_minus_R = relation_result[0], relation_result[1]
                    else:
                        # Fallback if only one value is returned
                        D_plus_R = D_minus_R = relation_result if isinstance(relation_result, (list, np.ndarray)) else [relation_result]

                    # Round matrices and vectors
                    T_rounded = round_matrix_values(T)
                    T_star_rounded = round_matrix_values(T_star)
                    D_rounded = round_matrix_values(D_vector, 3)
                    R_rounded = round_matrix_values(R_vector, 3)
                    D_plus_R_rounded = round_matrix_values(D_plus_R, 3)
                    D_minus_R_rounded = round_matrix_values(D_minus_R, 3)

                    normTab, totRelTab = st.tabs(["Normalized Matrix (T)", "Total Relation Matrix (T*)"])
                    with normTab:
                        st.subheader("Normalized Matrix (T)")
                        st.dataframe(pd.DataFrame(
                            T_rounded,
                            index=df.index,
                            columns=df.columns
                        ))
                    with totRelTab:
                        st.subheader("Total Relation Matrix (T*)")
                        st.dataframe(pd.DataFrame(
                            T_star_rounded,
                            index=df.index,
                            columns=df.columns
                        ))

                    # Display influence vectors
                    st.subheader("Cause-Effect Analysis")

                    # Create DataFrame for vectors
                    vectors_df = pd.DataFrame({
                        'Factor': df.index,
                        'D (Total Influence)': D_rounded,
                        'R (Total Impact)': R_rounded,
                        'D+R': D_plus_R_rounded,
                        'D-R': D_minus_R_rounded
                    })

                    # Add interpretation
                    vectors_df['Type'] = vectors_df['D-R'].apply(
                        lambda x: 'Cause Factor' if x > 0 else 'Effect Factor'
                    )

                    st.dataframe(vectors_df.set_index('Factor'))

                    # Add explanation
                    st.markdown("""
                    **Interpretation:**
                    - **D (Total Influence)**: How much each factor influences others
                    - **R (Total Impact)**: How much each factor is influenced by others
                    - **D+R**: Total influence + total impact
                    - **D-R**: Net influence (positive = cause factor, negative = effect factor)
                    """)

                    # Download section
                    st.subheader("Download Results")

                    col3, col4 = st.columns(2)

                    with col3:
                        t_csv = pd.DataFrame(T_rounded, index=df.index, columns=df.columns).to_csv()
                        st.download_button(
                            label="Download Normalized Matrix (T)",
                            data=t_csv,
                            file_name="normalized_matrix.csv",
                            mime="text/csv"
                        )

                    with col4:
                        t_star_csv = pd.DataFrame(T_star_rounded, index=df.index, columns=df.columns).to_csv()
                        st.download_button(
                            label="Download Total Relation Matrix (T*)",
                            data=t_star_csv,
                            file_name="total_relation_matrix.csv",
                            mime="text/csv"
                        )

                    # Add visualization using DEMATEL solver's drawCurve
                    # st.subheader("DEMATEL Visualization")
                    # try:
                    #     fig = solver.drawCurve()
                    #     st.pyplot(fig)
                    # except Exception as e:
                    #     st.warning(f"Could not generate visualization: {str(e)}")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted with space delimiters.")

    # Footer
    st.markdown("---")
    st.markdown("*DEMATEL: Decision Making Trial and Evaluation Laboratory*")


def round_matrix_values(matrix: np.ndarray, decimals: int = 3) -> np.ndarray:
    """
    Round matrix values to specified decimal places.

    Args:
        matrix: Input matrix
        decimals: Number of decimal places (default: 3)

    Returns:
        Rounded matrix
    """
    return np.round(matrix, decimals)


def validate_input_matrix(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate the input matrix format for DEMATEL analysis.

    Args:
        df: Input DataFrame representing the direct relation matrix

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if DataFrame is empty
    if df.empty:
        return False, "Input matrix is empty"

    # Check if square matrix
    rows, cols = df.shape
    if rows != cols:
        return False, f"Matrix must be square. Got {rows} rows and {cols} columns"

    # Check if all values are numeric (except first row/column which are labels)
    try:
        # Convert to numeric, should work for the data portion
        numeric_data = df.values.astype(float)
    except (ValueError, TypeError):
        return False, "All matrix values must be numeric"

    # Check for negative values (DEMATEL typically uses 0-4 scale)
    if np.any(numeric_data < 0):
        return False, "Matrix contains negative values. All values should be non-negative"

    # Check if diagonal is zeros (optional but common in DEMATEL)
    diagonal = np.diag(numeric_data)
    if not np.allclose(diagonal, 0):
        # Allow small numerical errors but warn
        if np.max(np.abs(diagonal)) > 1e-10:
            # Note: Diagonal elements are not zero. This is unusual for DEMATEL matrices.
            pass  # We'll let the UI handle warnings

    return True, ""


