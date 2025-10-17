"""
DEMATEL (Decision Making Trial and Evaluation Laboratory) computation module.

This module contains the core mathematical functions for DEMATEL analysis,
including matrix normalization and total relation matrix computation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import streamlit as st


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

                    # Normalize matrix
                    T = normalize_matrix(D)
                    T_rounded = round_matrix_values(T)

                    # Compute total relation matrix
                    T_star = compute_total_relation_matrix(T)
                    T_star_rounded = round_matrix_values(T_star)

                    # Calculate influence vectors
                    D_vector, R_vector = calculate_influence_vectors(T_star)
                    D_plus_R, D_minus_R = calculate_cause_effect_analysis(D_vector, R_vector)

                    # Round vectors to 3 decimal places
                    D_rounded = round_matrix_values(D_vector, 3)
                    R_rounded = round_matrix_values(R_vector, 3)
                    D_plus_R_rounded = round_matrix_values(D_plus_R, 3)
                    D_minus_R_rounded = round_matrix_values(D_minus_R, 3)

                    # Display results
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Normalized Matrix (T)")
                        st.dataframe(pd.DataFrame(
                            T_rounded,
                            index=df.index,
                            columns=df.columns
                        ))

                    with col2:
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

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted with space delimiters.")

    # Footer
    st.markdown("---")
    st.markdown("*DEMATEL: Decision Making Trial and Evaluation Laboratory*")


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


def normalize_matrix(D: np.ndarray) -> np.ndarray:
    """
    Compute the normalized direct-relation matrix T from direct matrix D.

    Formula: T = D / max(∑D_ij) where max is the largest row sum.

    Args:
        D: Direct relation matrix (n x n)

    Returns:
        Normalized matrix T (n x n)
    """
    # Calculate row sums
    row_sums = np.sum(D, axis=1)

    # Find the maximum row sum
    max_sum = np.max(row_sums)

    # Avoid division by zero
    if max_sum == 0:
        raise ValueError("Cannot normalize matrix: all row sums are zero")

    # Normalize the matrix
    T = D / max_sum

    return T


def compute_total_relation_matrix(T: np.ndarray) -> np.ndarray:
    """
    Compute the total relation matrix T* using DEMATEL formula.

    Formula: T* = T × (I - T)^(-1)

    Args:
        T: Normalized direct-relation matrix (n x n)

    Returns:
        Total relation matrix T* (n x n)
    """
    n = T.shape[0]

    # Create identity matrix
    I = np.eye(n)

    # Compute (I - T)
    I_minus_T = I - T

    # Check if matrix is invertible
    if np.linalg.det(I_minus_T) == 0:
        raise ValueError("Matrix (I - T) is singular and cannot be inverted")

    # Compute inverse of (I - T)
    I_minus_T_inv = np.linalg.inv(I_minus_T)

    # Compute total relation matrix: T* = T × (I - T)^(-1)
    T_star = np.dot(T, I_minus_T_inv)

    return T_star


def load_matrix_from_csv(file_path: str, delimiter: str = ',') -> pd.DataFrame:
    """
    Load and parse a DEMATEL matrix from CSV file.

    Args:
        file_path: Path to the CSV file
        delimiter: CSV delimiter (default: comma)

    Returns:
        DataFrame with the loaded matrix
    """
    try:
        # Read CSV with comma delimiter, first column as index
        df = pd.read_csv(file_path, index_col=0)

        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()

        return df
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {str(e)}")


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


def calculate_influence_vectors(T_star: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Total Influence Vector (D) and Total Impact Vector (R) from T*.

    D (Total Influence): Sum of each row in T* - how much each factor influences others
    R (Total Impact): Sum of each column in T* - how much each factor is influenced by others

    Args:
        T_star: Total relation matrix (n x n)

    Returns:
        Tuple of (D_vector, R_vector) where each is a 1D array of length n
    """
    # D: Sum of rows (total influence of each factor)
    D = np.sum(T_star, axis=1)

    # R: Sum of columns (total impact on each factor)
    R = np.sum(T_star, axis=0)

    return D, R


def calculate_cause_effect_analysis(D: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate D+R and D-R for cause-effect analysis.

    D+R: Total influence + total impact
    D-R: Net influence (positive = cause factor, negative = effect factor)

    Args:
        D: Total influence vector
        R: Total impact vector

    Returns:
        Tuple of (D_plus_R, D_minus_R)
    """
    D_plus_R = D + R
    D_minus_R = D - R

    return D_plus_R, D_minus_R