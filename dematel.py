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

                    # Visualization section
                    st.subheader("DEMATEL Visualizations")
                    viz_col1, viz_col2 = st.columns(2)
                    
                    # Causal diagram (D+R vs D-R)
                    with viz_col1:
                        st.write("**Causal Diagram**")
                        fig1, ax1 = plt.subplots(figsize=(8, 6))
                        
                        # Plot D+R vs D-R
                        x = D_minus_R_rounded  # D-R (Net Influence)
                        y = D_plus_R_rounded   # D+R (Prominence)
                        
                        scatter = ax1.scatter(x, y, alpha=0.7, s=100)
                        
                        # Add labels for each point
                        for i, txt in enumerate(df.index):
                            ax1.annotate(txt, (x[i], y[i]), fontsize=9, ha='right')
                        
                        ax1.set_xlabel('D-R (Net Influence)')
                        ax1.set_ylabel('D+R (Prominence)')
                        ax1.set_title('Causal Diagram: Influence vs Prominence')
                        ax1.grid(True, linestyle='--', alpha=0.6)
                        
                        # Draw quadrant lines
                        avg_x = np.mean(x) if len(x) > 0 else 0
                        avg_y = np.mean(y) if len(y) > 0 else 0
                        ax1.axhline(y=avg_y, color='r', linestyle='--', alpha=0.5, label='Avg Prominence')
                        ax1.axvline(x=avg_x, color='r', linestyle='--', alpha=0.5, label='Avg Net Influence')
                        
                        ax1.legend()
                        st.pyplot(fig1, clear_figure=False)
                        plt.close(fig1)
                    
                    # Influence bars
                    with viz_col2:
                        st.write("**Influence vs Impact Analysis**")
                        fig2, ax2 = plt.subplots(figsize=(8, 6))
                        
                        x_pos = np.arange(len(df.index))
                        width = 0.35
                        
                        ax2.bar(x_pos - width/2, D_rounded, width, label='D (Total Influence)', alpha=0.8)
                        ax2.bar(x_pos + width/2, R_rounded, width, label='R (Total Impact)', alpha=0.8)
                        
                        ax2.set_xlabel('Factors')
                        ax2.set_ylabel('Values')
                        ax2.set_title('Total Influence vs Total Impact')
                        ax2.set_xticks(x_pos)
                        ax2.set_xticklabels(df.index, rotation=45, ha='right')
                        ax2.legend()
                        ax2.grid(True, linestyle='--', alpha=0.6)
                        
                        st.pyplot(fig2, clear_figure=False)
                        plt.close(fig2)
                    
                    # Net influence and prominence bars
                    st.subheader("Factor Classification")
                    viz_col3, viz_col4 = st.columns(2)
                    
                    with viz_col3:
                        st.write("**Net Influence (D-R)**")
                        fig3, ax3 = plt.subplots(figsize=(8, 6))
                        
                        colors = ['red' if val > 0 else 'blue' for val in D_minus_R_rounded]
                        bars = ax3.bar(df.index, D_minus_R_rounded, color=colors, alpha=0.7)
                        
                        ax3.set_xlabel('Factors')
                        ax3.set_ylabel('Net Influence (D-R)')
                        ax3.set_title('Net Influence: Cause (Positive) vs Effect (Negative)')
                        ax3.grid(True, linestyle='--', alpha=0.6)
                        plt.xticks(rotation=45, ha='right')
                        
                        # Add horizontal line at y=0
                        ax3.axhline(y=0, color='black', linewidth=0.8)
                        
                        st.pyplot(fig3, clear_figure=False)
                        plt.close(fig3)
                    
                    with viz_col4:
                        st.write("**Prominence (D+R)**")
                        fig4, ax4 = plt.subplots(figsize=(8, 6))
                        
                        bars = ax4.bar(df.index, D_plus_R_rounded, color='green', alpha=0.7)
                        
                        ax4.set_xlabel('Factors')
                        ax4.set_ylabel('Prominence (D+R)')
                        ax4.set_title('Factor Prominence')
                        ax4.grid(True, linestyle='--', alpha=0.6)
                        plt.xticks(rotation=45, ha='right')
                        
                        st.pyplot(fig4, clear_figure=False)
                        plt.close(fig4)
                    
                    # Matrix heatmap visualization
                    st.subheader("Matrix Heatmaps")
                    matrix_cols = st.columns(2)
                    
                    with matrix_cols[0]:
                        st.write("**Normalized Matrix (T)**")
                        fig5, ax5 = plt.subplots(figsize=(10, 8))
                        im = ax5.imshow(T_rounded, cmap='viridis', aspect='auto', interpolation='nearest')
                        ax5.set_xticks(range(len(df.columns)))
                        ax5.set_yticks(range(len(df.index)))
                        ax5.set_xticklabels(df.columns, rotation=45, ha='right')
                        ax5.set_yticklabels(df.index)
                        ax5.set_title("Normalized Matrix Heatmap")
                        
                        # Add colorbar
                        cbar = plt.colorbar(im, ax=ax5)
                        cbar.set_label('Value')
                        
                        # Add value annotations for smaller matrices
                        if T_rounded.shape[0] <= 10 and T_rounded.shape[1] <= 10:
                            for i in range(T_rounded.shape[0]):
                                for j in range(T_rounded.shape[1]):
                                    text = ax5.text(j, i, f"{T_rounded[i, j]:.2f}",
                                                   ha="center", va="center", color="white", fontsize=8)
                        
                        st.pyplot(fig5, clear_figure=False)
                        plt.close(fig5)
                    
                    with matrix_cols[1]:
                        st.write("**Total Relation Matrix (T*)**")
                        fig6, ax6 = plt.subplots(figsize=(10, 8))
                        im = ax6.imshow(T_star_rounded, cmap='plasma', aspect='auto', interpolation='nearest')
                        ax6.set_xticks(range(len(df.columns)))
                        ax6.set_yticks(range(len(df.index)))
                        ax6.set_xticklabels(df.columns, rotation=45, ha='right')
                        ax6.set_yticklabels(df.index)
                        ax6.set_title("Total Relation Matrix Heatmap")
                        
                        # Add colorbar
                        cbar = plt.colorbar(im, ax=ax6)
                        cbar.set_label('Value')
                        
                        # Add value annotations for smaller matrices
                        if T_star_rounded.shape[0] <= 10 and T_star_rounded.shape[1] <= 10:
                            for i in range(T_star_rounded.shape[0]):
                                for j in range(T_star_rounded.shape[1]):
                                    text = ax6.text(j, i, f"{T_star_rounded[i, j]:.2f}",
                                                   ha="center", va="center", color="white", fontsize=8)
                        
                        st.pyplot(fig6, clear_figure=False)
                        plt.close(fig6)
                    
                    # Additional visualization using DEMATEL solver's drawCurve if available
                    try:
                        st.subheader("DEMATEL Solver Visualization")
                        fig = solver.drawCurve()
                        if fig is not None:
                            st.pyplot(fig, clear_figure=False)
                    except Exception as e:
                        st.info(f"Solver visualization not available: {str(e)}")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted with space delimiters.")

    # Footer
    # st.markdown("---")


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


