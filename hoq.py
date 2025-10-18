"""
House of Quality (HOQ) integration with DEMATEL analysis.

This module integrates DEMATEL (Decision Making Trial and Evaluation Laboratory)
with House of Quality methodology to provide a comprehensive decision-making framework.
DEMATEL provides causal relationships between factors, while HOQ helps prioritize
customer requirements against technical characteristics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional
import streamlit as st
from pyDEMATEL.DEMATELSolver import DEMATELSolver


class HOQIntegrator:
    """
    Class to integrate DEMATEL with House of Quality analysis.
    """
    
    def __init__(self):
        self.dematel_solver = None
        self.customer_requirements = []
        self.technical_characteristics = []
        self.relationship_matrix = None
        self.importance_weights = None
        self.hoq_matrix = None
    
    def perform_dematel_analysis(self, direct_relation_matrix: np.ndarray, 
                                factors: List[str]) -> Dict:
        """
        Perform DEMATEL analysis to identify causal relationships.
        
        Args:
            direct_relation_matrix: Square matrix of direct relationships
            factors: List of factor names
            
        Returns:
            Dictionary containing DEMATEL results
        """
        solver = DEMATELSolver()
        solver.setMatrix([direct_relation_matrix])
        solver.addExpert("user")
        solver.setFactors(factors)
        solver.setNumberOfExperts(1)
        solver.setNumberOfFactors(len(factors))
        
        solver.step1()
        solver.step2()
        solver.step3()
        solver.step4()
        
        # Safe unpacking of results
        prominence_result = solver.getProminence()
        relation_result = solver.getRalation()
        
        if isinstance(prominence_result, (list, tuple)) and len(prominence_result) >= 2:
            D_vector, R_vector = prominence_result[0], prominence_result[1]
        else:
            D_vector = R_vector = prominence_result if isinstance(prominence_result, (list, np.ndarray)) else [prominence_result]
        
        if isinstance(relation_result, (list, tuple)) and len(relation_result) >= 2:
            D_plus_R, D_minus_R = relation_result[0], relation_result[1]
        else:
            D_plus_R = D_minus_R = relation_result if isinstance(relation_result, (list, np.ndarray)) else [relation_result]
        
        return {
            'solver': solver,
            'prominence_D': D_vector,
            'prominence_R': R_vector,
            'relation_plus': D_plus_R,
            'relation_minus': D_minus_R,
            'total_influence_matrix': solver.getTotalInfluenceMatrix()
        }
    
    def create_hoq_matrix(self, 
                          customer_requirements: List[str],
                          technical_characteristics: List[str],
                          relationship_matrix: np.ndarray,
                          customer_importance: np.ndarray,
                          technical_correlation: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Create House of Quality matrix integrating with DEMATEL results.
        
        Args:
            customer_requirements: List of customer requirements
            technical_characteristics: List of technical characteristics
            relationship_matrix: Matrix (CR x TC) showing relationship between customer requirements and technical characteristics
            customer_importance: Array of importance weights for customer requirements
            technical_correlation: Optional correlation matrix between technical characteristics
            
        Returns:
            DataFrame representing the HOQ matrix
        """
        self.customer_requirements = customer_requirements
        self.technical_characteristics = technical_characteristics
        
        # Calculate technical importance based on relationship matrix and customer importance
        technical_importance = relationship_matrix.T @ customer_importance
        
        # Create the HOQ matrix
        hoq_data = {
            'Technical Characteristic': technical_characteristics,
            'Technical Importance': technical_importance
        }
        
        # Add customer requirement importances
        for i, cr in enumerate(customer_requirements):
            hoq_data[f'CR-{cr}'] = relationship_matrix[i, :]
            hoq_data[f'Weight-{cr}'] = customer_importance[i]
        
        if technical_correlation is not None:
            for i, tc_row in enumerate(technical_characteristics):
                hoq_data[f'Correlation-{tc_row}'] = technical_correlation[i, :]
        
        hoq_df = pd.DataFrame(hoq_data)
        self.hoq_matrix = hoq_df
        
        return hoq_df
    
    def integrate_dematel_hoq(self, 
                              dematel_results: Dict,
                              customer_requirements: List[str],
                              technical_characteristics: List[str],
                              relationship_matrix: np.ndarray,
                              customer_importance: np.ndarray) -> Dict:
        """
        Integrate DEMATEL results with HOQ analysis.
        
        Args:
            dematel_results: Results from DEMATEL analysis
            customer_requirements: List of customer requirements
            technical_characteristics: List of technical characteristics
            relationship_matrix: Matrix showing relationship between CR and TC
            customer_importance: Array of importance weights for customer requirements
            
        Returns:
            Dictionary containing integrated analysis results
        """
        try:
            # Use DEMATEL prominence values to adjust importance weights
            dematel_importance = dematel_results.get('relation_plus', [])  # Combined influence values
            
            # Create the basic HOQ matrix first
            hoq_matrix = self.create_hoq_matrix(
                customer_requirements, 
                technical_characteristics, 
                relationship_matrix, 
                customer_importance
            )
            
            # Create the return dictionary with basic HOQ results
            result = {
                'hoq_matrix': hoq_matrix,
                'dematel_results': dematel_results
            }
            
            # Only try to adjust if we have matching lengths
            if len(dematel_importance) > 0:
                if len(dematel_importance) == len(customer_requirements):
                    # DEMATEL applied to customer requirements
                    adjusted_cr_importance = customer_importance * (1 + np.array(dematel_importance)/10)  # Normalize influence
                    result['adjusted_customer_importance'] = adjusted_cr_importance
                elif len(dematel_importance) == len(technical_characteristics):
                    # DEMATEL applied to technical characteristics
                    base_tech_importance = relationship_matrix.T @ customer_importance
                    adjusted_tech_importance = base_tech_importance * (1 + np.array(dematel_importance)/10)
                    
                    # Update the HOQ matrix with adjusted values
                    hoq_matrix['Technical Importance (Adjusted)'] = adjusted_tech_importance
                    result['hoq_matrix'] = hoq_matrix  # Update with adjusted matrix
                    result['adjusted_technical_importance'] = adjusted_tech_importance
            
            return result
        except Exception as e:
            # Ensure we always return a valid dictionary with at least the HOQ matrix
            hoq_matrix = self.create_hoq_matrix(
                customer_requirements, 
                technical_characteristics, 
                relationship_matrix, 
                customer_importance
            )
            
            return {
                'hoq_matrix': hoq_matrix,
                'dematel_results': dematel_results,
                'error': str(e)
            }


def hoq_form():
    """Main form for HOQ and HOQ-DEMATEL integrated analysis."""
    st.title("HOQ Analysis")
    st.markdown("""
    This tool provides two analysis options:
    1. Standalone House of Quality (HOQ) analysis
    2. Integrated HOQ-DEMATEL analysis that combines causal relationships with quality planning
    """)
    
    # Selection for analysis type
    analysis_type = st.radio(
        "Select Analysis Type",
        ["Standalone HOQ", "HOQ with DEMATEL Integration"],
        index=0
    )
    
    if analysis_type == "Standalone HOQ":
        st.header("Standalone House of Quality Analysis")
        
        # Customer Requirements Input
        st.subheader("Customer Requirements")
        cr_count = st.number_input("Number of Customer Requirements", min_value=1, max_value=20, value=4, key='cr_count')
        customer_requirements = []
        for i in range(int(cr_count)):
            req = st.text_input(f"Customer Requirement {i+1}", value=f"CR{i+1}", key=f'cr_{i}_standalone')
            customer_requirements.append(req)
        
        # Technical Characteristics Input
        st.subheader("Technical Characteristics")
        tc_count = st.number_input("Number of Technical Characteristics", min_value=1, max_value=20, value=4, key='tc_count')
        technical_characteristics = []
        for i in range(int(tc_count)):
            tc = st.text_input(f"Technical Characteristic {i+1}", value=f"TC{i+1}", key=f'tc_{i}_standalone')
            technical_characteristics.append(tc)
        
        # Relationship Matrix Input
        st.subheader("Relationship Matrix (Customer Requirements vs Technical Characteristics)")
        st.markdown("Define the relationship strength between each CR and TC (0-9 scale):")
        
        relationship_matrix = np.zeros((int(cr_count), int(tc_count)))
        for i, cr in enumerate(customer_requirements):
            cols = st.columns(len(technical_characteristics))
            for j, tc in enumerate(technical_characteristics):
                relationship_matrix[i][j] = cols[j].number_input(
                    f"{cr} vs {tc}",
                    min_value=0.0,
                    max_value=9.0,
                    value=0.0,
                    step=0.5,
                    key=f'rel_{i}_{j}_standalone'
                )
        
        # Customer Importance Input
        st.subheader("Customer Requirement Importances")
        customer_importance = np.zeros(int(cr_count))
        for i, cr in enumerate(customer_requirements):
            customer_importance[i] = st.number_input(
                f"Importance of {cr}",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                key=f'cr_imp_{i}_standalone'
            )
        
        # Technical Correlation Matrix (Optional)
        st.subheader("Technical Correlation Matrix (Optional)")
        st.markdown("Define correlation between technical characteristics (0-9 scale):")
        add_correlation = st.checkbox("Add technical correlation matrix", value=False)
        
        technical_correlation = None
        if add_correlation:
            technical_correlation = np.zeros((int(tc_count), int(tc_count)))
            for i, tc_row in enumerate(technical_characteristics):
                cols = st.columns(len(technical_characteristics))
                for j, tc_col in enumerate(technical_characteristics):
                    if i == j:
                        # Diagonal is always 1 (perfect correlation with itself)
                        technical_correlation[i][j] = 1.0
                    else:
                        technical_correlation[i][j] = cols[j].number_input(
                            f"{tc_row} ↔ {tc_col}",
                            min_value=0.0,
                            max_value=9.0,
                            value=0.0,
                            step=0.5,
                            key=f'corr_{i}_{j}_standalone'
                        )
        
        # Run standalone HOQ analysis
        if st.button("Perform Standalone HOQ Analysis", type="primary"):
            with st.spinner("Performing HOQ analysis..."):
                integrator = HOQIntegrator()
                
                # Create HOQ matrix
                hoq_matrix = integrator.create_hoq_matrix(
                    customer_requirements,
                    technical_characteristics,
                    relationship_matrix,
                    customer_importance,
                    technical_correlation
                )
                
                # Display results
                st.subheader("House of Quality Matrix")
                st.dataframe(hoq_matrix)
                
                # Technical Importance Chart
                st.subheader("Technical Importance")
                fig, ax = plt.subplots(figsize=(10, 6))
                tech_importance = hoq_matrix['Technical Importance']
                ax.bar(technical_characteristics, tech_importance)
                ax.set_xlabel('Technical Characteristics')
                ax.set_ylabel('Importance Score')
                ax.set_title('Technical Importance based on HOQ')
                plt.xticks(rotation=45, ha='right')
                ax.grid(True, linestyle='--', alpha=0.7)
                
                st.pyplot(fig, clear_figure=False)
                
    else:  # HOQ with DEMATEL Integration
        # Tabs for different aspects of the analysis
        hoq_tab, dematel_tab, integration_tab = st.tabs(["HOQ Input", "DEMATEL Input", "Integration"])
        
        with hoq_tab:
            st.header("House of Quality Input")
            
            # Customer Requirements Input
            st.subheader("Customer Requirements")
            cr_count = st.number_input("Number of Customer Requirements", min_value=1, max_value=20, value=4, key='cr_count_integrated')
            customer_requirements = []
            for i in range(int(cr_count)):
                req = st.text_input(f"Customer Requirement {i+1}", value=f"CR{i+1}", key=f'cr_{i}_integrated')
                customer_requirements.append(req)
            
            # Technical Characteristics Input
            st.subheader("Technical Characteristics")
            tc_count = st.number_input("Number of Technical Characteristics", min_value=1, max_value=20, value=4, key='tc_count_integrated')
            technical_characteristics = []
            for i in range(int(tc_count)):
                tc = st.text_input(f"Technical Characteristic {i+1}", value=f"TC{i+1}", key=f'tc_{i}_integrated')
                technical_characteristics.append(tc)
            
            # Relationship Matrix Input
            st.subheader("Relationship Matrix (Customer Requirements vs Technical Characteristics)")
            st.markdown("Define the relationship strength between each CR and TC (0-9 scale):")
            
            relationship_matrix = np.zeros((int(cr_count), int(tc_count)))
            for i, cr in enumerate(customer_requirements):
                cols = st.columns(len(technical_characteristics))
                for j, tc in enumerate(technical_characteristics):
                    relationship_matrix[i][j] = cols[j].number_input(
                        f"{cr} vs {tc}",
                        min_value=0.0,
                        max_value=9.0,
                        value=0.0,
                        step=0.5,
                        key=f'rel_{i}_{j}_integrated'
                    )
            
            # Customer Importance Input
            st.subheader("Customer Requirement Importances")
            customer_importance = np.zeros(int(cr_count))
            for i, cr in enumerate(customer_requirements):
                customer_importance[i] = st.number_input(
                    f"Importance of {cr}",
                    min_value=0.0,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    key=f'cr_imp_{i}_integrated'
                )
        
        with dematel_tab:
            st.header("DEMATEL Input")
            
            # DEMATEL matrix input
            st.subheader("Direct Relation Matrix for DEMATEL")
            dematel_elements = st.selectbox(
                "Which elements to analyze with DEMATEL?",
                ["Customer Requirements", "Technical Characteristics", "Custom Factors"],
                key='dematel_elements'
            )
            
            if dematel_elements == "Customer Requirements":
                dematel_factors = customer_requirements
            elif dematel_elements == "Technical Characteristics":
                dematel_factors = technical_characteristics
            else:
                factor_count = st.number_input("Number of DEMATEL Factors", min_value=1, max_value=20, value=4, key='dematel_factor_count')
                dematel_factors = []
                for i in range(int(factor_count)):
                    factor = st.text_input(f"DEMATEL Factor {i+1}", value=f"Factor{i+1}", key=f'dem_factor_{i}')
                    dematel_factors.append(factor)
            
            # Create DEMATEL matrix
            dematel_matrix = np.zeros((len(dematel_factors), len(dematel_factors)))
            st.markdown(f"Define the direct relationship matrix for {len(dematel_factors)} factors:")
            
            for i, factor_row in enumerate(dematel_factors):
                cols = st.columns(len(dematel_factors))
                for j, factor_col in enumerate(dematel_factors):
                    dematel_matrix[i][j] = cols[j].number_input(
                        f"{factor_row} → {factor_col}",
                        min_value=0.0,
                        max_value=4.0,
                        value=0.0,
                        step=0.1,
                        key=f'dem_mat_{i}_{j}'
                    )
        
        with integration_tab:
            st.header("Integration Analysis")
            
            if st.button("Perform Integrated HOQ-DEMATEL Analysis", type="primary"):
                with st.spinner("Performing integrated analysis..."):
                    integrator = HOQIntegrator()
                    
                    # Perform DEMATEL analysis
                    dematel_results = integrator.perform_dematel_analysis(
                        dematel_matrix, dematel_factors
                    )
                    
                    # Perform integrated analysis
                    integration_results = integrator.integrate_dematel_hoq(
                        dematel_results,
                        customer_requirements,
                        technical_characteristics,
                        relationship_matrix,
                        customer_importance
                    )
                    
                    # Display results
                    st.subheader("DEMATEL Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Prominence Values (D+R):**")
                        prominence_df = pd.DataFrame({
                            'Factor': dematel_factors,
                            'D+R (Prominence)': dematel_results['relation_plus']
                        })
                        st.dataframe(prominence_df)
                    
                    with col2:
                        st.write("**Net Influence (D-R):**")
                        net_influence_df = pd.DataFrame({
                            'Factor': dematel_factors,
                            'D-R (Net Influence)': dematel_results['relation_minus']
                        })
                        st.dataframe(net_influence_df)
                    
                    st.subheader("House of Quality Matrix")
                    if 'hoq_matrix' in integration_results and integration_results['hoq_matrix'] is not None:
                        st.dataframe(integration_results['hoq_matrix'])
                    else:
                        st.error("HOQ matrix could not be generated. Please check your inputs.")
                    
                    # Visualization
                    st.subheader("Causal Diagram (DEMATEL Results)")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Plot prominence vs net influence
                    x = dematel_results['relation_minus']  # D-R
                    y = dematel_results['relation_plus']   # D+R
                    
                    ax.scatter(x, y, alpha=0.7)
                    
                    # Add labels
                    for i, txt in enumerate(dematel_factors):
                        ax.annotate(txt, (x[i], y[i]), fontsize=9)
                    
                    ax.set_xlabel('D-R (Net Influence)')
                    ax.set_ylabel('D+R (Prominence)')
                    ax.set_title('DEMATEL Causal Diagram')
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                    # Draw quadrant lines
                    ax.axhline(y=np.mean(y), color='r', linestyle='--', alpha=0.5)
                    ax.axvline(x=np.mean(x), color='r', linestyle='--', alpha=0.5)
                    
                    st.pyplot(fig, clear_figure=False)
                    
                    # Technical Importance Chart
                    st.subheader("Technical Importance Comparison")
                    if 'adjusted_technical_importance' in integration_results:
                        fig2, ax2 = plt.subplots(figsize=(10, 6))
                        
                        x_pos = np.arange(len(technical_characteristics))
                        width = 0.35
                        
                        base_importance = integration_results['hoq_matrix']['Technical Importance']
                        adjusted_importance = integration_results['adjusted_technical_importance']
                        
                        ax2.bar(x_pos - width/2, base_importance, width, label='Baseline', alpha=0.8)
                        ax2.bar(x_pos + width/2, adjusted_importance, width, label='DEMATEL Adjusted', alpha=0.8)
                        
                        ax2.set_xlabel('Technical Characteristics')
                        ax2.set_ylabel('Importance Score')
                        ax2.set_title('Comparison of Technical Importance: Baseline vs DEMATEL Adjusted')
                        ax2.set_xticks(x_pos)
                        ax2.set_xticklabels(technical_characteristics, rotation=45, ha='right')
                        ax2.legend()
                        ax2.grid(True, linestyle='--', alpha=0.7)
                        
                        st.pyplot(fig2, clear_figure=False)
                    else:
                        if 'hoq_matrix' in integration_results and integration_results['hoq_matrix'] is not None:
                            fig2, ax2 = plt.subplots(figsize=(10, 6))
                            tech_importance = integration_results['hoq_matrix']['Technical Importance']
                            ax2.bar(technical_characteristics, tech_importance)
                            ax2.set_xlabel('Technical Characteristics')
                            ax2.set_ylabel('Importance Score')
                            ax2.set_title('Technical Importance based on HOQ')
                            plt.xticks(rotation=45, ha='right')
                            ax2.grid(True, linestyle='--', alpha=0.7)
                            
                            st.pyplot(fig2, clear_figure=False)
    
    # Footer
    # st.markdown("---")


def demonstrate_hoq_dematel_integration():
    """
    Demonstrate the integration with a simple example.
    """
    st.write("## Example: Product Development")
    st.write("""
    This example shows how to integrate DEMATEL with House of Quality for product development:
    
    1. **DEMATEL Analysis**: Identify causal relationships between technical characteristics
    2. **HOQ**: Map customer requirements to technical characteristics
    3. **Integration**: Use DEMATEL results to adjust importance weights in HOQ
    """)
    
    # Sample data
    sample_cr = ["Reliability", "Cost", "Performance", "Ease of Use"]
    sample_tc = ["Material Strength", "Manufacturing Cost", "Processing Speed", "User Interface"]
    
    # Sample relationship matrix
    sample_relationship = np.array([
        [3, 1, 4, 2],  # Reliability vs all TCs
        [1, 4, 1, 3],  # Cost vs all TCs
        [2, 2, 4, 1],  # Performance vs all TCs
        [1, 3, 1, 4]   # Ease of Use vs all TCs
    ])
    
    # Sample customer importance
    sample_importance = np.array([4.5, 4.0, 5.0, 3.5])
    
    # Sample DEMATEL matrix for technical characteristics
    sample_dematel = np.array([
        [0, 2, 1, 3],
        [1, 0, 3, 2],
        [2, 1, 0, 2],
        [3, 2, 1, 0]
    ])
    
    if st.button("Run Sample Analysis"):
        with st.spinner("Running sample analysis..."):
            integrator = HOQIntegrator()
            
            # DEMATEL analysis
            dematel_results = integrator.perform_dematel_analysis(
                sample_dematel, sample_tc
            )
            
            # Integration
            integration_results = integrator.integrate_dematel_hoq(
                dematel_results,
                sample_cr,
                sample_tc,
                sample_relationship,
                sample_importance
            )
            
            st.subheader("Sample Results")
            st.write("**DEMATEL Results:**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Prominence (D+R):")
                for i, factor in enumerate(sample_tc):
                    st.write(f"{factor}: {dematel_results['relation_plus'][i]:.2f}")
            
            with col2:
                st.write("Net Influence (D-R):")
                for i, factor in enumerate(sample_tc):
                    st.write(f"{factor}: {dematel_results['relation_minus'][i]:.2f}")
            
            st.write("**HOQ Matrix:**")
            if 'hoq_matrix' in integration_results and integration_results['hoq_matrix'] is not None:
                st.dataframe(integration_results['hoq_matrix'])
            else:
                st.error("HOQ matrix could not be generated for the sample analysis.")


if __name__ == "__main__":
    # For testing the module directly
    print("HOQ Integration module loaded successfully")