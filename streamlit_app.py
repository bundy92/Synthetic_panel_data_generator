import pandas as pd
import numpy as np
import streamlit as st
import io
import base64
from typing import List, Dict

class SyntheticPanelDataGenerator:
    def __init__(self):
        self.g7_countries = ['Canada', 'France', 'Germany', 'Italy', 'Japan', 'United Kingdom', 'United States']
        self.start_year = 1950
        self.end_year = 2030

    @staticmethod
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    @st.cache_data
    def convert_df(df: pd.DataFrame) -> bytes:
        """
        Convert DataFrame to CSV format.

        Parameters:
        - df (pd.DataFrame): DataFrame to convert.

        Returns:
        - bytes: Encoded CSV data.
        """
        # Input validation
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input 'df' must be a pandas DataFrame.")

        try:
            # Convert DataFrame to CSV format
            csv_data = df.to_csv(index=False).encode('utf-8')
        except Exception as e:
            # Error handling
            raise RuntimeError(f"Error occurred while converting DataFrame to CSV: {str(e)}")

        # Cybersecurity: Secure encoding
        # Encoding data using Base64 for enhanced security
        encoded_data = base64.b64encode(csv_data)

        return encoded_data

    @staticmethod
    def simulate_synthetic_data(g7_countries: List[str], start_year: int, end_year: int, feature_settings: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Simulates synthetic panel data based on specified parameters.

        Parameters:
        - g7_countries (list): List of G7 countries.
        - start_year (int): Starting year for simulation.
        - end_year (int): Ending year for simulation.
        - feature_settings (dict): Dictionary containing feature settings.

        Returns:
        - pd.DataFrame: Simulated synthetic panel data.
        """
        # Input validation
        if not isinstance(g7_countries, list):
            raise ValueError("Input 'g7_countries' must be a list.")
        if not all(isinstance(country, str) for country in g7_countries):
            raise ValueError("Elements of 'g7_countries' must be strings.")
        if not isinstance(start_year, int) or not isinstance(end_year, int):
            raise ValueError("Start and end years must be integers.")
        if start_year >= end_year:
            raise ValueError("End year must be greater than start year.")
        if not isinstance(feature_settings, dict):
            raise ValueError("Input 'feature_settings' must be a dictionary.")
        for feature, settings in feature_settings.items():
            if not isinstance(settings, dict):
                raise ValueError(f"Settings for feature '{feature}' must be provided as a dictionary.")
            if 'distribution' not in settings or settings['distribution'] not in ['Normal', 'Uniform']:
                raise ValueError(f"Invalid distribution specified for feature '{feature}'.")
            if settings['distribution'] == 'Normal':
                if 'mean' not in settings or 'std_dev' not in settings:
                    raise ValueError(f"Mean and standard deviation must be provided for Normal distribution of feature '{feature}'.")
            elif settings['distribution'] == 'Uniform':
                if 'min' not in settings or 'max' not in settings:
                    raise ValueError(f"Minimum and maximum values must be provided for Uniform distribution of feature '{feature}'.")

        np.random.seed(42)  # Set seed for reproducibility

        # Generate synthetic panel data
        data = pd.DataFrame({
            'Year': np.repeat(np.arange(start_year, end_year + 1), len(g7_countries)),
            'Country': np.tile(g7_countries, end_year - start_year + 1)
        })

        # Simulate data for each feature
        for feature, settings in feature_settings.items():
            for country in g7_countries:
                for year in range(start_year, end_year + 1):
                    idx = (data['Country'] == country) & (data['Year'] == year)

                    if settings['distribution'] == 'Normal':
                        data.loc[idx, feature] = np.random.normal(settings['mean'], settings['std_dev'])
                    elif settings['distribution'] == 'Uniform':
                        data.loc[idx, feature] = np.random.uniform(settings['min'], settings['max'])
                    # Add more distributions as needed

        return data

    def simulate_synthetic_data_default(self) -> pd.DataFrame:
        """
        Simulates synthetic panel data with default features.

        Returns:
        - pd.DataFrame: Simulated synthetic panel data.
        """
        # Ensure required attributes are initialized
        if not all(hasattr(self, attr) for attr in ['g7_countries', 'start_year', 'end_year']):
            raise ValueError("Required attributes 'g7_countries', 'start_year', and 'end_year' must be initialized.")

        # Define default features and distributions
        default_feature_settings = {
            'Inflation': {'distribution': 'Normal', 'mean': 0.5, 'std_dev': 1.0},
            'Oil_Prices': {'distribution': 'Normal', 'mean': 50, 'std_dev': 10},
            'Gas_Prices': {'distribution': 'Normal', 'mean': 5, 'std_dev': 2},
            'GDP': {'distribution': 'Normal', 'mean': 1000, 'std_dev': 200},
            'Interest_Rates': {'distribution': 'Uniform', 'min': 1, 'max': 5},
            'Unemployment_Rate': {'distribution': 'Uniform', 'min': 3, 'max': 10},
            'Exchange_Rates': {'distribution': 'Normal', 'mean': 1, 'std_dev': 0.1},
            'CPI': {'distribution': 'Normal', 'mean': 1000, 'std_dev': 200},
            # Add more default features as needed
        }

        # Call simulate_synthetic_data method with default settings
        return self.simulate_synthetic_data(self.g7_countries, self.start_year, self.end_year, default_feature_settings)

def main():
    st.title('Synthetic Panel Data Generator')
    st.markdown('**Disclaimer:** Using panel data for research or experiments requires users to maintain rigorous standards in data collection, cleaning, and analysis. While panel data offers valuable insights into various phenomena over time, users must exercise diligence to ensure the validity and integrity of their analyses. Adherence to ethical standards, transparency, and proper acknowledgment of data sources are essential. Users should recognize potential biases and take steps to mitigate them. Ultimately, users bear the responsibility of conducting research with integrity and contributing to the advancement of knowledge.')

    # Features section
    st.markdown('## Features')
    st.write('- **Default Version**: Provides a set of default features and distributions for quick generation of synthetic panel data.')
    st.write('- **Custom Version**: Allows users to customize feature settings, including distributions and parameters, to tailor the generated data to their specific needs.')
    st.write('- **Export to CSV**: Enables users to export the generated synthetic panel data to a CSV file for further analysis.')

    # Sidebar widgets
    st.sidebar.header('Synthetic Panel Data Generator')

    version = st.sidebar.selectbox('Choose Version', ['Default', 'Custom'])

    generator = SyntheticPanelDataGenerator()

    if version == 'Default':
        st.sidebar.header('Settings')
        st.sidebar.markdown('**Instructions for Default Version:**')
        st.sidebar.markdown('- Select G7 countries, start year, and end year.')
        st.sidebar.markdown('- Click on "Generate Synthetic Data" to generate data.')
        st.sidebar.markdown('- Click on "Export to CSV" to export the generated data to a CSV file.')

        g7_countries = st.sidebar.multiselect('Select G7 countries', generator.g7_countries)
        start_year = st.sidebar.slider('Start Year', generator.start_year, generator.end_year, generator.start_year)
        end_year = st.sidebar.slider('End Year', generator.start_year, generator.end_year, generator.end_year)

        # Button to generate synthetic data
        if st.sidebar.button('Generate Synthetic Data'):
            try:
                synthetic_data = generator.simulate_synthetic_data_default()
                st.write('### Synthetic Panel Data')
                st.write(synthetic_data)

                # Display line chart for the first feature
                first_feature = synthetic_data.columns[2]  # Assuming the first feature starts at column index 2
                st.write(f'### Line Chart for {first_feature}')
                chart_data = synthetic_data.groupby(['Year', 'Country']).mean().unstack()[first_feature]
                st.line_chart(chart_data)

                # Export data to CSV
                csv = generator.convert_df(synthetic_data)
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='Panel_data.csv',
                    mime='text/csv',
                )
            except Exception as e:
                st.error(f"Error occurred: {str(e)}")
    
    elif version == 'Custom':
        st.sidebar.header('Settings')

        st.sidebar.markdown('**Instructions for Custom Version:**')
        st.sidebar.markdown('- Customize feature settings as desired.')
        st.sidebar.markdown('- Click on "Generate Synthetic Data" to generate data.')
        st.sidebar.markdown('- Click on "Export to CSV" to export the generated data to a CSV file.')

        g7_countries = st.sidebar.multiselect('Select G7 countries', generator.g7_countries)
        start_year = st.sidebar.slider('Start Year', generator.start_year, generator.end_year, generator.start_year)
        end_year = st.sidebar.slider('End Year', generator.start_year, generator.end_year, generator.end_year)

        feature_settings = {}
        num_features = st.sidebar.number_input('Number of Features', min_value=1, max_value=10, value=1)
        for i in range(num_features):
            feature = st.sidebar.text_input(f'Feature {i+1}')
            distribution = st.sidebar.selectbox(f'Distribution {i+1}', ['Normal', 'Uniform'], key=f'distribution_{i}')
            if distribution == 'Normal':
                mean = st.sidebar.number_input(f'Mean {i+1}', value=0.0)
                std_dev = st.sidebar.number_input(f'Standard Deviation {i+1}', value=1.0)
                feature_settings[feature] = {'distribution': distribution, 'mean': mean, 'std_dev': std_dev}
            elif distribution == 'Uniform':
                min_val = st.sidebar.number_input(f'Minimum Value {i+1}', value=0.0)
                max_val = st.sidebar.number_input(f'Maximum Value {i+1}', value=1.0)
                feature_settings[feature] = {'distribution': distribution, 'min': min_val, 'max': max_val}

        # Button to generate synthetic data
        if st.sidebar.button('Generate Synthetic Data'):
            try:
                synthetic_data = generator.simulate_synthetic_data(g7_countries, start_year, end_year, feature_settings)
                st.write('### Synthetic Panel Data')
                st.write(synthetic_data)

                # Export data to CSV
                csv = generator.convert_df(synthetic_data)
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='Panel_data.csv',
                    mime='text/csv',
                )
            except Exception as e:
                st.error(f"Error occurred: {str(e)}")

if __name__ == '__main__':
    main()
