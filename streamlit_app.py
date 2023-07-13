import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import base64
import matplotlib.pyplot as plt
import seaborn as sns

# Function to format numbers as millions
def format_as_millions(num):
    return f'{num / 1e6:.2f}M' if num > 1e4 else str(num)

# Sidebar inputs
num_simulations = st.sidebar.number_input('Number of Simulations', min_value=1000, max_value=100000, value=10000)
num_bins = st.sidebar.number_input('Number of Bins', min_value=10, max_value=100, value=50)

# User input for cost ranges
overhead_range = st.sidebar.slider('Overhead Range ($)', min_value=2000, max_value=200000, value=(2000, 200000))
cots_chips_range = st.sidebar.slider('COTS Chips Range ($)', min_value=1000, max_value=10000, value=(1000, 10000))
custom_chips_range = st.sidebar.slider('Custom Chips Range ($)', min_value=1000, max_value=10000, value=(1000, 10000))
custom_chips_nre_range = st.sidebar.slider('Custom Chips NRE Range ($)', min_value=1000000, max_value=10000000, value=(1000000, 10000000))
custom_chips_licensing_range = st.sidebar.slider('Custom Chips Licensing Range ($)', min_value=0, max_value=1000000, value=(0, 1000000))
ebrick_chiplets_range = st.sidebar.slider('eBrick Chiplets Range ($)', min_value=20, max_value=150, value=(20, 150))
ebrick_chiplets_licensing_range = st.sidebar.slider('eBrick Chiplets Licensing Range ($)', min_value=0, max_value=1000000, value=(0, 1000000))
osat_range = st.sidebar.slider('OSAT Range ($)', min_value=500000, max_value=750000, value=(500000, 750000))
vv_tests_range = st.sidebar.slider('V&V Tests Range ($)', min_value=500000, max_value=750000, value=(500000, 750000))
profit_margin_range = st.sidebar.slider('Profit Margin Range (%)', min_value=20, max_value=30, value=(20, 30))

# User input for business value simulation
sales_volume = st.sidebar.number_input('Sales Volume', min_value=1, max_value=1000000, value=10000)
sales_price = st.sidebar.number_input('Sales Price ($)', min_value=1, max_value=100000, value=10000)
operating_expenses = st.sidebar.number_input('Operating Expenses ($)', min_value=0, max_value=10000000, value=1000000)
tax_rate = st.sidebar.slider('Tax Rate (%)', min_value=0, max_value=50, value=20)
discount_rate = st.sidebar.slider('Discount Rate (%)', min_value=0, max_value=50, value=10)

run_simulation = st.button('Run Simulation')

def business_plots(df):
    # Profit Margin Over Time
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=df.index, y='Profit Margin')
    plt.title('Profit Margin Over Time')
    plt.show()

    # Revenue vs. Cost
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x=df.index, y='Revenue', color='blue', label='Revenue')
    sns.barplot(data=df, x=df.index, y='Total Cost', color='red', label='Cost')
    plt.title('Revenue vs. Cost')
    plt.legend()
    plt.show()

    # Net Profit Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Net Profit', bins=30, kde=True)
    plt.title('Net Profit Distribution')
    plt.show()

    # Return on Investment (ROI)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=df.index, y='ROI')
    plt.title('Return on Investment Over Time')
    plt.show()

    # Cash Flow
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=df.index, y='Cash Flow')
    plt.title('Cash Flow Over Time')
    plt.show()
    
# Perform the simulations
@st.cache
def simulate(num_simulations, sales_volume, sales_price, operating_expenses, tax_rate, discount_rate, overhead_range, cots_chips_range, custom_chips_range, custom_chips_nre_range, custom_chips_licensing_range, ebrick_chiplets_range, ebrick_chiplets_licensing_range, osat_range, vv_tests_range, profit_margin_range):
    simulation_data = []  # Create an empty list to store simulation data
    for _ in range(num_simulations):
        overhead = round(np.random.uniform(*overhead_range), -2)
        cots_chips = round(np.random.randint(1, 6) * np.random.uniform(*cots_chips_range), -2)
        custom_chips = round(np.random.randint(0, 3) * np.random.uniform(*custom_chips_range), -2)
        custom_chips_nre = round(np.random.uniform(*custom_chips_nre_range), -2)
        custom_chips_licensing = round(np.random.uniform(*custom_chips_licensing_range), -2)
        ebrick_chiplets = round(np.random.choice(np.arange(16, 257, 16)) * np.random.uniform(*ebrick_chiplets_range), -2)
        ebrick_chiplets_licensing = round(np.random.uniform(*ebrick_chiplets_licensing_range), -2)
        osat = round(np.random.uniform(*osat_range), -2)
        vv_tests = round(np.random.uniform(*vv_tests_range), -2)
        cost_before_profit = round((overhead + cots_chips + custom_chips + custom_chips_nre +
                              custom_chips_licensing + ebrick_chiplets + ebrick_chiplets_licensing +
                              osat + vv_tests), -2)
        profit = round(np.random.uniform(profit_margin_range[0]/100, profit_margin_range[1]/100) * cost_before_profit, -2)
        total_cost = round(cost_before_profit + profit, -2)

        # Business value simulation
        revenue = sales_volume * sales_price
        gross_profit = revenue - total_cost
        net_profit_before_taxes = gross_profit - operating_expenses
        taxes = net_profit_before_taxes * tax_rate / 100
        net_profit = net_profit_before_taxes - taxes
        npv = net_profit / (1 + discount_rate / 100)

        simulation_data.append({
            'Overhead': format_as_millions(overhead),
            'COTS Chips': format_as_millions(cots_chips),
            'Custom Chips': format_as_millions(custom_chips),
            'Custom Chips NRE': format_as_millions(custom_chips_nre),
            'Custom Chips Licensing': format_as_millions(custom_chips_licensing),
            'eBrick Chiplets': format_as_millions(ebrick_chiplets),
            'eBrick Chiplets Licensing': format_as_millions(ebrick_chiplets_licensing),
            'OSAT': format_as_millions(osat),
            'V&V Tests': format_as_millions(vv_tests),
            'Profit': format_as_millions(profit),
            'Total Cost': format_as_millions(total_cost),
            'Revenue': format_as_millions(revenue),
            'Gross Profit': format_as_millions(gross_profit),
            'Net Profit Before Taxes': format_as_millions(net_profit_before_taxes),
            'Taxes': format_as_millions(taxes),
            'Net Profit': format_as_millions(net_profit),
            'NPV': format_as_millions(npv)
        })

    df = pd.DataFrame(simulation_data)
    return df

# Check if the Run Simulation button is clicked
if run_simulation:
    # Perform the simulations
    df = simulate(num_simulations, sales_volume, sales_price, operating_expenses, tax_rate, discount_rate, overhead_range, cots_chips_range, custom_chips_range, custom_chips_nre_range, custom_chips_licensing_range, ebrick_chiplets_range, ebrick_chiplets_licensing_range, osat_range, vv_tests_range, profit_margin_range)

    # Create a DataFrame to display the ranges for each variable
    ranges_df = pd.DataFrame(index=['min', 'max'])
    for column in df.columns:
        ranges_df[column] = [df[column].min(), df[column].max()]
    
    # Display the ranges DataFrame
    st.dataframe(ranges_df)

    # Display the dataframe
    st.dataframe(df)

    # Plot the histogram of total costs
    st.subheader('Histogram of Total Costs')
    fig = px.histogram(df, x='Total Cost', nbins=num_bins, marginal='box')
    st.plotly_chart(fig)

    # Identify the largest cost drivers
    st.subheader('Largest Cost Drivers')
    cost_drivers = df.drop(columns=['Total Cost', 'Revenue', 'Gross Profit', 'Net Profit Before Taxes', 'Taxes', 'Net Profit', 'NPV']).mean().sort_values(ascending=False)
    st.write(cost_drivers)

    # Identify the ideal value range for each variable to bring the average total cost below $5M
    st.subheader('Ideal Value Range for Each Variable')
    for column in df.columns:
        if column != 'Total Cost':
            ideal_range = df[df['Total Cost'] < '5.00M'][column].agg(['min', 'max'])
            st.write(f'{column}: {ideal_range[0]} - {ideal_range[1]}')

    # Identify the profit margin needed to keep the average total cost below $5M
    st.subheader('Profit Margin Needed')
    profit_margin_needed = round(df[df['Total Cost'] < '5.00M']['Profit'].mean() / df[df['Total Cost'] < '5.00M'].drop(columns='Profit').sum(axis=1).mean(), 2)
    st.write(f'Profit margin needed to keep the average total cost below $5M: {profit_margin_needed * 100}%')   

    # Create and display business plots
    business_plots(df)

# Downloadable results
if st.button('Download Results as CSV'):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="simulation_results.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)
