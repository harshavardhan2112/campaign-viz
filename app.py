# Streamlit version of Harsha's Campaign Finance Visualization App
import streamlit as st
import pandas as pd
import plotly.express as px
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud

st.set_page_config(layout="wide")

st.title("U.S. Campaign Finance Dashboard")

# Sidebar for navigation
pages = st.sidebar.selectbox(
    "Choose Visualization",
    [
        "Choropleth: Total Disbursements",
        "Treemap: Party Fundraising"
    ]
)

# Load a sample dataset for demonstration (replace with actual data loading logic)
@st.cache_data
def load_data():
    df = pd.read_csv("weball24.txt", sep="|", header=None, low_memory=True)
    return df

if pages == "Choropleth: Total Disbursements":
    st.subheader("Total Disbursements by State")
    df = pd.read_csv("weball24.txt", sep="|", header=None, low_memory=True)
    df.columns = ["CAND_ID", "CAND_NAME", "CAND_ICI", "PTY_CD", "CAND_PTY_AFFILIATION", "TTL_RECEIPTS", "TRANS_FROM_AUTH", "TTL_DISB", "TRANS_TO_AUTH", "COH_BOP", "COH_COP", "CAND_CONTRIB", "CAND_LOANS", "OTHER_LOANS", "CAND_LOAN_REPAY", "OTHER_LOAN_REPAY", "DEBTS_OWED_BY", "TTL_INDIV_CONTRIB", "CAND_OFFICE_ST", "CAND_OFFICE_DISTRICT", "SPEC_ELECTION", "PRIM_ELECTION", "RUN_ELECTION", "GEN_ELECTION", "GEN_ELECTION_PERCENT", "OTHER_POL_CMTE_CONTRIB", "POL_PTY_CONTRIB", "CVG_END_DT", "INDIV_REFUNDS", "CMTE_REFUNDS"]
    df = df[df["CAND_OFFICE_ST"] != "00"]
    state_totals = df.groupby("CAND_OFFICE_ST")["TTL_DISB"].sum().reset_index()
    fig = px.choropleth(
        state_totals,
        locations="CAND_OFFICE_ST",
        locationmode="USA-states",
        color="TTL_DISB",
        scope="usa",
        title="Total Disbursements by State"
    )
    st.plotly_chart(fig)

elif pages == "Treemap: Party Fundraising":
    st.subheader("Treemap: Party → Candidate Fundraising")
    df = pd.read_csv("weball22.txt", sep="|", header=None, low_memory=True)
    df.columns = ["CAND_ID", "CAND_NAME", "CAND_ICI", "PTY_CD", "CAND_PTY_AFFILIATION", "TTL_RECEIPTS", "TRANS_FROM_AUTH", "TTL_DISB", "TRANS_TO_AUTH", "COH_BOP", "COH_COP", "CAND_CONTRIB", "CAND_LOANS", "OTHER_LOANS", "CAND_LOAN_REPAY", "OTHER_LOAN_REPAY", "DEBTS_OWED_BY", "TTL_INDIV_CONTRIB", "CAND_OFFICE_ST", "CAND_OFFICE_DISTRICT", "SPEC_ELECTION", "PRIM_ELECTION", "RUN_ELECTION", "GEN_ELECTION", "GEN_ELECTION_PRECENT", "OTHER_POL_CMTE_CONTRIB", "POL_PTY_CONTRIB", "CVG_END_DT", "INDIV_REFUNDS", "CMTE_REFUNDS"]
    df = df.dropna(subset=['CAND_NAME', 'CAND_PTY_AFFILIATION', 'TTL_RECEIPTS', 'TTL_DISB', 'COH_BOP', 'COH_COP'])
    df = df[(df['TTL_RECEIPTS'] > 0) & (df['TTL_DISB'] > 0)]
    
    def clean_party(party):
        if party in ['DEM', 'DFL']:
            return 'Democratic'
        elif party == 'REP':
            return 'Republican'
        elif party == 'IND':
            return 'Independent'
        else:
            return 'Other'

    df['Party_Clean'] = df['CAND_PTY_AFFILIATION'].apply(clean_party)
    df['Health_Score'] = (df['TTL_RECEIPTS'] + df['COH_COP']) - df['TTL_DISB']
    
    fig = px.treemap(
        df,
        path=['Party_Clean', 'CAND_NAME'],
        values='TTL_RECEIPTS',
        color='Health_Score',
        hover_data={
            'TTL_RECEIPTS': ':,.0f',
            'COH_COP': ':,.0f',
            'Health_Score': ':,.0f'
        },
        color_continuous_scale='Viridis',
        title="Nested Treemap: Party → Candidate (Fundraising and Financial Health)"
    )
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    st.plotly_chart(fig)

st.markdown("---")
st.caption("Built with 💙 by Harsha Vardhan")
