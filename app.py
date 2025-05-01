import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Configure page
st.set_page_config(page_title="Campaign Finance Dashboard", layout="wide")
st.title("Campaign Finance Visualization Dashboard")

@st.cache_data
def load_candidate_summary(path):
    cols = [
        "CAND_ID","CAND_NAME","CAND_ICI","PTY_CD","CAND_PTY_AFFILIATION",
        "TTL_RECEIPTS","TRANS_FROM_AUTH","TTL_DISB","TRANS_TO_AUTH",
        "COH_BOP","COH_COP","CAND_CONTRIB","CAND_LOANS","OTHER_LOANS",
        "CAND_LOAN_REPAY","OTHER_LOAN_REPAY","DEBTS_OWED_BY","TTL_INDIV_CONTRIB",
        "CAND_OFFICE_ST","CAND_OFFICE_DISTRICT","SPEC_ELECTION","PRIM_ELECTION",
        "RUN_ELECTION","GEN_ELECTION","GEN_ELECTION_PERCENT",
        "OTHER_POL_CMTE_CONTRIB","POL_PTY_CONTRIB","CVG_END_DT",
        "INDIV_REFUNDS","CMTE_REFUNDS"
    ]
    df = pd.read_csv(path, delimiter='|', header=None, names=cols, low_memory=True,
                     parse_dates=["CVG_END_DT"])
    return df[df["CAND_OFFICE_ST"] != "00"]

@st.cache_data
def load_committee(path):
    cols = [
        'CMTE_ID','CMTE_NM','TRES_NM','CMTE_ST1','CMTE_ST2',
        'CMTE_CITY','CMTE_ST','CMTE_ZIP','CMTE_DSGN','CMTE_TP',
        'CMTE_PTY_AFFILIATION','CMTE_FILING_FREQ','ORG_TP',
        'CONNECTED_ORG_NM','CAND_ID'
    ]
    return pd.read_csv(path, delimiter='|', header=None, names=cols, low_memory=True)

# Load data
cand_df = load_candidate_summary("weball24.txt")
old_df = load_candidate_summary("weball20.txt").rename(columns={"TTL_INDIV_CONTRIB": "DON_OLD"})
new_df = load_candidate_summary("weball22.txt").rename(columns={"TTL_INDIV_CONTRIB": "DON_NEW"})
comm_df = load_committee("cm.txt")

# Precompute state-level summaries
disburse_df = cand_df.groupby("CAND_OFFICE_ST")["TTL_DISB"].sum().reset_index()
coh_df = cand_df.groupby("CAND_OFFICE_ST").agg({"COH_BOP":"sum","COH_COP":"sum"}).reset_index()
coh_df['NET_COH'] = coh_df['COH_COP'] - coh_df['COH_BOP']

# Precompute change in donations by state
old_tot = old_df.groupby("CAND_OFFICE_ST")["DON_OLD"].sum().reset_index()
new_tot = new_df.groupby("CAND_OFFICE_ST")["DON_NEW"].sum().reset_index()
change_df = old_tot.merge(new_tot, on="CAND_OFFICE_ST", how="outer").fillna(0)
change_df['CHANGE'] = change_df['DON_NEW'] - change_df['DON_OLD']
M = max(abs(change_df['CHANGE'].min()), change_df['CHANGE'].max())

# 1. Treemap: Party → Candidate Fundraising
st.header("1. Treemap: Party → Candidate Fundraising")
df1 = cand_df[cand_df['TTL_RECEIPTS'] > 0].copy()
df1['Party'] = df1['CAND_PTY_AFFILIATION'].map({'DEM':'Democratic','REP':'Republican'}).fillna('Other')
fig1 = px.treemap(
    df1, path=['Party','CAND_NAME'], values='TTL_RECEIPTS',
    color='Party',
    color_discrete_map={'Democratic':'blue','Republican':'red','Other':'gray'},
    title='Fundraising Treemap'
)
st.plotly_chart(fig1, use_container_width=True)

# 2. Radar Chart: Party Receipts Comparison Across Top States
st.header("2. Radar Chart: Top Party Receipts by State")
party_state = cand_df.groupby(['CAND_OFFICE_ST','CAND_PTY_AFFILIATION'])['TTL_RECEIPTS'].sum().reset_index()
party_state = party_state[party_state['CAND_PTY_AFFILIATION'].isin(['DEM','REP'])]
pivot_rs = party_state.pivot(index='CAND_OFFICE_ST', columns='CAND_PTY_AFFILIATION', values='TTL_RECEIPTS').fillna(0)
pivot_rs['TOTAL'] = pivot_rs.sum(axis=1)
top_states = pivot_rs.nlargest(6,'TOTAL').index.tolist()
pivot_rs = pivot_rs.loc[top_states]
states = pivot_rs.index.tolist()
dem_vals = pivot_rs['DEM'].tolist()
rep_vals = pivot_rs['REP'].tolist()
states += [states[0]]
dem_vals += [dem_vals[0]]
rep_vals += [rep_vals[0]]
fig2 = go.Figure(
    data=[
        go.Scatterpolar(r=dem_vals, theta=states, fill='toself', name='Democratic', line_color='blue'),
        go.Scatterpolar(r=rep_vals, theta=states, fill='toself', name='Republican', line_color='red')
    ]
)
fig2.update_layout(
    polar=dict(radialaxis=dict(visible=True)),
    showlegend=True,
    title='Party Receipts by Top States'
)
st.plotly_chart(fig2, use_container_width=True)

# 3. Violin Plot: Distribution of Money Raised Per Party
st.header("3. Violin Plot: Distribution of Money Raised Per Party")
df3 = cand_df[['CAND_PTY_AFFILIATION','TTL_RECEIPTS']].copy()
df3 = df3.rename(columns={'CAND_PTY_AFFILIATION':'Party','TTL_RECEIPTS':'Amount'})
df3['Party'] = df3['Party'].map({'DEM':'Democratic','REP':'Republican'}).fillna('Other')
fig3 = px.violin(
    df3, x='Party', y='Amount', box=True, points='all',
    color='Party', color_discrete_map={'Democratic':'blue','Republican':'red','Other':'gray'},
    title='Money Raised Distribution by Party'
)
st.plotly_chart(fig3, use_container_width=True)

# 4. Choropleth: Change in Individual Donations by State
st.header("4. Choropleth: Change in Individual Donations by State")
fig4 = px.choropleth(
    change_df, locations="CAND_OFFICE_ST", locationmode="USA-states", color="CHANGE",
    range_color=[-M,M], color_continuous_midpoint=0,
    scope="usa", title="Δ Individual Donations by State"
)
st.plotly_chart(fig4, use_container_width=True)

# 5. Net Cash On Hand by State
st.header("5. Net Cash On Hand by State")
fig5 = px.choropleth(
    coh_df, locations="CAND_OFFICE_ST", locationmode="USA-states", color="NET_COH",
    labels={"NET_COH":"Net Cash On Hand"},
    scope="usa", title="Net Cash On Hand by State"
)
st.plotly_chart(fig5, use_container_width=True)

# 6. Total Disbursements by State
st.header("6. Total Disbursements by State")
fig6 = px.choropleth(
    disburse_df, locations="CAND_OFFICE_ST", locationmode="USA-states", color="TTL_DISB",
    labels={"TTL_DISB":"Total Disbursements"},
    scope="usa", title="Total Disbursements by State"
)
st.plotly_chart(fig6, use_container_width=True)