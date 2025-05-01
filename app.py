import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
    # Exclude invalid state codes
    return df[df["CAND_OFFICE_ST"] != "00"]

# Load data
cand_current = load_candidate_summary("weball24.txt")
cand_old = load_candidate_summary("weball20.txt").rename(columns={"TTL_INDIV_CONTRIB": "DON_OLD"})
cand_new = load_candidate_summary("weball22.txt").rename(columns={"TTL_INDIV_CONTRIB": "DON_NEW"})

# Committee load (for potential future use)
# comm_df = load_committee("cm.txt")

# Precompute state summaries
# Total disbursements
disburse_df = cand_current.groupby("CAND_OFFICE_ST")["TTL_DISB"].sum().reset_index()
# Cash on hand
coh_df = cand_current.groupby("CAND_OFFICE_ST").agg({"COH_BOP":"sum","COH_COP":"sum"}).reset_index()
coh_df['NET_COH'] = coh_df['COH_COP'] - coh_df['COH_BOP']
# Change in donations
old_tot = cand_old.groupby("CAND_OFFICE_ST")["DON_OLD"].sum().reset_index()
new_tot = cand_new.groupby("CAND_OFFICE_ST")["DON_NEW"].sum().reset_index()
change_df = old_tot.merge(new_tot, on="CAND_OFFICE_ST", how="outer").fillna(0)
change_df['CHANGE'] = change_df['DON_NEW'] - change_df['DON_OLD']
M = max(abs(change_df['CHANGE'].min()), change_df['CHANGE'].max())

# Party color mapping\party_mapping = {
    'DEM': 'Democratic', 'DFL': 'Democratic',
    'GOP': 'Republican', 'REP': 'Republican',
    'LIB': 'Libertarian', 'GRE': 'Green',
    'IND': 'Independent', 'CON': 'Constitution',
    'NPA': 'No Party Affiliation', 'OTH': 'Other',
    'UUP': 'United Utah Party'
}
color_map = {
    'Democratic': 'blue', 'Republican': 'red',
    'Libertarian': 'gold', 'Green': 'green',
    'Independent': 'purple', 'Constitution': 'brown',
    'No Party Affiliation': 'gray', 'Other': 'lightgray'
}

# 1. Treemap: Party → Candidate Fundraising
st.header("1. Treemap: Party → Candidate Fundraising")
df1 = cand_current[cand_current['TTL_RECEIPTS'] > 0].copy()
df1['Party'] = df1['CAND_PTY_AFFILIATION'].map(party_mapping).fillna('Other')
fig1 = px.treemap(
    df1, path=['Party','CAND_NAME'], values='TTL_RECEIPTS',
    color='Party', color_discrete_map=color_map,
    title='Fundraising Treemap'
)
st.plotly_chart(fig1, use_container_width=True)

# 2. Radar Chart: Party Receipts by Top States
st.header("2. Radar Chart: Party Receipts by Top States")
party_state = cand_current.groupby(['CAND_OFFICE_ST','CAND_PTY_AFFILIATION'])['TTL_RECEIPTS'].sum().reset_index()
party_state = party_state[party_state['CAND_PTY_AFFILIATION'].isin(['DEM','REP'])]
pivot_rs = party_state.pivot(index='CAND_OFFICE_ST', columns='CAND_PTY_AFFILIATION', values='TTL_RECEIPTS').fillna(0)
pivot_rs['TOTAL'] = pivot_rs.sum(axis=1)
top_states = pivot_rs.nlargest(6,'TOTAL').index.tolist()
pivot_rs = pivot_rs.loc[top_states]
states = pivot_rs.index.tolist()
dem_vals = pivot_rs['DEM'].tolist() + [pivot_rs['DEM'].tolist()[0]]
rep_vals = pivot_rs['REP'].tolist() + [pivot_rs['REP'].tolist()[0]]
states = states + [states[0]]
fig2 = go.Figure(
    data=[
        go.Scatterpolar(r=dem_vals, theta=states, fill='toself', name='Democratic', line_color='blue'),
        go.Scatterpolar(r=rep_vals, theta=states, fill='toself', name='Republican', line_color='red')
    ]
)
fig2.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
st.plotly_chart(fig2, use_container_width=True)

# 3. Violin Plot: Distribution of Fundraising Amounts by Party (Log Scale)
st.header("3. Violin Plot: Distribution of Fundraising Amounts by Party (Log Scale)")
candidate22_df = cand_new.rename(columns={'TTL_RECEIPTS':'TTL_RECEIPTS'}).copy()
violin_df = candidate22_df[['CAND_PTY_AFFILIATION','TTL_RECEIPTS']]
violin_df = violin_df[violin_df['TTL_RECEIPTS'] > 0]
violin_df['Party'] = violin_df['CAND_PTY_AFFILIATION'].map(party_mapping).fillna('Other')
violin_df['Log_Receipts'] = np.log1p(violin_df['TTL_RECEIPTS'])
plt.figure(figsize=(14, 8))
sns.violinplot(data=violin_df, x='Party', y='Log_Receipts', palette='muted')
plt.title('Distribution of Fundraising Amounts by Party (Log Scale)', fontsize=18)
plt.xlabel('Political Party', fontsize=14)
plt.ylabel('Log(Total Receipts)', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True)
st.pyplot(plt.gcf())

# 4. Choropleth: Change in Individual Donations by State
st.header("4. Choropleth: Change in Individual Donations by State")
fig4 = px.choropleth(change_df, locations='CAND_OFFICE_ST', locationmode='USA-states',
                     color='CHANGE', range_color=[-M,M], color_continuous_midpoint=0,
                     scope='usa', title='Δ Individual Donations by State')
st.plotly_chart(fig4, use_container_width=True)

# 5. Net Cash On Hand by State
st.header("5. Net Cash On Hand by State")
fig5 = px.choropleth(coh_df, locations='CAND_OFFICE_ST', locationmode='USA-states',
                     color='NET_COH', labels={'NET_COH':'Net Cash On Hand'},
                     scope='usa', title='Net Cash On Hand by State')
st.plotly_chart(fig5, use_container_width=True)

# 6. Total Disbursements by State
st.header("6. Total Disbursements by State")
fig6 = px.choropleth(disburse_df, locations='CAND_OFFICE_ST', locationmode='USA-states',
                     color='TTL_DISB', labels={'TTL_DISB':'Total Disbursements'},
                     scope='usa', title='Total Disbursements by State')
st.plotly_chart(fig6, use_container_width=True)