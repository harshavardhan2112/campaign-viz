import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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
    df = df[df["CAND_OFFICE_ST"] != "00"]
    return df

@st.cache_data
def load_committee(path):
    cols = [
        'CMTE_ID','CMTE_NM','TRES_NM','CMTE_ST1','CMTE_ST2',
        'CMTE_CITY','CMTE_ST','CMTE_ZIP','CMTE_DSGN','CMTE_TP',
        'CMTE_PTY_AFFILIATION','CMTE_FILING_FREQ','ORG_TP',
        'CONNECTED_ORG_NM','CAND_ID'
    ]
    df = pd.read_csv(path, delimiter='|', header=None, names=cols, low_memory=True)
    return df

# Load data
cand_df = load_candidate_summary("weball24.txt")
old_df = load_candidate_summary("weball20.txt").rename(columns={"TTL_INDIV_CONTRIB":"DON_OLD"})
new_df = load_candidate_summary("weball22.txt").rename(columns={"TTL_INDIV_CONTRIB":"DON_NEW"})
comm_df = load_committee("cm.txt")

# Sidebar selection
section = st.sidebar.selectbox("Choose Visualization Section", [
    "State-Level Choropleth",
    "Change in Donations Choropleth",
    "Top 10 Disbursement Gap",
    "Sunburst Fundraising Hierarchy",
    "Treemap of Fundraising by Party",
    "Financial Health Score",
    "Common Themes in Committee Names"
])

if section == "State-Level Choropleth":
    st.header("State-Level Spending Choropleth")
    df = cand_df.groupby("CAND_OFFICE_ST").agg(
        {"TTL_DISB":"sum","TTL_RECEIPTS":"sum","COH_BOP":"sum","COH_COP":"sum"}
    ).reset_index()
    df['NET_COH'] = df['COH_COP'] - df['COH_BOP']
    metric = st.selectbox("Select Metric", ["TTL_DISB","TTL_RECEIPTS","NET_COH"])
    fig = px.choropleth(
        df, locations="CAND_OFFICE_ST", locationmode="USA-states", color=metric,
        scope="usa", labels={metric:metric}, title=f"{metric} by State"
    )
    st.plotly_chart(fig, use_container_width=True)

elif section == "Change in Donations Choropleth":
    st.header("Change in Individual Donations by State")
    old = old_df.groupby("CAND_OFFICE_ST")["DON_OLD"].sum().reset_index()
    new = new_df.groupby("CAND_OFFICE_ST")["DON_NEW"].sum().reset_index()
    merged = old.merge(new, on="CAND_OFFICE_ST", how="outer").fillna(0)
    merged['CHANGE'] = merged['DON_NEW'] - merged['DON_OLD']
    M = max(abs(merged['CHANGE'].min()), merged['CHANGE'].max())
    fig = px.choropleth(
        merged, locations="CAND_OFFICE_ST", locationmode="USA-states", color="CHANGE",
        range_color=[-M, M], color_continuous_midpoint=0,
        scope="usa", title="Δ Individual Donations by State"
    )
    st.plotly_chart(fig, use_container_width=True)

elif section == "Top 10 Disbursement Gap":
    st.header("Top 10 States by Disbursement Gap (DEM vs REP)")
    df1 = cand_df[["CAND_PTY_AFFILIATION","CAND_OFFICE_ST","TTL_DISB"]].copy()
    df1 = df1[df1['CAND_PTY_AFFILIATION'].isin(['DEM','REP'])]
    agg = df1.groupby(['CAND_OFFICE_ST','CAND_PTY_AFFILIATION'], as_index=False).sum()
    pivot = agg.pivot(index='CAND_OFFICE_ST', columns='CAND_PTY_AFFILIATION', values='TTL_DISB').fillna(0)
    pivot['GAP'] = (pivot['DEM'] - pivot['REP']).abs()
    top10 = pivot.nlargest(10,'GAP').reset_index()
    long = top10.melt(id_vars='CAND_OFFICE_ST', value_vars=['DEM','REP'], var_name='Party', value_name='Disbursements')
    fig = px.line(long, x='Party', y='Disbursements', color='CAND_OFFICE_ST', markers=True, title='Disbursement Gap')
    st.plotly_chart(fig, use_container_width=True)

elif section == "Sunburst Fundraising Hierarchy":
    st.header("Fundraising Hierarchy: Party → Candidate → State")
    sun = cand_df[["CAND_PTY_AFFILIATION","CAND_NAME","CAND_OFFICE_ST","TTL_RECEIPTS"]]
    sun = sun[sun['TTL_RECEIPTS']>0]
    mapping = {'DEM':'Dem','REP':'Rep'}
    sun['Party'] = sun['CAND_PTY_AFFILIATION'].map(mapping).fillna('Other')
    sun = sun.rename(columns={'CAND_NAME':'Candidate','CAND_OFFICE_ST':'State','TTL_RECEIPTS':'Amount'})
    fig = px.sunburst(sun, path=['Party','Candidate','State'], values='Amount', title='Fundraising Hierarchy')
    st.plotly_chart(fig, use_container_width=True)

elif section == "Treemap of Fundraising by Party":
    st.header("Treemap: Party → Candidate Fundraising")
    df = cand_df[cand_df['TTL_RECEIPTS']>0]
    df['Party'] = df['CAND_PTY_AFFILIATION'].map({'DEM':'Democratic','REP':'Republican'}).fillna('Other')
    fig = px.treemap(df, path=['Party','CAND_NAME'], values='TTL_RECEIPTS', title='Fundraising Treemap')
    st.plotly_chart(fig, use_container_width=True)

elif section == "Financial Health Score":
    st.header("Top 20 Candidates by Financial Health")
    df = cand_df.dropna(subset=['TTL_RECEIPTS','TTL_DISB','COH_COP'])
    df = df[(df['TTL_RECEIPTS']>0)&(df['TTL_DISB']>0)&(df['COH_COP']>0)]
    df['Health'] = df['TTL_RECEIPTS'] + df['COH_COP'] - df['TTL_DISB']
    top = df.nlargest(20,'Health')[['CAND_NAME','Health']].set_index('CAND_NAME')
    fig, ax = plt.subplots(figsize=(10,8))
    top['Health'].plot(kind='barh', ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel('Health Score')
    st.pyplot(fig)

else:
    st.header("Common Themes in Committee Names")
    text = ' '.join(comm_df['CMTE_NM'].dropna().str.lower().tolist())
    text = ''.join([c for c in text if c.isalpha() or c.isspace()])
    words = [w for w in text.split() if w not in set(['for','committee','the','and','of','in','to','a','an'])]
    freqs = pd.Series(words).value_counts().head(50).to_dict()
    wc = WordCloud(width=800,height=400, background_color='white').generate_from_frequencies(freqs)
    fig, ax = plt.subplots(figsize=(12,6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
