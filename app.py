import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from vega_datasets import data
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Configure page
st.set_page_config(page_title="Campaign Finance Dashboard", layout="wide")
st.title("Campaign Finance Visualization Dashboard")

# --- Data loaders ---
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

@st.cache_data
def load_candidate_master(path):
    cols = [
        'CAND_ID','CAND_NAME','CAND_PTY_AFFILIATION','CAND_ELECTION_YR',
        'CAND_OFFICE_ST','CAND_OFFICE','CAND_OFFICE_DISTRICT',
        'CAND_ICI','CAND_STATUS','CAND_PCC','CAND_ST1','CAND_ST2',
        'CAND_CITY','CAND_ST','CAND_ZIP'
    ]
    return pd.read_csv(path, sep='|', header=None, names=cols, dtype=str)

@st.cache_data
def load_individual(path):
    cols = [
        'CMTE_ID','AMNDT_IND','RPT_TP','TRANSACTION_PGI','IMAGE_NUM',
        'TRANSACTION_TP','ENTITY_TP','NAME','CITY','STATE','ZIP_CODE',
        'EMPLOYER','OCCUPATION','TRANSACTION_DT','TRANSACTION_AMT',
        'OTHER_ID','TRAN_ID','FILE_NUM','MEMO_CD','MEMO_TEXT','SUB_ID'
    ]
    df = pd.read_csv(path, delimiter='|', header=None, names=cols, low_memory=True)
    return df[df['STATE'].notna()]

# --- Load datasets ---
cand_current = load_candidate_summary('weball24.txt')
cand_old = load_candidate_summary('weball20.txt').rename(columns={'TTL_INDIV_CONTRIB':'DON_OLD'})
cand_new = load_candidate_summary('weball22.txt').rename(columns={'TTL_INDIV_CONTRIB':'DON_NEW'})
comm_df = load_committee('cm.txt')
master = load_candidate_master('cn.txt')
indiv_df = load_individual('itcont_2025_2026.txt')

# --- Topo for Altair maps ---
us_states = alt.topo_feature(data.us_10m.url, 'states')

# --- 0. House Race Competitiveness by State ---
st.header('0. House Race Competitiveness by State (Number of Candidates)')
house = master[master['CAND_OFFICE']=='H']
state_comp = house.groupby('CAND_OFFICE_ST').size().reset_index(name='num_candidates')
state_to_fips = { 'AL':1,'AK':2,'AZ':4,'AR':5,'CA':6,'CO':8,'CT':9,'DE':10,'DC':11,'FL':12,'GA':13,'HI':15,'ID':16,'IL':17,'IN':18,'IA':19,'KS':20,'KY':21,'LA':22,'ME':23,'MD':24,'MA':25,'MI':26,'MN':27,'MS':28,'MO':29,'MT':30,'NE':31,'NV':32,'NH':33,'NJ':34,'NM':35,'NY':36,'NC':37,'ND':38,'OH':39,'OK':40,'OR':41,'PA':42,'RI':44,'SC':45,'SD':46,'TN':47,'TX':48,'UT':49,'VT':50,'VA':51,'WA':53,'WV':54,'WI':55,'WY':56 }
state_comp['id'] = state_comp['CAND_OFFICE_ST'].map(state_to_fips)
chor_house = alt.Chart(us_states).mark_geoshape(stroke='white',strokeWidth=0.5).encode(
    color=alt.Color('num_candidates:Q',title='House Candidates',scale=alt.Scale(scheme='blues')),
    tooltip=[alt.Tooltip('num_candidates:Q',title='House Candidates')]
).transform_lookup(
    lookup='id',from_=alt.LookupData(state_comp,'id',['num_candidates'])
).project('albersUsa').properties(width=800,height=400)
st.altair_chart(chor_house, use_container_width=True)

# --- 1. Senate Race Competitiveness by State ---
st.header('1. Senate Race Competitiveness by State (Number of Candidates)')
senate = master[master['CAND_OFFICE']=='S']
sen_comp = senate.groupby('CAND_OFFICE_ST').size().reset_index(name='num_candidates')
sen_comp['id'] = sen_comp['CAND_OFFICE_ST'].map(state_to_fips)
chor_senate = alt.Chart(us_states).mark_geoshape(stroke='white',strokeWidth=0.5).encode(
    color=alt.Color('num_candidates:Q',title='Senate Candidates',scale=alt.Scale(scheme='viridis')),
    tooltip=[alt.Tooltip('num_candidates:Q',title='Senate Candidates')]
).transform_lookup(
    lookup='id',from_=alt.LookupData(sen_comp,'id',['num_candidates'])
).project('albersUsa').properties(width=800,height=400)
st.altair_chart(chor_senate, use_container_width=True)

# --- 2. Top 10 PACs by Total Receipts (2022) ---
st.header('2. Top 10 PACs by Total Receipts (2022)')
pac_cols=["CMTE_ID","CMTE_NM","CMTE_TP","CMTE_DSGN","CMTE_FILING_FREQ","TTL_RECEIPTS","TRANS_FROM_AFF","INDV_CONTRIB","OTHER_POL_CMTE_CONTRIB","CAND_CONTRIB","CAND_LOANS","TTL_LOANS_RECEIVED","TTL_DISB","TRANF_TO_AFF","INDV_REFUNDS","OTHER_POL_CMTE_REFUNDS","CAND_LOAN_REPAY","LOAN_REPAY","COH_BOP","COH_COP","DEBTS_OWED_BY","NONFED_TRANS_RECEIVED","CONTRIB_TO_OTHER_CMTE","IND_EXP","PTY_COORD_EXP","NONFED_SHARE_EXP","CVG_END_DT"]
pac_df=pd.read_csv('webk22.txt',delimiter='|',header=None,names=pac_cols,dtype=str)
pac_df['TTL_RECEIPTS']=pd.to_numeric(pac_df['TTL_RECEIPTS'],errors='coerce')
pac_df=pac_df.dropna(subset=['TTL_RECEIPTS'])
top10=pac_df.nlargest(10,'TTL_RECEIPTS')
pac_map={'ACTBLUE':'Democrat','DCCC':'Democrat','SMP':'Democrat','DNC SERVICES CORP / DEMOCRATIC NATIONAL COMMITTEE':'Democrat','DSCC':'Democrat','WINRED':'Republican','REPUBLICAN NATIONAL COMMITTEE':'Republican','SENATE LEADERSHIP FUND':'Republican','NRCC':'Republican','CONGRESSIONAL LEADERSHIP FUND':'Republican'}
top10['Party']=top10['CMTE_NM'].str.upper().map(pac_map)
top10=top10[top10['Party'].notna()]
t10=top10[['CMTE_NM','TTL_RECEIPTS','Party']]
bars=alt.Chart(t10).mark_bar(stroke='black',strokeWidth=1.5).encode(
    x=alt.X('TTL_RECEIPTS:Q',title='Total Receipts (USD)',axis=alt.Axis(format='$,.0f')),
    y=alt.Y('CMTE_NM:N',sort='-x',title='PAC Name'),
    color=alt.Color('Party:N',scale=alt.Scale(domain=['Democrat','Republican'],range=['#4F81BD','#C0504D']),legend=None),
    strokeDash=alt.condition(alt.datum.Party=='Republican',alt.value([5,5]),alt.value([0])),
    tooltip=[alt.Tooltip('CMTE_NM:N',title='PAC Name'),alt.Tooltip('TTL_RECEIPTS:Q',title='Total Raised ($)',format=',.0f'),alt.Tooltip('Party:N',title='Party')]
).properties(width=700,height=400)
st.altair_chart(bars,use_container_width=True)

# --- 3. Number of Campaign Donations by Party (Log-Log Scale) ---
st.header('3. Number of Campaign Donations by Party (Log-Log Scale)')
merge_df=load_individual('itcont_2025_2026.txt').merge(comm_df[['CMTE_ID','CMTE_PTY_AFFILIATION']],on='CMTE_ID',how='left')
merge_df=merge_df[merge_df['CMTE_PTY_AFFILIATION'].isin(['DEM','REP'])]
dem=merge_df[merge_df['CMTE_PTY_AFFILIATION']=='DEM']['TRANSACTION_AMT'].astype(float)
rep=merge_df[merge_df['CMTE_PTY_AFFILIATION']=='REP']['TRANSACTION_AMT'].astype(float)
plt.figure(figsize=(10,6))
bins=[1,10,25,50,100,250,500,1000,5000,10000,50000]
plt.hist(dem,bins=bins,alpha=0.6,label='Democratic',color='blue')
plt.hist(rep,bins=bins,alpha=0.6,label='Republican',color='red')
plt.xscale('log');plt.yscale('log')
plt.xlabel('Donation Amount ($)');plt.ylabel('Number of Donations')
plt.title('Number of Campaign Donations by Party (Log-Log Scale)')
plt.legend();plt.grid(True,which='both',linestyle='--',alpha=0.7)
st.pyplot(plt.gcf())

# --- 4. Treemap: Party → Candidate Fundraising ---
st.header('4. Treemap: Party → Candidate Fundraising')
df1=cand_current[cand_current['TTL_RECEIPTS']>0].copy()
df1['Party']=df1['CAND_PTY_AFFILIATION'].map({'DEM':'Democratic','REP':'Republican'})
fig1=px.treemap(df1,path=['Party','CAND_NAME'],values='TTL_RECEIPTS',color='Party',color_discrete_map={'Democratic':'blue','Republican':'red'},title='Fundraising Treemap')
st.plotly_chart(fig1,use_container_width=True)

# 2. Radar Chart: Party Receipts by Top States
st.header("2. Radar Chart: Party Receipts by Top States")
party_state = cand_current.groupby(['CAND_OFFICE_ST', 'CAND_PTY_AFFILIATION'])['TTL_RECEIPTS'].sum().reset_index()
party_state = party_state[party_state['CAND_PTY_AFFILIATION'].isin(['DEM', 'REP'])]
pivot_rs = party_state.pivot(
    index='CAND_OFFICE_ST',
    columns='CAND_PTY_AFFILIATION',
    values='TTL_RECEIPTS'
).fillna(0)
pivot_rs['TOTAL'] = pivot_rs.sum(axis=1)
top_states = pivot_rs.nlargest(6, 'TOTAL').index.tolist()
pivot_rs = pivot_rs.loc[top_states]
states = list(pivot_rs.index) + [top_states[0]]
dem_vals = list(pivot_rs['DEM']) + [pivot_rs['DEM'].iloc[0]]
rep_vals = list(pivot_rs['REP']) + [pivot_rs['REP'].iloc[0]]
fig2 = go.Figure(
    data=[
        go.Scatterpolar(
            r=dem_vals,
            theta=states,
            fill='toself',
            name='Democratic',
            line_color='blue'
        ),
        go.Scatterpolar(
            r=rep_vals,
            theta=states,
            fill='toself',
            name='Republican',
            line_color='red'
        )
    ]
)
fig2.update_layout(
    polar=dict(radialaxis=dict(visible=True)),
    showlegend=True,
    title='Party Receipts by Top States'
)
st.plotly_chart(fig2, use_container_width=True)

# 3. Violin Plot: Distribution of Fundraising Amounts by Party (Log Scale)
st.header("3. Violin Plot: Distribution of Fundraising Amounts by Party (Log Scale)")
candidate22_df = cand_new.copy()
violin_df = candidate22_df[['CAND_PTY_AFFILIATION', 'TTL_RECEIPTS']]
violin_df = violin_df[violin_df['TTL_RECEIPTS'] > 0]
violin_df['Party'] = violin_df['CAND_PTY_AFFILIATION'].map(party_mapping).fillna('Other')
violin_df['Log_Receipts'] = np.log1p(violin_df['TTL_RECEIPTS'])
plt.figure(figsize=(14, 8))
sns.violinplot(
    data=violin_df,
    x='Party',
    y='Log_Receipts',
    palette='muted'
)
plt.title('Distribution of Fundraising Amounts by Party (Log Scale)', fontsize=18)
plt.xlabel('Political Party', fontsize=14)
plt.ylabel('Log(Total Receipts)', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True)
st.pyplot(plt.gcf())

# 4. Choropleth: Change in Individual Donations by State
st.header("4. Choropleth: Change in Individual Donations by State")
fig4 = px.choropleth(
    change_df,
    locations='CAND_OFFICE_ST',
    locationmode='USA-states',
    color='CHANGE',
    range_color=[-M, M],
    color_continuous_midpoint=0,
    scope='usa',
    title='Δ Individual Donations by State'
)
st.plotly_chart(fig4, use_container_width=True)

# 5. Net Cash On Hand by State
st.header("5. Net Cash On Hand by State")
fig5 = px.choropleth(
    coh_df,
    locations='CAND_OFFICE_ST',
    locationmode='USA-states',
    color='NET_COH',
    labels={'NET_COH': 'Net Cash On Hand'},
    scope='usa',
    title='Net Cash On Hand by State'
)
st.plotly_chart(fig5, use_container_width=True)

# 6. Total Disbursements by State
st.header("6. Total Disbursements by State")
fig6 = px.choropleth(
    disburse_df,
    locations='CAND_OFFICE_ST',
    locationmode='USA-states',
    color='TTL_DISB',
    labels={'TTL_DISB': 'Total Disbursements'},
    scope='usa',
    title='Total Disbursements by State'
)
st.plotly_chart(fig6, use_container_width=True)