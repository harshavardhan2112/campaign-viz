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

# --- Load datasets ---
cand_current = load_candidate_summary('weball24.txt')
cand_old = load_candidate_summary('weball20.txt').rename(columns={'TTL_INDIV_CONTRIB':'DON_OLD'})
cand_new = load_candidate_summary('weball22.txt').rename(columns={'TTL_INDIV_CONTRIB':'DON_NEW'})
# Party mapping for treemap
party_map = {'DEM':'Democratic', 'REP':'Republican'}
comm_df = load_committee('cm.txt')
master = load_candidate_master('cn.txt')

alt.data_transformers.disable_max_rows()

# --- Load Data for Contributor Type Visualization ---
@st.cache_data
def load_combined_data():
    cn_master = pd.read_csv("cn_master_combined.csv")
    weball = pd.read_csv("weball_combined.csv")

    cn_master.drop(columns=['CAND_ST2'], inplace=True, errors='ignore')
    cn_master = cn_master[cn_master['file_year'] >= 1980]

    weball.rename(columns={'file_year': 'CAND_ELECTION_YR'}, inplace=True)
    weball.drop(columns=[
        'SPEC_ELECTION', 'PRIM_ELECTION', 'RUN_ELECTION',
        'GEN_ELECTION', 'GEN_ELECTION_PRECENT'
    ], inplace=True, errors='ignore')
    weball = weball[weball['CAND_ELECTION_YR'] >= 1980]

    return cn_master, weball

cn_master, all_cand = load_combined_data()

st.header("1. Contributions Per Party and Donor Type")

contrib_type_labels = {
    "CAND_CONTRIB":            "Candidate Contributions",
    "CAND_LOANS":              "Candidate Loans",
    "OTHER_LOANS":             "Other Loans",
    "OTHER_POL_CMTE_CONTRIB":  "Other Committee Contributions",
    "POL_PTY_CONTRIB":         "Party Committee Contributions"
}
contrib_cols = list(contrib_type_labels.keys())

melted = (
    all_cand
    .assign(CAND_ELECTION_YR=all_cand["CAND_ELECTION_YR"].astype(int))
    .melt(
        id_vars=["CAND_ELECTION_YR", "CAND_PTY_AFFILIATION"],
        value_vars=contrib_cols,
        var_name="contrib_type",
        value_name="amount"
    )
    .assign(contrib_label=lambda df: df["contrib_type"].map(contrib_type_labels))
)

grouped = (
    melted
    .groupby(["CAND_ELECTION_YR", "CAND_PTY_AFFILIATION", "contrib_label"], as_index=False)
    .sum()
)

min_year = int(grouped["CAND_ELECTION_YR"].min())
max_year = int(grouped["CAND_ELECTION_YR"].max())
selected_year = st.slider("Select Election Year", min_value=min_year, max_value=max_year, step=2, value=min_year)

filtered = grouped[grouped["CAND_ELECTION_YR"] == selected_year]

chart = (
    alt.Chart(filtered)
    .mark_bar()
    .encode(
        x=alt.X("CAND_PTY_AFFILIATION:N", title="Party"),
        y=alt.Y("amount:Q", title="Total Contribution (USD)"),
        color=alt.Color(
            "contrib_label:N",
            title="Contributor Type",
            legend=alt.Legend(orient="right", titleFontSize=14, labelFontSize=10, symbolSize=100)
        ),
        tooltip=[
            alt.Tooltip("CAND_PTY_AFFILIATION:N", title="Party"),
            alt.Tooltip("contrib_label:N", title="Type"),
            alt.Tooltip("amount:Q", title="Amount", format=",.2f")
        ]
    )
    .properties(
        width=700,
        height=400,
        title=f"Contributions by Party in {selected_year}"
    )
    .configure_axis(labelFontSize=14, titleFontSize=16)
    .configure_title(fontSize=18, anchor="middle")
)

st.altair_chart(chart, use_container_width=True)

st.markdown("""
**Stacked Bar Chart:** Showing contributions for each party per year, with contributions stacked by donor category.  
**Key Takeaway:** Democrats and Republicans dominated the rest of the parties financially.  
**Design Note:** Hover-over tooltips increase interactivity without cluttering the view.
""")


# Party mapping for violin
party_map = {
    'DEM': 'Democratic',
    'DFL': 'Democratic',
    'GOP': 'Republican',
    'REP': 'Republican',
    'LIB': 'Libertarian',
    'GRE': 'Green',
    'IND': 'Independent',
    'CON': 'Constitution',
    'NPA': 'No Party Affiliation',
    'OTH': 'Other'
}

# --- 1. Violin Plot: Distribution of Fundraising Amounts by Party (Log Scale) --- Distribution of Fundraising Amounts by Party (Log Scale) ---
st.header("2. Distribution of Fundraising Amounts by Political Party")
violin_df = cand_current[['CAND_PTY_AFFILIATION','TTL_RECEIPTS']].copy()
violin_df = violin_df[violin_df['TTL_RECEIPTS'] > 0]
violin_df['Party'] = violin_df['CAND_PTY_AFFILIATION'].map(party_map).fillna('Other')
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

st.markdown("""
**Violin Plot:** showing distribution of funding by party.
**Key Takeaway:** Both Democrats and Republicans had higher mean campaign contribution amounts than other parties with constitution party being the lowest.
**Design Notes:** Log scale used to manage extreme fundraising outliers. Violin plot shape emphasizes spread, median, and distribution thickness.
""")


# --- Section: Donor Category Trends Over Time ---
st.header("3. Contributions over time by different donors")

contrib_type_labels = {
    "CAND_CONTRIB":            "Candidate Contributions",
    "CAND_LOANS":              "Candidate Loans",
    "OTHER_LOANS":             "Other Loans",
    "OTHER_POL_CMTE_CONTRIB":  "Other Committee Contributions",
    "POL_PTY_CONTRIB":         "Party Committee Contributions"
}
donor_cols = list(contrib_type_labels.keys())

df_long = (
    all_cand
    .melt(
        id_vars=["CAND_ELECTION_YR"],
        value_vars=donor_cols,
        var_name="donor_category",
        value_name="amount"
    )
    .assign(
        donor_label=lambda d: d["donor_category"].map(contrib_type_labels),
        amount=lambda d: pd.to_numeric(d["amount"], errors="coerce").fillna(0)
    )
)

plot_df = (
    df_long
    .groupby(["CAND_ELECTION_YR", "donor_label"], as_index=False)
    .agg(total_amount=("amount", "sum"))
)

plot_df["CAND_ELECTION_YR"] = plot_df["CAND_ELECTION_YR"].astype(str)

line_chart = (
    alt.Chart(plot_df)
    .mark_line(point=True)
    .encode(
        x=alt.X("CAND_ELECTION_YR:O", title="Election Year"),
        y=alt.Y("total_amount:Q", title="Total Contribution (USD)"),
        color=alt.Color("donor_label:N", title="Donor Category"),
        tooltip=[
            alt.Tooltip("CAND_ELECTION_YR:O", title="Election Year"),
            alt.Tooltip("donor_label:N",       title="Donor Category"),
            alt.Tooltip("total_amount:Q",      title="Total (USD)", format=",.0f")
        ]
    )
    .properties(width=800, height=450,
                title="Contributions by Donor Category Over Time")
    .configure_axis(labelFontSize=14, titleFontSize=16)
    .configure_title(fontSize=18, anchor="middle")
)

st.altair_chart(line_chart, use_container_width=True)

st.markdown("""
**Line Chart:**  showing contributions by each donor type over time.
**Key takeaway:** In 2020, there is an anomaly showing problems with the data for that year. Election costs have been steadily growing each year.
**Design note:** colour schemes kept consistent with previous graph, picked colors to align with colorblind accessibility.
""")


# --- Section: Interactive Contribution Category Breakdown ---
st.header("4. Contributions by Different Donors")

contrib_type_labels = {
    "CAND_CONTRIB":            "Candidate Contributions",
    "CAND_LOANS":              "Candidate Loans",
    "OTHER_LOANS":             "Other Loans",
    "OTHER_POL_CMTE_CONTRIB":  "Other Committee Contrib",
    "POL_PTY_CONTRIB":         "Party Committee Contrib"
}
donor_cols = list(contrib_type_labels.keys())

df_long = all_cand.melt(
    id_vars=["CAND_ELECTION_YR"],
    value_vars=donor_cols,
    var_name="donor_category",
    value_name="amount"
)
df_long["donor_label"] = df_long["donor_category"].map(contrib_type_labels)

min_year = int(df_long["CAND_ELECTION_YR"].min())
max_year = int(df_long["CAND_ELECTION_YR"].max())

year_slider = alt.binding_range(
    min=min_year,
    max=max_year,
    step=2,
    name="Select year: "
)
year_param = alt.param(
    name="YearParam",
    bind=year_slider,
    value=min_year
)

hover = alt.selection_point(
    fields=["donor_label"],
    on="mouseover",
    nearest=True,
    empty="none"
)

chart = (
    alt.Chart(df_long)
    .add_params(year_param, hover)
    .transform_filter("datum.CAND_ELECTION_YR === YearParam")
    .mark_bar()
    .encode(
        x=alt.X(
            "donor_label:N",
            title="Contributor Category",
            sort=list(contrib_type_labels.values())
        ),
        y=alt.Y("sum(amount):Q", title="Total Contribution (USD)"),
        color=alt.Color("donor_label:N", title="Contributor Category"),
        opacity=alt.condition(hover, alt.value(1), alt.value(0.85)),
        tooltip=[
            alt.Tooltip("donor_label:N", title="Category"),
            alt.Tooltip("sum(amount):Q", title="Total (USD)", format=",.0f"),
            alt.Tooltip("CAND_ELECTION_YR:O", title="Election Year")
        ]
    )
    .properties(
        width=700,
        height=450,
        title="Contributions by Contributor Category"
    )
    .configure_axis(labelFontSize=14, titleFontSize=16)
    .configure_title(fontSize=18, anchor="middle")
)

st.altair_chart(chart, use_container_width=True)

st.markdown("""
**Bar Chart:** Displays contributions by each donor category for a selected year using a year slider.  
**Key Takeaways:**  
- In **1984**, most campaign finances came from **loans**, marking a unique pattern not seen in other years.  
- In **2022**, the dominant sources were **candidate loans**, **candidate contributions**, and **action committee contributions**.  

**Design Note:**  
A **year slider** has been included below the chart to allow interactive exploration of contribution patterns across different election years without overwhelming the viewer.
""")



# --- 2. Top 10 PACs by Total Receipts (2022) ---
st.header('5. Top 10 PACs')
pac_cols=[
    "CMTE_ID","CMTE_NM","CMTE_TP","CMTE_DSGN","CMTE_FILING_FREQ",
    "TTL_RECEIPTS","TRANS_FROM_AFF","INDV_CONTRIB","OTHER_POL_CMTE_CONTRIB",
    "CAND_CONTRIB","CAND_LOANS","TTL_LOANS_RECEIVED","TTL_DISB",
    "TRANF_TO_AFF","INDV_REFUNDS","OTHER_POL_CMTE_REFUNDS",
    "CAND_LOAN_REPAY","LOAN_REPAY","COH_BOP","COH_COP","DEBTS_OWED_BY",
    "NONFED_TRANS_RECEIVED","CONTRIB_TO_OTHER_CMTE","IND_EXP",
    "PTY_COORD_EXP","NONFED_SHARE_EXP","CVG_END_DT"
]
pac_df = pd.read_csv('webk22.txt',delimiter='|',header=None,names=pac_cols,dtype=str)
pac_df['TTL_RECEIPTS'] = pd.to_numeric(pac_df['TTL_RECEIPTS'],errors='coerce')
pac_df = pac_df.dropna(subset=['TTL_RECEIPTS'])
top10 = pac_df.nlargest(10,'TTL_RECEIPTS')
pac_map = {
    'ACTBLUE':'Democrat','DCCC':'Democrat','SMP':'Democrat',
    'DNC SERVICES CORP / DEMOCRATIC NATIONAL COMMITTEE':'Democrat','DSCC':'Democrat',
    'WINRED':'Republican','REPUBLICAN NATIONAL COMMITTEE':'Republican',
    'SENATE LEADERSHIP FUND':'Republican','NRCC':'Republican','CONGRESSIONAL LEADERSHIP FUND':'Republican'
}
top10['Party'] = top10['CMTE_NM'].str.upper().map(pac_map)
top10 = top10[top10['Party'].notna()]
t10 = top10[['CMTE_NM','TTL_RECEIPTS','Party']]
bars = alt.Chart(t10).mark_bar(stroke='black',strokeWidth=1.5).encode(
    x=alt.X('TTL_RECEIPTS:Q',title='Total Receipts (USD)',axis=alt.Axis(format='$,.0f')),
    y=alt.Y('CMTE_NM:N',sort='-x',title='PAC Name'),
    color=alt.Color('Party:N',scale=alt.Scale(domain=['Democrat','Republican'],range=['#4F81BD','#C0504D']),legend=None),
    strokeDash=alt.condition(alt.datum.Party=='Republican',alt.value([5,5]),alt.value([0])),
    tooltip=[
        alt.Tooltip('CMTE_NM:N',title='PAC Name'),
        alt.Tooltip('TTL_RECEIPTS:Q',title='Total Raised ($)',format=',.0f'),
        alt.Tooltip('Party:N',title='Party')
    ]
).properties(width=700,height=400)
st.altair_chart(bars,use_container_width=True)

st.markdown("""
**Bar Chart:** Top 10 Political Action Committees (PACs) by total receipts.  
**Key Takeaway:** Democratic fundraising appears to be more **PAC-driven**, especially at smaller contribution levels.  
**Design Note:** **Dashed outlines** were used for Republican PACs to ensure accessibility in **black-and-white printing**.
""")


st.header("6. Histogram of Individual Donations")

st.image("fig7.png",use_container_width=True)

st.markdown("""
**Bar Chart:** Histogram of individual donations, plotted on a log-log scale to capture the wide variance in contribution amounts.  
**Key Takeaway:** Most donations are less than $5,000 for both parties, but Republicans show a higher concentration of large donors.  
**Design Note:** Log-log scaling was used to compress the wide data range while still revealing underlying distribution trends.
""")


st.header("7. Small Donation Distribution")

st.image("fig8.png",use_container_width=True)

st.markdown("""
**Density Plot:** Kernel Density Estimate (KDE) focusing on donations under $100, plotted on a linear scale for clearer interpretation of small-value contributions.  

**Key Takeaways:**  
- A visible peak occurs around $25 for both parties, with a consistent decline as donation size increases.  
- A noticeable drop around $75 suggests psychological donation thresholds at common values like $5, $10, $25, $50, and $100.

**Design Note:** KDE was chosen instead of a histogram to provide a smoother view of overall distribution trends.
""")





st.header("8. Top Occupations Driving Political Contributions")

st.image("fig9.png", use_container_width=True)

st.markdown("""
**Bar Chart:** Displays the top occupations contributing to political campaigns.  
**Key Takeaway:** Retired individuals and unemployed donors contribute the most overall, followed by CEOs and attorneys. Republicans generally lead in high-value occupation categories.  
**Design Note:** Bars are color-coded to distinguish between political parties and contribution sources clearly.
""")



st.header('9. Party-Wise Fundraising and Candidate Financial Strength')

@st.cache_data
def get_candidate_columns():
    return [
        "CAND_ID", "CAND_NAME", "CAND_ICI", "PTY_CD", "CAND_PTY_AFFILIATION",
        "TTL_RECEIPTS", "TRANS_FROM_AUTH", "TTL_DISB", "TRANS_TO_AUTH",
        "COH_BOP", "COH_COP", "CAND_CONTRIB", "CAND_LOANS", "OTHER_LOANS",
        "CAND_LOAN_REPAY", "OTHER_LOAN_REPAY", "DEBTS_OWED_BY", "TTL_INDIV_CONTRIB",
        "CAND_OFFICE_ST", "CAND_OFFICE_DISTRICT", "SPEC_ELECTION", "PRIM_ELECTION",
        "RUN_ELECTION", "GEN_ELECTION", "GEN_ELECTION_PERCENT",
        "OTHER_POL_CMTE_CONTRIB", "POL_PTY_CONTRIB", "CVG_END_DT",
        "INDIV_REFUNDS", "CMTE_REFUNDS"
    ]


# Load candidate data
columns = get_candidate_columns()
df = pd.read_csv("weball22.txt", sep="|", header=None, names=columns, low_memory=True)

# Filter and clean data
df = df[(df['CAND_NAME'].notna()) & (df['TTL_RECEIPTS'] > 0)].copy()

# Party mapping
party_map = {
    'DEM': 'Democratic', 'DFL': 'Democratic',
    'REP': 'Republican', 'GOP': 'Republican',
    'IND': 'Independent',
    'LIB': 'Libertarian', 'GRE': 'Green',
    'CON': 'Constitution', 'NPA': 'No Party Affiliation',
    'OTH': 'Other', 'UUP': 'United Utah Party'
}
df['Party'] = df['CAND_PTY_AFFILIATION'].map(party_map).fillna('Other')

# Calculate Health Score
df['Health_Score'] = ((df['TTL_RECEIPTS'] + df['COH_COP']) - df['TTL_DISB'])
df['Health_Score'] = df['Health_Score'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Plot treemap
fig = px.treemap(
    df,
    path=['Party', 'CAND_NAME'],
    values='TTL_RECEIPTS',
    color='Health_Score',
    hover_data={
        'TTL_RECEIPTS': ':,.0f',
        'COH_COP': ':,.0f',
        'Health_Score': ':.0f'
    },
    color_continuous_scale='Viridis',
    width=1000,
    height=600
)

fig.update_layout(
    margin=dict(t=50, l=25, r=25, b=25),
    coloraxis_colorbar=dict(
        title="Health Score",
        tickformat=",.0f"
    )
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Treemap:** Visualizes fundraising totals and financial strength by candidate, grouped by party.  
**Key Takeaway:** Financial health varies significantly among candidates within each party. Larger blocks represent higher fundraising totals, while darker colors indicate stronger cash-on-hand performance.  
**Design Note:** The nested structure first separates by party, then by candidate. The *Viridis* colormap was selected for its clarity and accessibility.
""")


# --- 4. Radar Chart: Party Receipts by Top States ---
# --- Radar Chart: DEM vs REP Financial Profile for a State ---
st.header("10. Comparing Financial Profiles of Democrats vs Republicans")

# First, compute state‐party aggregates for your metrics
metrics = ['TTL_RECEIPTS', 'TTL_DISB', 'COH_BOP', 'COH_COP']
state_party = (
    cand_current[cand_current['CAND_OFFICE_ST']=='TX']
    .groupby('CAND_PTY_AFFILIATION')[metrics]
    .sum()
    .reset_index()
)
# Compute net cash
state_party['Net_Cash'] = state_party['COH_COP'] - state_party['COH_BOP']

# Define the categories in the order you want them around the radar
categories = ['TTL_RECEIPTS','TTL_DISB','COH_BOP','COH_COP','Net_Cash']

# Extract values for each party, closing the loop
dem = state_party[state_party['CAND_PTY_AFFILIATION']=='DEM']
rep = state_party[state_party['CAND_PTY_AFFILIATION']=='REP']

dem_vals = dem[categories].iloc[0].tolist()
rep_vals = rep[categories].iloc[0].tolist()
# close the loop
dem_vals += [dem_vals[0]]
rep_vals += [rep_vals[0]]
cats_loop = categories + [categories[0]]

# Build the radar
fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=dem_vals,
    theta=cats_loop,
    fill='toself',
    name='Democrats',
    line_color='blue',
))
fig.add_trace(go.Scatterpolar(
    r=rep_vals,
    theta=cats_loop,
    fill='toself',
    name='Republicans',
    line_color='red',
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            # adjust max if you need tighter/fixed scale
            range=[0, max(dem_vals + rep_vals)]  
        )
    ),
    showlegend=True
)

# And render it with Streamlit
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Radar Chart:** Displays the financial profile of each major political party across key metrics such as starting cash, donations, spending, and ending cash.  

**Key Takeaways:**  
- Democrats started with less cash but received more donations.  
- Despite lower initial funds, they ended the cycle with stronger cash reserves.  
- Republicans spent a larger portion of their funds and ended with less than they began with.  

**Design Note:**  
A simple two-party radar plot was used to emphasize contrasts in financial behavior, allowing for clear visual comparison across all metrics.
""")



# --- Topo for Altair maps ---
us_states = alt.topo_feature(data.us_10m.url, 'states')

# State to FIPS mapping
state_to_fips = {
    'AL':1,'AK':2,'AZ':4,'AR':5,'CA':6,'CO':8,'CT':9,'DE':10,'DC':11,
    'FL':12,'GA':13,'HI':15,'ID':16,'IL':17,'IN':18,'IA':19,'KS':20,
    'KY':21,'LA':22,'ME':23,'MD':24,'MA':25,'MI':26,'MN':27,'MS':28,
    'MO':29,'MT':30,'NE':31,'NV':32,'NH':33,'NJ':34,'NM':35,'NY':36,
    'NC':37,'ND':38,'OH':39,'OK':40,'OR':41,'PA':42,'RI':44,'SC':45,
    'SD':46,'TN':47,'TX':48,'UT':49,'VT':50,'VA':51,'WA':53,'WV':54,
    'WI':55,'WY':56
}

# --- 1. House Race Competitiveness by State ---
st.header('11. House Race Competitiveness')
house = master[master['CAND_OFFICE']=='H']
state_comp = house.groupby('CAND_OFFICE_ST').size().reset_index(name='num_candidates')
# Add district counts and most competitive district if you have data
state_comp['id'] = state_comp['CAND_OFFICE_ST'].map(state_to_fips)
chor_house = alt.Chart(us_states).mark_geoshape(
    stroke='white', strokeWidth=0.5
).encode(
    color=alt.Color('num_candidates:Q', title='House Candidates', scale=alt.Scale(scheme='viridis')),
    tooltip=[
        alt.Tooltip('num_candidates:Q', title='House Candidates')
    ]
).transform_lookup(
    lookup='id',
    from_=alt.LookupData(state_comp, 'id', ['num_candidates'])
).project('albersUsa').properties(width=800, height=400)
st.altair_chart(chor_house, use_container_width=True)

st.markdown("""
**Choropleth Map:** Shows the competitiveness of U.S. House races by visualizing the number of candidates per state.  

**Key Takeaways:**  
- States with higher populations tend to have more candidates overall.  
- Battleground states such as **Georgia**, **Florida**, **Ohio**, **Pennsylvania**, and **Wisconsin** attract a significantly higher number of challengers.  

**Design Note:**  
A choropleth map was used to maximize geographic clarity, and the *Viridis* colormap was selected to reduce artifacting and maintain visual consistency.
""")



# --- 1. Senate Race Competitiveness by State ---
st.header('12. Senate Race Competitiveness')
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

st.markdown("""
**Choropleth Map:** Visualizes the competitiveness of Senate races by state based on the number of candidates.  

**Key Takeaways:**  
- Swing states appear prominently once again, but even traditionally “safe” states show surprising levels of competition.  
- Senate races overall exhibit **greater competitiveness** than House races.  

**Design Note:**  
The map uses the same visual style as the House race choropleth to enable easy side-by-side comparison.
""")




# --- 5. Choropleth: Change in Individual Donations by State ---
st.header("13. State-Level Shifts in Individual Donations")
old_tot = cand_old.groupby("CAND_OFFICE_ST")["DON_OLD"].sum().reset_index()
new_tot = cand_new.groupby("CAND_OFFICE_ST")["DON_NEW"].sum().reset_index()
change_df = old_tot.merge(new_tot, on="CAND_OFFICE_ST", how="outer").fillna(0)
change_df['CHANGE'] = change_df['DON_NEW'] - change_df['DON_OLD']
# Map to FIPS for Altair
change_df['id'] = change_df['CAND_OFFICE_ST'].map(state_to_fips)
chor_change = alt.Chart(us_states).mark_geoshape(
    stroke='white', strokeWidth=0.5
).encode(
    color=alt.Color(
        'CHANGE:Q', title='Δ Individual Donations',
        scale=alt.Scale(
            domain=[change_df['CHANGE'].min(), change_df['CHANGE'].max()],
            range=px.colors.diverging.RdYlGn
        )
    ),
    tooltip=[alt.Tooltip('CHANGE:Q', title='Δ Donations')]
).transform_lookup(
    lookup='id',
    from_=alt.LookupData(change_df, 'id', ['CHANGE'])
).project('albersUsa').properties(width=800, height=400)
st.altair_chart(chor_change, use_container_width=True)

st.markdown("""
**Choropleth Map:** Shows the change in individual political contributions at the state level between election cycles.  

**Key Takeaway:**  
Key battleground states such as **Florida (FL)**, **Georgia (GA)**, and **Pennsylvania (PA)** experienced significant increases in individual donations, indicating rising political engagement in these regions.  

**Design Note:**  
A **diverging color scale** from green (increases) to red (decreases) draws immediate attention to states with the most notable shifts, highlighting regional trends in donor behavior.
""")



st.header("14. Net Cash Reserves of Candidates by State")
coh_df = cand_current.groupby('CAND_OFFICE_ST').agg({'COH_BOP':'sum','COH_COP':'sum'}).reset_index()
coh_df['NET_COH'] = coh_df['COH_COP'] - coh_df['COH_BOP']
# Map to FIPS for Altair
coh_df['id'] = coh_df['CAND_OFFICE_ST'].map(state_to_fips)
chor_coh = alt.Chart(us_states).mark_geoshape(
    stroke='white', strokeWidth=0.5
).encode(
    color=alt.Color(
        'NET_COH:Q', title='Net Cash On Hand',
        scale=alt.Scale(
            domain=[coh_df['NET_COH'].min(), coh_df['NET_COH'].max()],
            range=px.colors.sequential.Inferno
        )
    ),
    tooltip=[alt.Tooltip('NET_COH:Q', title='Net Cash On Hand')]
).transform_lookup(
    lookup='id',
    from_=alt.LookupData(coh_df, 'id', ['NET_COH'])
).project('albersUsa').properties(width=800, height=400)
st.altair_chart(chor_coh, use_container_width=True)

st.markdown("""
**Choropleth Map:** Visualizes the net cash reserves of all political candidates, aggregated by state.  

**Key Takeaway:**  
Most states maintained **net cash balances close to zero**, with notable outliers in **California (CA)**, **South Carolina (SC)**, and **New York (NY)**.  
Interestingly, **CA and NY**, despite their demographic similarities, exhibited **opposing financial profiles** in terms of candidate cash health.  

**Design Note:**  
A **dark-to-light shading** scheme was used to indicate relative financial strength across states.
""")





# --- 7. Choropleth: Total Disbursements by State ---
st.header("15. Total Fundraising Amounts by State")
receipts_df = cand_current.groupby('CAND_OFFICE_ST')['TTL_RECEIPTS'].sum().reset_index()
receipts_df['id'] = receipts_df['CAND_OFFICE_ST'].map(state_to_fips)
chor_receipts = alt.Chart(us_states).mark_geoshape(stroke='white', strokeWidth=0.5).encode(
    color=alt.Color('TTL_RECEIPTS:Q', title='Total Receipts',
                    scale=alt.Scale(domain=[receipts_df['TTL_RECEIPTS'].min(), receipts_df['TTL_RECEIPTS'].max()],
                                    range=px.colors.sequential.Viridis)),
    tooltip=[alt.Tooltip('TTL_RECEIPTS:Q', title='Total Receipts')]
).transform_lookup(
    lookup='id', from_=alt.LookupData(receipts_df, 'id', ['TTL_RECEIPTS'])
).project('albersUsa').properties(width=800, height=400)
st.altair_chart(chor_receipts, use_container_width=True)

st.markdown("""
**Choropleth Map:** Depicts the total amount of political fundraising by state.  

**Key Takeaway:**  
**California**, **Texas**, and **Florida** once again lead the nation in total political contributions, while smaller states raise significantly less overall.  

**Design Note:**  
A **sequential color scale** using the *Viridis* colormap highlights the magnitude of total receipts without applying normalization, emphasizing absolute fundraising strength.
""")

