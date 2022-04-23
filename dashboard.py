import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

st.set_page_config(page_title='Football Prediction Analysis',layout="wide")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

### Data Import ###

final_data = pd.read_csv("./data/final_table.csv")
df = pd.read_csv("./data/this_week.csv")
df_total = pd.read_csv("./data/df_total.csv")

types = ["Mean","Total","Median","Maximum","Minimum"]

label_attr_dict_teams = {"Goals Scored":"Goals","Goals Received":"Goals_Received","Halftime Goals Scored":"HT_Goals","Halftime Goals Received":"HT_Goals_Received","Shots on opposing Goal":"Shots_Target","Shots on own Goal":"Shots_Target_Received", "Fouls":"Fouls", "Got Fouled":"Got_Fouled","Corners":"Corner", "Corners Received":"Corner_Received", "Yellow Cards":"Yellow", "Yellow Cards Received":"Yellow_Received","Red Cards":"Red","Red Cards Received":"Red_Received"}


row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))
with row0_1:
    st.title('Football Stats And Prediction Analysis')

row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3.2, .1))
with row3_1:
    st.markdown("English Premier League (EPL) is a football league in which 20 teams play 2 matches with every other team in their league - one at their home stadium and the opponent's home stadium.")
    st.markdown("Each match has 3 possible outcomes - home team wins, away team wins or the match ends up in a draw.")
    st.markdown("This project is based on the prediction of the Full Time Result (FTR) of upcoming matches based on the factors analysed by a machine learning model on past data.")
    see_data = st.expander('You can click here to see the raw data first 👉')
    with see_data:
        st.dataframe(data=final_data.reset_index(drop=True))

row6_spacer1, row6_1, row6_spacer2 = st.columns((.2, 7.1, .2))
with row6_1:
    st.markdown(" ")
    st.subheader("Predictions for next week:")
    st.markdown(" ")

row2_spacer1, row2_1, row2_spacer2, row2_2, row2_spacer3, row2_3, row2_spacer4, row2_4, row2_spacer5   = st.columns((.2, 1.6, .2, 1.6, .2, 1.6, .2, 1.6, .2))

with row2_1:
    num_games = df.HomeTeam.nunique()
    st.metric(label="Total Matches ⚽", value=num_games)
with row2_2:
    h = df['Full Time Result'].str.count("H").sum()
    st.metric(label="Home Team Wins 🏆", value=h)
with row2_3:
    a = df['Full Time Result'].str.count("A").sum()
    st.metric(label="Away Team Wins 🏆", value=a)
with row2_4:
    d = df['Full Time Result'].str.count("D").sum()
    st.metric(label="Draws 🎗", value=d)

row6_spacer1, row6_1, row6_spacer2 = st.columns((.2, 7.1, .2))
with row6_1:
    st.markdown(" ")
    st.table(data=df.reset_index(drop=True))

#Get Seasons & Teams - methods


def get_unique_seasons_modified(df_total):
    #returns unique season list in the form "Season 13/14" for labels
    unique_seasons = np.unique(df_total['Season']).tolist()
    seasons_modified = []
    for s,season in enumerate(unique_seasons):
        if s==0:
            season = " " + season
        if s==len(unique_seasons)-1:
            season = season + " "
        seasons_modified.append(season.replace("-","/"))
    return seasons_modified

def filter_season(df_total):
    df_filtered_season = pd.DataFrame()
    seasons = np.unique(df_total.Season).tolist() #season list "13-14"
    start_raw = start_season.replace("/","-").replace(" ","") #get raw start season "13-14"
    end_raw = end_season.replace("/","-").replace(" ","") #get raw end season "19-20"
    start_index = seasons.index(start_raw)
    end_index = seasons.index(end_raw)+1
    seasons_selected = seasons[start_index:end_index]
    df_filtered_season = df_total[df_total['Season'].isin(seasons_selected)]
    return df_filtered_season

def get_unique_teams(df_total):
    unique_teams = np.unique(df_total.Team).tolist()
    return unique_teams

def filter_teams(df_total):
    df_filtered_team = pd.DataFrame()
    df_filtered_team = df_total[df_total['Team']]
    return df_filtered_team

def group_measure_by_attribute(aspect,attribute,measure):
    df_return = pd.DataFrame()
    if(measure == "Total"):
        df_return = df_total.groupby([aspect]).sum()            
    
    if(measure == "Mean"):
        df_return = df_total.groupby([aspect]).mean()
        
    if(measure == "Median"):
        df_return = df_total.groupby([aspect]).median()
    
    if(measure == "Minimum"):
        df_return = df_total.groupby([aspect]).min()
    
    if(measure == "Maximum"):
        df_return = df_total.groupby([aspect]).max()
    
    df_return["aspect"] = df_return.index
    if aspect == "team":
        df_return = df_return.sort_values(by=[attribute], ascending = False)
    return df_return

### SELECTION SLIDER - SEASON SELECTOR ###


st.sidebar.markdown("**Select the season range you want to analyze:** 👇")

unique_seasons = get_unique_seasons_modified(df_total)

start_season, end_season = st.sidebar.select_slider('Select the season range you want to include', unique_seasons, value = ["2/3","19/20"])

df_data_filtered_season = filter_season(df_total)

### SELECTION SLIDER - TEAM SELECTOR ###

unique_teams = get_unique_teams(df_total)
# all_teams_selected = st.sidebar.selectbox('Do you want to only include specific teams? If the answer is yes, please check the box below and then select the team(s) in the new field.', ['Include all available teams','Select teams manually (choose below)'])
# if all_teams_selected == 'Select teams manually (choose below)':
#     selected_teams = st.sidebar.multiselect("Select and deselect the teams you would like to include in the analysis. You can clear the current selection by clicking the corresponding x-button on the right", unique_teams, default = unique_teams)
# df_data_filtered = filter_teams(df_total)        
# ### SEE DATA ###
# row6_spacer1, row6_1, row6_spacer2 = st.columns((.2, 7.1, .2))
# with row6_1:
#     st.subheader("Currently selected data:")

    
### PLOTS - TEAM ANALYSIS ###

def plot_x_per_team(attr,measure): 
    rc = {'figure.figsize':(8,4.5),
          'axes.facecolor':'#0e1117',
          'axes.edgecolor': '#0e1117',
          'axes.labelcolor': 'white',
          'figure.facecolor': '#0e1117',
          'patch.edgecolor': '#0e1117',
          'text.color': 'white',
          'xtick.color': 'white',
          'ytick.color': 'white',
          'grid.color': 'grey',
          'font.size' : 8,
          'axes.labelsize': 12,
          'xtick.labelsize': 8,
          'ytick.labelsize': 12}
    
    plt.rcParams.update(rc)
    fig, ax = plt.subplots()
    ### Goals
    attribute = label_attr_dict_teams[attr]
    df_plot = pd.DataFrame()
    df_plot = group_measure_by_attribute("Team",attribute,measure)

    ax = sns.barplot(x="aspect", y=attribute, data=df_plot.reset_index(), color = "#0b70f3")

    y_str = measure + " " + attr + " " + "per Game"
    if measure == "Absolute":
        y_str = measure + " " + attr
    if measure == "Minimum" or measure == "Maximum":
        y_str = measure + " " + attr + " in a Game"
    ax.set(xlabel = "Team", ylabel = y_str)
    plt.xticks(rotation=66,horizontalalignment="right")

    if measure == "Mean": #more accurate result in float dtype instead of int
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'), 
                  (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center',
                   va = 'center', 
                   xytext = (0, 18),
                   rotation = 90,
                   textcoords = 'offset points')
    else:
        for p in ax.patches:
            ax.annotate(format(str(int(p.get_height()))), 
                  (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center',
                   va = 'center', 
                   xytext = (0, 18),
                   rotation = 90,
                   textcoords = 'offset points')
    st.pyplot(fig)

### PLOTS - TEAM ANALYSIS ###

def plot_x_per_season(attr,measure):
    rc = {'figure.figsize':(8,4.5),
          'axes.facecolor':'#0e1117',
          'axes.edgecolor': '#0e1117',
          'axes.labelcolor': 'white',
          'figure.facecolor': '#0e1117',
          'patch.edgecolor': '#0e1117',
          'text.color': 'white',
          'xtick.color': 'white',
          'ytick.color': 'white',
          'grid.color': 'grey',
          'font.size' : 8,
          'axes.labelsize': 12,
          'xtick.labelsize': 8,
          'ytick.labelsize': 12}
    
    plt.rcParams.update(rc)
    fig, ax = plt.subplots()

    attribute = label_attr_dict_teams[attr]
    df_plot = pd.DataFrame()
    df_plot = group_measure_by_attribute("Season",attribute,measure)
    ax = sns.barplot(x="aspect", y=attribute, data=df_plot, color = "#b80606")

    y_str = measure + " " + attr + " " + " per Team"
    if measure == "Absolute":
        y_str = measure + " " + attr
    if measure == "Minimum" or measure == "Maximum":
        y_str = measure + " " + attr + " by a Team"

    plt.xticks(rotation=66,horizontalalignment="right")    
    ax.set(xlabel = "Season", ylabel = y_str)
    if measure == "Mean":
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'), 
                  (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center',
                   va = 'center', 
                   xytext = (0, 15),
                   textcoords = 'offset points')
    else:
        for p in ax.patches:
            ax.annotate(format(str(int(p.get_height()))), 
                  (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center',
                   va = 'center', 
                   xytext = (0, 15),
                   textcoords = 'offset points')
    st.pyplot(fig)


### TEAM ANALYSIS ###

row4_spacer1, row4_1, row4_spacer2 = st.columns((.2, 7.1, .2))
with row4_1:
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.subheader('Analysis Per Team')
row5_spacer1, row5_1, row5_spacer2, row5_2, row5_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
with row5_1:
    st.markdown('Statistics for each of the 40 teams to ever have taken part in the EPL.')
    st.markdown('Answers questions such as: What is the total number of goals scored by a team? Which is the team with the minimum number of foul ever received in the league?')    
    plot_x_per_team_selected = st.selectbox ("Which attribute do you want to analyze?", list(label_attr_dict_teams.keys()), key = 'attribute_team')
    plot_x_per_team_type = st.selectbox ("Which measure do you want to analyze?", types, key = 'measure_team')
    specific_team_colors = st.checkbox("Use team specific color scheme")
with row5_2:
    plot_x_per_team(plot_x_per_team_selected, plot_x_per_team_type)


### SEASON ANALYSIS ###

row6_spacer1, row6_1, row6_spacer2 = st.columns((.2, 7.1, .2))
with row6_1:
    st.subheader('Analysis Per Season')
row7_spacer1, row7_1, row7_spacer2, row7_2, row7_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
with row7_1:
    st.markdown('Statistics for all seasons of the EPL analysed on various measures.')
    st.markdown('Answers question such as: Which season had teams score the most goals? Has the amount of passes per games changed?')    
    plot_x_per_season_selected = st.selectbox ("Which attribute do you want to analyze?", list(label_attr_dict_teams.keys()), key = 'attribute_season')
    plot_x_per_season_type = st.selectbox ("Which measure do you want to analyze?", types, key = 'measure_season')
with row7_2:
    #  if all_teams_selected != 'Select teams manually (choose below)' or selected_teams:
    plot_x_per_season(plot_x_per_season_selected,plot_x_per_season_type)
    # else:
    #     st.warning('Please select at least one team')