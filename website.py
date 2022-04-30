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

### DICTIONARIES ###

types = ["Mean","Total","Median","Maximum","Minimum"]

label_attr_dict = {"Goals":"Goals","Halftime Goals":"HT_Goals","Shots On Target":"Shots_Target", "Fouls":"Fouls","Corners":"Corner","Yellow Cards":"Yellow","Red Cards": "Red"}
label_attr_dict_teams = {"Goals Scored":"Goals","Goals Received":"Goals_Received","Halftime Goals Scored":"HT_Goals","Halftime Goals Received":"HT_Goals_Received","Shots on opposing Goal":"Shots_Target","Shots on own Goal":"Shots_Target_Received", "Fouls":"Fouls", "Got Fouled":"Got_Fouled","Corners":"Corner", "Corners Received":"Corner_Received", "Yellow Cards":"Yellow", "Yellow Cards Received":"Yellow_Received","Red Cards":"Red","Red Cards Received":"Red_Received"}
dict_correlation = {"Goals":"Delta_Goals", "Halftime Goals":"Delta_HT_Goals","Shots On Target":"Delta_Shots_On_Target","Fouls":"Delta_Foul","Corner":"Delta_Corner","Yellow Cards":"Delta_Yellow","Red Cards":"Delta_Red"}  

### INTRODUCTION ###

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

### NEXT WEEK PREDICTIONS ###

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


def group_measure_by_attribute(aspect,attribute,measure):
    df_data = df_total
    df_return = pd.DataFrame()
    if(measure == "Total"):
        df_return = df_data.groupby([aspect]).sum()            
    
    if(measure == "Mean"):
        df_return = df_data.groupby([aspect]).mean()
        
    if(measure == "Median"):
        df_return = df_data.groupby([aspect]).median()

    if(measure == "Minimum"):
        df_return = df_data.groupby([aspect]).min()
    
    if(measure == "Maximum"):
        df_return = df_data.groupby([aspect]).max()
    
    df_return["aspect"] = df_return.index
    if aspect == "team":
        df_return = df_return.sort_values(by=[attribute], ascending = False)
    return df_return


### PLOTS - TEAM ANALYSIS ###

def plot_x_per_team(attr,measure): 
    rc = {'figure.figsize':(8,4.5),
          'axes.facecolor':'white',
          'axes.edgecolor': 'white',
          'axes.labelcolor': 'black',
          'figure.facecolor': 'white',
          'patch.edgecolor': 'white',
          'text.color': 'black',
          'xtick.color': 'black',
          'ytick.color': 'black',
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

### PLOTS - SEASON ANALYSIS ###

def plot_x_per_season(attr,measure):
    rc = {'figure.figsize':(8,4.5),
          'axes.facecolor':'white',
          'axes.edgecolor': 'white',
          'axes.labelcolor': 'black',
          'figure.facecolor': 'white',
          'patch.edgecolor': 'white',
          'text.color': 'black',
          'xtick.color': 'black',
          'ytick.color': 'black',
          'font.size' : 8,
          'axes.labelsize': 12,
          'xtick.labelsize': 8,
          'ytick.labelsize': 12}
    
    plt.rcParams.update(rc)
    fig, ax = plt.subplots()

    attribute = label_attr_dict[attr]
    df_plot = pd.DataFrame()
    df_plot = group_measure_by_attribute("Season",attribute,measure)
    ax = sns.barplot(x="aspect", y=attribute, data=df_plot, color = "#0b70f3")

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

### PLOT - CORRELATION ANALYSIS ###

def plt_attribute_correlation(aspect1, aspect2):
    # df_plot = df_data_filtered
    rc = {'figure.figsize':(5,5),
          'axes.facecolor':'white',
          'axes.edgecolor': 'white',
          'axes.labelcolor': 'black',
          'figure.facecolor': 'white',
          'patch.edgecolor': 'white',
          'text.color': 'black',
          'xtick.color': 'black',
          'ytick.color': 'black',
          'font.size' : 8,
          'axes.labelsize': 12,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8}
    plt.rcParams.update(rc)
    fig, ax = plt.subplots()
    asp1 = dict_correlation[aspect1]
    asp2 = dict_correlation[aspect2]
    if(corr_type=="Regression Plot (Recommended)"):
        ax = sns.regplot(x=asp1, y=asp2, x_jitter=.1, data=df_total, color = '#f21111',scatter_kws={"color": "#0b70f3"},line_kws={"color": "#c2dbfc"})
    if(corr_type=="Standard Scatter Plot"):
        ax = sns.scatterplot(x=asp1, y=asp2, data=df_total, color = '#0b70f3')
    ax.set(xlabel = aspect1, ylabel = aspect2)
    st.pyplot(fig, ax)


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
with row5_2:
    plot_x_per_team(plot_x_per_team_selected, plot_x_per_team_type)


### SEASON ANALYSIS ###

row6_spacer1, row6_1, row6_spacer2 = st.columns((.2, 7.1, .2))
with row6_1:
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.subheader('Analysis Per Season')
row7_spacer1, row7_1, row7_spacer2, row7_2, row7_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
with row7_1:
    st.markdown('Statistics for all seasons of the EPL analysed on various measures.')
    st.markdown('Answers question such as: Which season had teams score the most goals? What is the total number of red cards received in each season?')
    plot_x_per_season_selected = st.selectbox ("Which attribute do you want to analyze?", list(label_attr_dict.keys()), key = 'attribute_season')
    plot_x_per_season_type = st.selectbox ("Which measure do you want to analyze?", types, key = 'measure_season')
with row7_2:
    plot_x_per_season(plot_x_per_season_selected,plot_x_per_season_type)

### CORRELATION ANALYSIS ###

corr_plot_types = ["Regression Plot (Recommended)","Standard Scatter Plot"]

row10_spacer1, row10_1, row10_spacer2 = st.columns((.2, 7.1, .2))
with row10_1:
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.subheader('Correlation of Game Stats')
row11_spacer1, row11_1, row11_spacer2, row11_2, row11_spacer3  = st.columns((.2, 2.3, .4, 3, .2))
with row11_1:
    st.markdown("Investigate the correlation of attributes.")
    st.markdown("(📈) - Postive Correlation") 
    st.markdown("(📉)- Negative Correlation")
    st.markdown("Answers questions such as:  Do teams that have more shots on targets have more corners? Do the team's shots on target say anything about the probablility of goals")    
    corr_type = st.selectbox ("What type of correlation plot do you want to see?", corr_plot_types)
    y_axis_aspect2 = st.selectbox ("Which attribute do you want on the y-axis?", list(dict_correlation.keys()))
    x_axis_aspect1 = st.selectbox ("Which attribute do you want on the x-axis?", list(dict_correlation.keys()))
with row11_2:
    plt_attribute_correlation(x_axis_aspect1, y_axis_aspect2)