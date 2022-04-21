import streamlit as st
import pandas as pd

st.set_page_config(page_title='Football Prediction Analysis',layout="wide")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)



final_data = pd.read_csv("./data/final_data.csv")
df = pd.read_csv("./data/this_week.csv")

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
