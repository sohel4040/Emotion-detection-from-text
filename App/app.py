#All packages to be used
import streamlit as st 
import altair as alt
import plotly.express as px 
import pandas as pd 
import numpy as np 
from datetime import datetime
import joblib 


pipe_lr = joblib.load(open("models/emotion_classifier_pipe.pkl","rb"))

from track_utils import create_page_visited_table,add_page_visited_details,view_all_page_visited_details,add_prediction_details,view_all_prediction_details,create_emotionclf_table

def predict_emotions(docx):
	results = pipe_lr.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}


# Main Application
def main():
	st.title("Emotion Detection from text")
	menu = ["Main Page","Dashboard","About"]
	choice = st.sidebar.selectbox("Navigation Menu",menu)
	create_page_visited_table()
	create_emotionclf_table()
	if choice == "Main Page":
		add_page_visited_details("Main Page",datetime.now())

		with st.form(key='emotion_clf_form'):
			raw_text = st.text_area("Type Here")
			submit_text = st.form_submit_button(label='Submit')
		if submit_text :
			if raw_text == "" : 
				st.warning("Please Enter some text to continue")
			else : 
				col1,col2  = st.columns(2)

				prediction = predict_emotions(raw_text)
				probability = get_prediction_proba(raw_text)
				
				add_prediction_details(raw_text,prediction,np.max(probability),datetime.now())

				with col1:
					st.success("Original Text")
					st.write(raw_text)

					st.success("Prediction")
					emoji_icon = emotions_emoji_dict[prediction]
					st.write("{}:{}".format(prediction,emoji_icon))
					st.write("Confidence:{}".format(np.max(probability)))

				with col2:
					st.success("Prediction Probability")
					# st.write(probability)
					proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
					# st.write(proba_df.T)
					proba_df_clean = proba_df.T.reset_index()
					proba_df_clean.columns = ["emotions","probability"]

					fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
					st.altair_chart(fig,use_container_width=True)

	else:
		st.subheader("About")
		add_page_visited_details("About",datetime.now())
		st.markdown("*Emotion detection from text* is an application which gives you prediction based on emotions expressed via text.")


if __name__ == '__main__':
	main()
