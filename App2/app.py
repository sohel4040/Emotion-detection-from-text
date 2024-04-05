import streamlit as st 
import altair as alt #Complex visualizations(clear syntax)
import plotly.express as px #simple visualizations
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
	st.title("Emotion Detection from Text")
	menu = ["Home","Dashboard","About"]
	choice = st.sidebar.selectbox("Menu",menu)
	create_page_visited_table()
	create_emotionclf_table()
	if choice == "Home":
		add_page_visited_details("Home",datetime.now())
		st.markdown("#### Discover the Emotions Hidden in Your Words")
		# st.markdown("*Discover the Emotions Hidden in Your Words*", unsafe_allow_html=True)

		with st.form(key='emotion_clf_form'):
			raw_text = st.text_area("Type Here")
			submit_text = st.form_submit_button(label='Submit')
			
		if submit_text and not raw_text.strip():
			st.warning("Please Enter some text above and click 'Submit' to analyze emotions.")

		elif submit_text:
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
				proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
				proba_df_clean = proba_df.T.reset_index()
				proba_df_clean.columns = ["emotions","probability"]

				fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
				st.altair_chart(fig,use_container_width=True)



	elif choice == "Dashboard":
		add_page_visited_details("Monitor",datetime.now())
		st.subheader("Dashboard")

		with st.expander("Page Metrics"):
			page_visited_details = pd.DataFrame(view_all_page_visited_details(),columns=['Pagename','Time_of_Visit'])
			st.dataframe(page_visited_details)	

			pg_count = page_visited_details['Pagename'].value_counts().rename_axis('Pagename').reset_index(name='Counts')
			c = alt.Chart(pg_count).mark_bar().encode(x='Pagename',y='Counts',color='Pagename')
			st.altair_chart (c,use_container_width=True)	

			p = px.pie(pg_count,values='Counts',names='Pagename')
			st.plotly_chart(p,use_container_width=True)

		with st.expander('Emotion Classifier Metrics'):
			df_emotions = pd.DataFrame(view_all_prediction_details(),columns=['Rawtext','Prediction','Probability','Time_of_Visit'])
			# st.dataframe(df_emotions)

			prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
			pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction',y='Counts',color='Prediction')
			st.altair_chart(pc,use_container_width=True)	

		 

	else:
		st.subheader("About")
		add_page_visited_details("About",datetime.now())
		st.markdown("*Emotion detection from text* is an application which gives you prediction based on emotions expressed via text.")
		

	footer_text = "© 2024 , Desinged and Developed by School of Computer Engineering (Group No 1). All rights reserved.".format(datetime.now().year)
	st.markdown(footer_text)


if __name__ == '__main__':
	main()