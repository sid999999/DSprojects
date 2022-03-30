#core pkgs
import streamlit as st
import altair as alt
import plotly.express as px 

#EDA pkgs
import pandas as pd
import numpy as np
import joblib
import emoji


pipe_lr = joblib.load(open("models/emotion_classifier_pipe_lr_03_2022.pkl","rb"))

#Fxn
def predict_emotions(docx):
	results=pipe_lr.predict([docx])
	return results
def get_prediction_proba(docx):
	results=pipe_lr.predict_proba([docx])
	return results

#emojis
#emotions_emoji_dict={"anger":"U+1F620","disgusting":"U+1F92E","joy":"U+1F604","sadness":"U+1F622","fear":"U+1F631","surprise":"U+1F640","shame":"U+1F648","neutral":"U+1F610"}
#emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}



def main():
	st.title('Emotion Detector')
	menu = ['Home','Monitor','About']
	choice = st.sidebar.selectbox('Menu',menu)

	if choice == 'Home':
		st.subheader('Home-Emotion In Text')

		with st.form(key='emotion_clf_form'):
			raw_text = st.text_area('Type Here')
			submit_text = st.form_submit_button(label='Submit')

		if submit_text:
			col1,col2 = st.columns(2)

			#apply fxn here
			prediction=predict_emotions(raw_text)
			probability = get_prediction_proba(raw_text)

			with col1:
				st.success("Original Text")
				st.write(raw_text)

				st.success("Prediction")
				#emoji_icon = emotions_emoji_dict[prediction]
				st.write(prediction)
				st.write("Confidence:{}".format(np.max(probability)))

			with col2:
				st.success("Prediction Probability")
				st.write(probability)
				proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
				st.write(proba_df.T)
				proba_df_clean = proba_df.T.reset_index()
				proba_df_clean.columns = ["emotions","probability"]

				fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
				st.altair_chart(fig,use_container_width=True)



	elif choice == 'Monitor':
		st.subheader('Monitor App')

	else:
		st.subheader('About')


if __name__ == '__main__':
	main()