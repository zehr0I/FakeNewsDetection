import streamlit as st
import requests

def main():
    st.title("Fake news Classification APP")

    user_input=st.text_area("Enter the news text:")
    data={"news": user_input}

    if st.button("classify"):
        if user_input:
            response=requests.post("http://localhost:8501/predict",json=data)

            if response.status_code ==200:
                prediction =response.json()["prediction"]

                if prediction==0:
                    final_prediction="Real News"
                else:
                    final_prediction="Fake News"
                
                st.success(f"The news is : {final_prediction}")
            
            else:
                st.error("Error while making prediction")
            
if __name__=="__main__":
    main()