import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Phishing Website Detection",
    page_icon="🛡️",
    layout="wide"
)

# Load Trained Model
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")

model = load_model()

# Feature names and allowed values from UCI dataset
features = {
    "having_IP_Address": [-1, 1],
    "URL_Length": [1, 0, -1],
    "Shortining_Service": [1, -1],
    "having_At_Symbol": [1, -1],
    "double_slash_redirecting": [-1, 1],
    "Prefix_Suffix": [-1, 1],
    "having_Sub_Domain": [-1, 0, 1],
    "SSLfinal_State": [-1, 1, 0],
    "Domain_registeration_length": [-1, 1],
    "Favicon": [1, -1],
    "port": [1, -1],
    "HTTPS_token": [-1, 1],
    "Request_URL": [1, -1],
    "URL_of_Anchor": [-1, 0, 1],
    "Links_in_tags": [1, -1, 0],
    "SFH": [-1, 1, 0],
    "Submitting_to_email": [-1, 1],
    "Abnormal_URL": [-1, 1],
    "Redirect": [0, 1],
    "on_mouseover": [1, -1],
    "RightClick": [1, -1],
    "popUpWidnow": [1, -1],
    "Iframe": [1, -1],
    "age_of_domain": [-1, 1],
    "DNSRecord": [-1, 1],
    "web_traffic": [-1, 0, 1],
    "Page_Rank": [-1, 1],
    "Google_Index": [1, -1],
    "Links_pointing_to_page": [1, 0, -1],
    "Statistical_report": [-1, 1]
}

# Application Heading
st.title("🛡️ Phishing Website Detection System")

st.write(
    "Enter website characteristics below "
    "to predict whether a website is phishing or legitimate."
)

st.info(
    "Select the extracted feature values from the UCI "
    "Phishing Websites Dataset."
)

# Feature Input Form
with st.form("phishing_form"):

    st.subheader("Website Feature Input")

    input_values = {}

    columns = st.columns(3)

    for index, (feature, options) in enumerate(features.items()):

        with columns[index % 3]:

            input_values[feature] = st.selectbox(
                feature,
                options=options,
                key=feature
            )

    submitted = st.form_submit_button(
        "🔍 Predict Website",
        use_container_width=True
    )

# Prediction
if submitted:

    input_df = pd.DataFrame(
        [input_values],
        columns=list(features.keys())
    )

    prediction = model.predict(input_df)[0]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == -1:

        st.error("⚠️ Potentially Phishing Website")

        st.warning(
            "The model classified this website as potentially phishing. "
            "Avoid entering sensitive information."
        )

    else:

        st.success("✅ Likely Legitimate Website")

        st.info(
            "The model classified this website as likely legitimate. "
            "This does not guarantee that the website is safe."
        )

    st.write("Prediction Value:", prediction)

    with st.expander("View Submitted Features"):
        st.dataframe(input_df)

st.divider()

st.caption(
    "MCA Section E | Phishing Website Detection Using Machine Learning"
)
st.caption(
    "Disclaimer: Predictions are model-based estimates, "
    "not a guarantee of website safety."
)
