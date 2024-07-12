import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

# Updated Dataset
dataset = [
    (['Abnormal Vaginal Discharge','Painful Urination','Penile Discharge'], 'Chlamydia'),
    (['Genital Sores','Genital Itching'], 'Genital herpes'),
    (['Abnormal Vaginal Discharge','Abdominal Pain','Painful Urination','Penile Discharge','Bleeding','Vomitting','Fever'], 'Gonorrhea'),
    (['Headache','Difficult Swallowing','Fever','Night Sweats','Fatigue','Appetite Loss','Chronic Diarrhea','Rash','Vomitting','Weight Loss','Chronic Cough'], 'HIV/AIDS'),
    (['Genital Warts','Genital Itching', 'Abnormal Vaginal Discharge','Bleeding'], 'HPV'),
    (['Headache','Fatigue','Mild Fever','Painless Sore', 'Rash','Sore Throat','Pathchy Hair Loss','Appetite Loss','Weight Loss'], 'Syphilis'),
]

# Separate symptoms and labels
symptoms = [data[0] for data in dataset]
labels = [data[1] for data in dataset]

# Create symptom vocabulary
symptom_vocabulary = list(set(symptom for symptoms in symptoms for symptom in symptoms))

# Convert symptoms into numerical features using binary indicators
def convert_to_features(data):
    features = []
    for symptoms in data:
        symptom_counts = {symptom: 1 if symptom in symptoms else 0 for symptom in symptom_vocabulary}
        symptom_vector = list(symptom_counts.values())
        features.append(symptom_vector)
    return np.array(features)

# Convert training and testing data into numerical features
features = convert_to_features(symptoms)

# Initialize and train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(features, labels)

# Function to predict disease and visualize results
def predict_disease(symptoms, classifier, labels):
    test_features = convert_to_features([symptoms])
    predicted_label_index = np.argmax(classifier.predict_proba(test_features))
    predicted_label = labels[predicted_label_index]
    predicted_probabilities = classifier.predict_proba(test_features)[0]

    # Sort the probabilities and labels in descending order
    sorted_indices = np.argsort(predicted_probabilities)[::-1]
    sorted_labels = np.array(labels)[sorted_indices]
    sorted_probabilities = predicted_probabilities[sorted_indices]

    # Plot the results
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Bar Chart
    axes[0].barh(sorted_labels, sorted_probabilities)
    axes[0].set_xlabel('Probability')
    axes[0].set_ylabel('Disease')
    axes[0].set_title('Bar Chart: Predicted Disease Probabilities')

    # Pie Chart
    axes[1].pie(sorted_probabilities, labels=sorted_labels, autopct='%1.1f%%')
    axes[1].set_title('Pie Chart: Predicted Disease Probabilities')

    plt.tight_layout()

    # Display the plots in Streamlit
    st.pyplot(fig)

    # Print the predicted disease
    st.header(f"Predicted Disease: {predicted_label}")

    # User input for symptoms
    predicted_label = predicted_label
  
    if predicted_label == 'Chlamydia':
        st.subheader('Chlamydia:')
        st.write("Chlamydia is a bacterial infection that can infect both men and women and can lead to serious reproductive health problems if left untreated")
        # st.write("It is spread through direct contact with the rash or through respiratory droplets.")
    elif predicted_label == 'Genital herpes':
        # st.image('measles-1.jpg')
        st.subheader('Genital herpes:')
        st.write("Genital herpes is a common STI caused by the herpes simplex virus (HSV), with two types: HSV-1 and HSV-2. It is transmitted through sexual contact and can be passed on even without visible sores. Symptoms include painful blisters, itching, and flu-like symptoms. Recurrences can occur triggered by various factors. Diagnosis involves examination and lab tests. While there is no cure, antiviral medications can manage symptoms and reduce outbreaks. Prevention includes safe sex practices, but transmission can still occur. Genital herpes can have emotional impact due to stigma, so seeking support and open communication is important. Consult a healthcare professional for diagnosis and advice.")
        # st.write("It is spread through direct contact with the rash or through respiratory droplets.")
    elif predicted_label == 'Gonorrhea':
        st.subheader("Gonorrhea :") 
        st.write("Gonorrhea is a common sexually transmitted infection (STI) caused by the bacterium Neisseria gonorrhoeae. It can infect both men and women and is primarily transmitted through sexual contact, including vaginal, anal, or oral sex.")
    elif predicted_label == 'HIV/AIDS':
        st.subheader("HIV/AIDS :") 
        st.write("HIV (Human Immunodeficiency Virus) is a virus that attacks the immune system, gradually weakening it over time. AIDS (Acquired Immunodeficiency Syndrome) is the advanced stage of HIV infection when the immune system is severely compromised.")
    elif predicted_label == 'HPV':
        st.subheader("Human Papillomavirus (HPV):") 
        st.write("Human Papillomavirus (HPV) is a common sexually transmitted infection that can affect both men and women. It is primarily transmitted through sexual contact, including vaginal, anal, and oral sex.")
    elif predicted_label == 'Syphilis': 
        st.subheader("Syphilis:") 
        st.write("Syphilis is a sexually transmitted infection (STI) caused by the bacterium Treponema pallidum. It can affect various parts of the body and progresses through different stages if left untreated")
    else:
        st.write("Unknown disease.")







# Page 1: Home
def home():
    st.title("Diagnostic System for Sexually Transmitted Diseases")
    st.write("Enter the symptoms below:")
    # User input for symptoms
    user_symptoms = st.multiselect("Select Symptoms", symptom_vocabulary)

    if st.button("Diagnose"):
        if len(user_symptoms) > 0:
            predict_disease(user_symptoms, classifier, labels)
        else:
            st.warning("Please select at least one symptom.")
            
# Page 2: Data and Mapping
def data_mapping():
    st.title("Data and Mapping")
    st.write("Here is the list of data and how they map to diseases:")
    for i, data in enumerate(dataset):
        symptoms, disease = data
        st.write(f"{i+1}. Symptoms: {symptoms}, Disease: {disease}")







# Page 3: Add Symptoms and Diseases
def add_symptoms_and_diseases():
    st.title("Add Symptoms and Diseases")
    st.write("Enter the symptoms and corresponding disease below:")

    # User input for symptoms and disease
    user_symptoms = st.multiselect("Enter Symptoms", symptom_vocabulary)
    user_disease = st.text_input("Enter Disease")

    if st.button("Add"):
        if len(user_symptoms) > 0 and user_disease != "":
            # Add the new symptoms and disease to the dataset
            dataset.append((user_symptoms, user_disease))
            st.success("Symptoms and Disease added successfully!")
        else:
            st.warning("Please enter at least one symptom and disease.")



# Page 4: Add Symptoms
def Diseases_info():
    st.title("Info about Diseases")
    # st.write("select diseases from list")

    # User input for symptoms
    predicted_label = st.selectbox('Select Diseases',labels)
  
    # # User input for symptoms
    # predicted_label = predicted_label
  
    if predicted_label == 'Chlamydia':
        st.subheader('Chlamydia:')
        st.write("Chlamydia is a bacterial infection that can infect both men and women and can lead to serious reproductive health problems if left untreated")
        # st.write("It is spread through direct contact with the rash or through respiratory droplets.")
    elif predicted_label == 'Genital herpes':
        # st.image('measles-1.jpg')
        st.subheader('Genital herpes:')
        st.write("Genital herpes is a common STI caused by the herpes simplex virus (HSV), with two types: HSV-1 and HSV-2. It is transmitted through sexual contact and can be passed on even without visible sores. Symptoms include painful blisters, itching, and flu-like symptoms. Recurrences can occur triggered by various factors. Diagnosis involves examination and lab tests. While there is no cure, antiviral medications can manage symptoms and reduce outbreaks. Prevention includes safe sex practices, but transmission can still occur. Genital herpes can have emotional impact due to stigma, so seeking support and open communication is important. Consult a healthcare professional for diagnosis and advice.")
        # st.write("It is spread through direct contact with the rash or through respiratory droplets.")
    elif predicted_label == 'Gonorrhea':
        st.subheader("Gonorrhea :") 
        st.write("Gonorrhea is a common sexually transmitted infection (STI) caused by the bacterium Neisseria gonorrhoeae. It can infect both men and women and is primarily transmitted through sexual contact, including vaginal, anal, or oral sex.")
    elif predicted_label == 'HIV/AIDS':
        st.subheader("HIV/AIDS :") 
        st.write("HIV (Human Immunodeficiency Virus) is a virus that attacks the immune system, gradually weakening it over time. AIDS (Acquired Immunodeficiency Syndrome) is the advanced stage of HIV infection when the immune system is severely compromised.")
    elif predicted_label == 'HPV':
        st.subheader("Human Papillomavirus (HPV):") 
        st.write("Human Papillomavirus (HPV) is a common sexually transmitted infection that can affect both men and women. It is primarily transmitted through sexual contact, including vaginal, anal, and oral sex.")
    elif predicted_label == 'Syphilis': 
        st.subheader("Syphilis:") 
        st.write("Syphilis is a sexually transmitted infection (STI) caused by the bacterium Treponema pallidum. It can affect various parts of the body and progresses through different stages if left untreated")
    else:
        st.write("Unknown disease.")




# Page 4: Add Symptoms
def add_symptoms():
    st.title("Add Symptoms")
    st.write("Enter the new symptoms below:")

    # User input for symptoms
    user_symptoms = st.text_area("Enter Symptoms (separated by commas)")

    if st.button("Add"):
        if user_symptoms != "":
            # Split the user input into individual symptoms
            new_symptoms = [symptom.strip() for symptom in user_symptoms.split(",")]

            # Update the symptom vocabulary
            symptom_vocabulary.extend(new_symptoms)
            symptom_vocabulary = list(set(symptom_vocabulary))

            st.success("Symptoms added successfully!")
        else:
            st.warning("Please enter at least one symptom.")


















# Main App
def main():
    pages = {
        "Home": home,
        "Data and Mapping": data_mapping,
        "Diseases-Info": Diseases_info,
    }



    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    page = pages[selection]
    page()

if __name__ == "__main__":
    main()
