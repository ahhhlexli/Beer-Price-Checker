# CONDENSED PROGRAM FOR STREAMLIT SHARE DEPLOYMENT

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps
import time
import pickle

#COUNTER FOR IMAGES CORRECTLY IDENTIFIED
count_pickle = pickle.load( open( "counter.p", "rb" ) )

st.set_page_config(
    page_title="Beer Price Checker!",
    layout="wide",
    initial_sidebar_state="expanded",
    )

## Sidebar
st.sidebar.subheader("Brands of Beer Trained")
st.sidebar.text("""
    Asahi
    Blue Girl
    Blue Ice
    Budweiser
    Carlsberg
    Corona Extra
    Guinness
    Heineken
    Kingway
    Kirin
    San Mig
    San Miguel
    Skol Beer
    Sol
    Stella Artois
    Tiger
    Tsingtao Beer
    Yanjing Beer""")
st.sidebar.subheader("Example of Good Image")

example = Image.open("./logo/coronasample.jpeg").resize([168,224])
st.sidebar.image(example)

##

@st.cache
def load_csv():
    return pd.read_csv("df_price.csv",header=0,index_col=0)

def temp_df():
    return df[df.Brand==predicted_class.title()]

@st.cache(suppress_st_warning=True)
def load_model(original_image):

    fixed_image = ImageOps.exif_transpose(original_image)
    image_to_resize = img_to_array(fixed_image)

    resized = tf.image.resize(image_to_resize, [224, 168], method="bilinear",antialias=False)
    img_array = tf.keras.preprocessing.image.img_to_array(resized)

    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    # img_show = tf.squeeze(img_array , axis=None, name=None)
    predicted_class = class_names[np.argmax(score)]

    # st.write(f"This image most likely belongs to {predicted_class}")

    percentages = [i * 100 for i in predictions.tolist()[0]]
    results = zip(class_names, percentages)
    sorted_by_second = sorted(results, key=lambda tup: tup[1],reverse=True)
    return predicted_class, sorted_by_second[:3]

st.title("Beer Price Check")
st.subheader("By Alex, Azwin, Jason")

st.text(f"{sum(count_pickle)} Beers Identified Correctly")

uploaded_file = st.file_uploader("Upload Image of Beer Logo")

col1, col2 = st.beta_columns(2)
sample = False
if uploaded_file is None:
    if st.button('Load Demo'):
        image_path = "./sample/blueicetest1.jpg"
        st.write('Sample Loaded')
        sample = Image.open(image_path).resize([336,448])
        col1.image(sample)
        uploaded_file = True
        sample = True

## Model Loading
model = tf.keras.models.load_model('SINGLE_MAR30MORN_9888.h5')
class_names = ['Asahi', 'Blue Girl', 'Blue Ice', 'Budweiser', 'Carlsberg', 'Corona Extra', 'Guinness', 'Heineken', 'Kingway', 'Kirin', 'San Mig', 'San Miguel', 'Skol Beer', 'Sol', 'Stella Artois', 'Tiger', 'Tsingtao Beer', 'Yanjing Beer']

if uploaded_file is not None:
    if sample == True:
        try:
            original_image = Image.open(image_path)
            predicted_class, top3 = load_model(original_image)
        except:
          pass
    else:
        # col1.image(Image.open(uploaded_file))

        col1.write("")
        original_image = Image.open(uploaded_file).convert("RGB")
        original_image.save("./sample/test.jpg")

        fixed_image = ImageOps.exif_transpose(original_image)
        ## Test Cropping
        width, height = fixed_image.size
        cropped = ImageOps.crop(fixed_image, border=width*0.2).resize([336,448])
        col1.image(cropped)
        cropped.save("./sample/test_cropped.jpg")
        ## Test Cropping

        predicted_class, top3 = load_model(original_image)

        predicted_class_cropped, top3_cropped = load_model(cropped)


        st.write("Cropped Photo Predictions")
        for i in top3_cropped:
            st.write(i)

    df = load_csv()
    st.header("Best Prices Found")

    temp_df = temp_df()

    st.table(temp_df.style.highlight_min(subset=['Wellcome','PARKnSHOP','Market_Place','Watsons','Aeon','DCH Food Mart'],color = '#D3D3D3', axis = 1))
    correct = "None"
    timestr = time.strftime("%Y%m%d-%H%M%S")

    if sample != True:
        col2.header("Is this {pronoun} {beer_class}?".format(pronoun = "a" if predicted_class[0].lower() not in ['a','e','i','o','u'] else "an", beer_class=predicted_class_cropped))
        col2.text(f"Confidence: {top3_cropped[0][1]}")
        if col2.button("Yes"):
            col2.text("Thank you!")
            correct = "True"

        if col2.button("No"):
            col2.text("Please take a photo with focus on the logo")
            correct = "False"


pickle.dump( count_pickle, open( "counter.p", "wb" ) )
#st.text(f"Model Version: SINGLE_MAR30MORN_9888.h5 {sum(count_pickle)/len(count_pickle) * 100}%")
