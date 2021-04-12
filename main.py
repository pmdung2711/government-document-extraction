print("import model again!")
import sys 
import streamlit as st
import numpy as np
import pandas as pd
import os 
import time
from inference import DocInfo, DocumentAnalysis
from PIL import  Image 
import cv2
from pdf2image import convert_from_bytes 
from os import listdir
import numpy


@st.cache(allow_output_mutation=True)
def load_model():
    document_cls = DocumentAnalysis("weights/")
    return document_cls


def prediction(opencv_image):
    doc_info = document_cls.extract_information(opencv_image)
    return doc_info


def main():

    #Title 
    st.title("Document Information Extraction")


    #Upload File
    file_uploader = st.file_uploader("Please select a file", type=["png","jpg","pdf"], accept_multiple_files=False)


    if st.button("Start"):

        if file_uploader:
            #Show the file name and type
            file_name = file_uploader.name
            file_type = file_uploader.type.split("/")[1]
            df = st.table({
                'File name:':[file_name],
                "Type":[file_type]
            })

            #read the image
            file_bytes = np.asarray(bytearray(file_uploader.read()), dtype=np.uint8)
            opencv_image = None 
            if file_type in ["jpg", "jpeg", "png"]:
                opencv_image = cv2.imdecode(file_bytes, 1)
            else:
                images = convert_from_bytes(file_bytes)
                open_cv_image = numpy.array(images[0]) 
                # Convert RGB to BGR 
                opencv_image = open_cv_image[:, :, ::-1].copy() 
                

            
            #Get the prediction
            loading_state = st.empty()
            bar = st.progress(0)
            loading_state.text("Predicting...")
            doc_info = prediction(opencv_image)
            loading_state.text("Finished predicting!")
            bar.progress(100)


            #Show the prediction
            st.subheader("Result")
            df = pd.DataFrame({
                "Type":["Document Number", "Publisher", "Date","Summary"],
                "Content":[doc_info.doc_code, doc_info.publisher, doc_info.date, doc_info.summary],
            }) 
            st.table(df)

            #show all the texts in the document
            st.subheader("All Paragraphs")
            df_texts = pd.DataFrame(doc_info.texts,columns=['Content'])
            st.table(df_texts)
            
            st.subheader("Original Image")
            st.image(opencv_image, channels="BGR")

        else: 
            st.write("Please select a file first!")




if __name__ == "__main__":
    document_cls = load_model()
    main()