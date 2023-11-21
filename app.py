import streamlit as st
import cv2, os, time
import numpy as np
from PIL import Image
from descriptors import bitdesc,haralick,glcm
import pandas as pd
from scipy.spatial import distance
from distances import distance_selection
from upload import upload_file
def main():
    print("App lunched!")
    # Enter number of results
    input_value = st.sidebar.number_input("Enter a value", min_value=1, max_value=500, value=10, step=1)
    # Display input value
    st.sidebar.write(f"You entered {input_value}")
    # Define distances
    options = ["Euclidean", "Canberra", "Manhattan", "Chebyshev", "Minkowsky"]
    distance_option = st.sidebar.selectbox("Select a distance", options)
    st.sidebar.write(f"You chose {distance_option}")
    # Select the descriptor type
    descriptor_options = ["Bitdesc", "Haralick", "GLCM"]
    descriptor_choice = st.sidebar.selectbox("Select a descriptor", descriptor_options)
    st.sidebar.write(f"You chose {descriptor_choice}")
    # Import off-line database (signatures)
    signatures = np.load('cbir_signatures_v1.npy')
    #signatures = np.load('cbir_bitdesc_signatures_v1.npy')
    if(descriptor_choice == "Bitdesc"):
        signatures = np.load('cbir_bitdesc_signatures_v1.npy')
    elif(descriptor_choice == 'Haralick'):
        signatures = np.load('cbir_haralick_signatures_v1.npy')
    elif(descriptor_choice == 'GLCM'):
        signatures = np.load('cbir_glcm_signatures_v1.npy')
    # st.write(signatures)
    # Define a list for computed distances
    distanceList = list()
    image_paths = ''
    # Upload image
    is_image_uploaded = upload_file()
    if is_image_uploaded:
        st.write('''
                 # Search Results
                 ''')
        # Retrieve query imahe
        query_image = 'uploaded_images/query_image.png'
        # Read image as gray-scale
        img = cv2.imread(query_image, 0)

        # Compute features using the selected descriptor
        if descriptor_choice == "Bitdesc":
            bit_feat = bitdesc(img)
        elif descriptor_choice == "Haralick":
            bit_feat = haralick(img)
        elif descriptor_choice == "GLCM":
            bit_feat = glcm(img)
        # Get signatures (extract features) of query image/Compute Bitdesc
        bit_feat = bitdesc(img)
        # Compute Similarity distance
        start_time = time.time()
        for sign in signatures:
            # Remove the last two columns ('subfolder', 'path')
            sign = np.array(sign)[0:-2].astype('float')
            # Convert numpy to list
            sign = sign.tolist()
            # Call distance function
            distance = distance_selection(distance_option, bit_feat, sign)
            distanceList.append(distance)
        print("Ditsnce computed successfully")
        # Compute n min distances
        minDistances =list()
        for i in range(input_value):
            array = np.array(distanceList)
            # Get index of min value from distance list and add to minDistances list
            index_min = np.argmin(array)
            minDistances.append(index_min)
            # Grab max value
            max = array.max()
            # Overwrite the min value with max value
            distanceList[index_min] = max
        print(minDistances)
        # Retrieve path of most similar images using their distances
        # image_paths = list()
        # for small in minDistances:
        #     image_paths.append(signatures[small][-2])
        image_paths = [signatures[small][-1] for small in minDistances]
        # Retrieve classes/Types of most similar images using their distances
        classes = [signatures[small][-2] for small in minDistances]
        classes = np.array(classes)
        # Get unique values of types and count all
        unique_values, counts = np.unique(classes, return_counts=True)
        list_classes = list()
        print("Unique value with their counts")
        for value, count in zip(unique_values, counts):
            print(f"{value}:{count}")
            list_classes.append(value)
        # Create pandas Dataframe with the unique value and their counts
        df = pd.DataFrame({"Value": unique_values, "frequency":counts})
        #st.write(df)
        # Plot bar chart and set Value as index. Frequency as value to display
        #affiche les donnees sous forme de graphique
        st.bar_chart(df.set_index("Value"))
        # Display the prediction, number of similar images, and time taken
        st.write("Prediction:", list_classes[0])  
        st.write("Number of Similar Images:", input_value)
         #mesurer le temps ecoule 
        elapsed_time = time.time() - start_time
        st.write("Time Taken (seconds):", elapsed_time)
        st.write('''

 

             # Similar Images

 

             ''')
    # Divisez l'espace en 3 colonnes
    columns = st.columns(3)
    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        #si image existe
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Afficher l'image dans la colonne correspondante
            #afficher une image RGB dans une application Streamlit en organisant les images dans trois colonnes et en attribuant une légende numérotée à chaque image. 
            columns[i % 3].image(img_rgb, caption=f"Similar Image {i+1}", use_column_width=True)
        else:
           st.write(f"Error loading image: {image_path}")
        
        # Display
        #for img in image_paths:
            #st.image(Image.open(img))
            
                        
        
    else:
        st.write("Welcome! Please upload an image to get started ...")
                                   
if __name__ == "__main__":
    main()
    