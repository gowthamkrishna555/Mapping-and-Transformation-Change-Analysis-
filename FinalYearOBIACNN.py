import streamlit as st
from streamlit_option_menu import option_menu
import folium
from streamlit_folium import st_folium
import cv2
import numpy as np
import os
import imageio
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import json

# Function to normalize image data
def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

# Function to calculate the rate of difference in water regions
def calculate_water_difference_rate(water_mask):
    if water_mask is not None:
        total_pixels = water_mask.size
        different_pixels = np.count_nonzero(water_mask == 255)
        difference_rate = (different_pixels / total_pixels) * 100
        return difference_rate
    else:
        return None

def obia(image1, image2, ndwi_threshold=0.02):
    hsv1 = cv2.cvtColor(image1, cv2.COLOR_RGB2HSV)
    hsv2 = cv2.cvtColor(image2, cv2.COLOR_RGB2HSV)

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([150, 255, 255])

    blue_mask1 = cv2.inRange(hsv1, lower_blue, upper_blue)
    blue_mask2 = cv2.inRange(hsv2, lower_blue, upper_blue)

    ndwi1 = ndwi(image1)
    ndwi2 = ndwi(image2)

    water_mask1 = cv2.bitwise_or(blue_mask1, (ndwi1 >= ndwi_threshold).astype(np.uint8) * 255)
    water_mask2 = cv2.bitwise_or(blue_mask2, (ndwi2 >= ndwi_threshold).astype(np.uint8) * 255)

    kernel = np.ones((10, 10), np.uint8)
    water_mask1 = cv2.morphologyEx(water_mask1, cv2.MORPH_CLOSE, kernel)
    water_mask2 = cv2.morphologyEx(water_mask2, cv2.MORPH_CLOSE, kernel)

    water_difference = cv2.absdiff(water_mask1, water_mask2)
    water_highlighted = np.zeros_like(image2)
    water_highlighted[water_difference == 255, 2] = 255

    water_difference_rate = calculate_water_difference_rate(water_difference)
    return water_difference_rate, water_highlighted

def ndwi(image):
    return (image[:, :, 1].astype(float) - image[:, :, 2]) / (image[:, :, 1] + image[:, :, 2])

def load_and_preprocess_data(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tiff')]
    images = []
    labels = []

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        images.append(img)

        label = int("water" in image_file.lower())
        labels.append(label)

    return np.array(images), np.array(labels)

# Define bg_img outside the main function
bg_img = '''
<style>
        [data-testid="stAppViewContainer"] {
        background-image: url('https://img.freepik.com/free-photo/realistic-water-drop-with-ecosystem_23-2151196442.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        }
</style>
'''
def home():
    st.markdown(bg_img, unsafe_allow_html=True)
    st.title("Welcome to the Water Bodies Mapping and Analysis App")
    st.markdown("""
        <style>
        .intro-text {
            color: #FFFF; /* Change this color to your preferred color */
            backdrop-filter: blur(5px); /* Adjust the blur amount as needed */
            background-color: rgba(0, 0, 0, 0.5); /* Adjust the opacity as needed */
            padding: 20px; /* Add padding for readability */
            border-radius: 10px; /* Optional: Add rounded corners */
        }
        </style>
        <div class="intro-text">
        Water scarcity in Karnataka’s coastal regions is a serious problem that is addressed by
        the project “Mapping and Transformation Change Analysis of Water Bodies Using Deep
        Learning Techniques for The Coastal Regions of Karnataka”. Dynamic environmental
        changes, such as fluctuating water levels and rising freshwater resource demands, can
        affect coastal locations. Sustainably managing these regions' water resources and being
        prepared for emergencies depend on accurate mapping and monitoring of the water bodies. The project uses the OBIA (Object-Based Image Analysis) using NDWI (Normalized
        Difference Water Index) to train a Convolutional Neural Network (CNN) for water change
        detection which automates the identification and delineation of water bodies from satellite 
        photos, giving decision-makers access to real-time data. The project intends to promote
        the resilience of coastal communities confronting difficulties related to water scarcity by
        utilizing deep learning, spectral indices, and geospatial data processing.
        </div>
    """, unsafe_allow_html=True)

def main():
    st.markdown(bg_img, unsafe_allow_html=True)
    with st.sidebar:
        selected = option_menu("Menu", ["Home", "Map", "Analysis"], 
            icons=["house", "map", "activity"], menu_icon="cast", default_index=0)

    if selected == "Home":
        home()
    elif selected == "Map":
        st.title("Water Change Map")

        locations = {
            "Mangalore": [12.9141, 74.8560],
            "Bantwala": [12.8915, 75.0345],
            "Kota": [13.521213, 74.713180 ],
            "Kinnigoli": [13.0878, 74.8803],
            "Malpe": [13.3533, 74.7030],
            "Ullal": [12.829986, 74.858139],
            "Malpe":[13.388200, 74.904850],
            "Perdoor":[13.388200, 74.904850],
            "Upoor":[13.398730, 74.748096 ],
            "Mulki":[13.093166, 74.783377],
            "Innaje":[13.247082, 74.772607 ],
            "Near Kota":[13.528514, 74.707067],
            "Sahyadri":[12.864332, 74.924168 ],
            "Brahmavara":[13.440907, 74.743310],
            "Padukudru":[13.406326, 74.713051],
            "Near Malpe":[13.356447, 74.706345],
            "Udupi":[13.349176, 74.743313],
            "Kaup":[13.235739, 74.750161],
            "Manipal":[13.342366, 74.785760],
            "Near Mulki":[13.079925, 74.792848],
            "Moodabidri":[13.072744, 74.992604],
            "Thokur":[13.054048, 74.808364 ],
            "Padubidri":[13.146603, 74.765632],
            "Tannirbhavi":[12.896955, 74.814833],
            "Karkala":[13.221604, 74.994153],
            "Barkur":[ 13.468499, 74.749343],
            "Msezl":[12.969446, 74.815683 ],
            "Uppinangady":[12.840336, 75.253841],
            "ManCentral":[12.863053, 74.842430],
            "Doopakatte":[13.334061, 74.905343],
            "Gurupura":[ 12.931565, 74.930940],
            "Jokatte":[12.963654, 74.834907],
            "Nandikur":[13.114969, 74.798516],
            "Padil":[12.872770, 74.888319],
            "Perdoor":[13.384576, 74.903630],
            "Pilikula":[ 12.929513, 74.895512],
            "Sashihitlu":[13.071045, 74.779596]
        }

        location_name = st.selectbox("Select Location:", list(locations.keys()))

        selected_coords = locations[location_name]
        st.header(f"Map Centered on {location_name}")

        # Load the GeoJSON data
        with open('Coastal_Region.geojson', 'r') as file:
            coastal_region = json.load(file)

        m = folium.Map(location=selected_coords, zoom_start=12)
        folium.GeoJson(coastal_region).add_to(m)

        for location, coords in locations.items():
            folium.Marker(location=coords, popup=location).add_to(m)

        # Display the map using streamlit_folium
        st_folium(m, width=700, height=500)

    elif selected == "Analysis":
        st.title("Water Change Detection Analysis")

        folder_path1 = "F:\\Folder1-20240410T051840Z-001\\Folder1"
        folder_path2 = "F:\\Folder2-20240410T051833Z-001\\Folder2"

        save_path = "F:\\WaterHighlightedImages"
        os.makedirs(save_path, exist_ok=True)

        image_files1 = [f for f in os.listdir(folder_path1) if f.lower().endswith('.tiff')]
        image_files2 = [f"{f.replace('.tiff', '.tiff')}" for f in image_files1]

        image_file1 = st.selectbox("Select image file for folder 1:", image_files1, index=0 if len(image_files1) > 0 else None)
        
        if image_file1:
            image_file2 = st.selectbox("Select image file for folder 2:", image_files2, index=0 if len(image_files2) > 0 else None)
            
            if image_file1[:-5] != image_file2[:-5]:
                st.error("Selected images do not match. Please select matching images.")
                st.stop()

            if st.button("Run Analysis"):
                image1_path = os.path.join(folder_path1, image_file1)
                image2_path = os.path.join(folder_path2, image_file2)

                img1 = imageio.imread(image1_path)
                img2 = imageio.imread(image2_path)

                # Display the selected images for testing
                st.image(img1, caption='Selected Image 1', use_column_width=True)
                st.image(img2, caption='Selected Image 2', use_column_width=True)

                # Preprocess the images for CNN training
                images, labels = load_and_preprocess_data(save_path)

                if len(images) > 0:
                    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

                    if len(X_train) > 0 and len(y_train) > 0:
                        # Train the CNN model
                        model = models.Sequential([
                            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                            layers.MaxPooling2D((2, 2)),
                            layers.Conv2D(64, (3, 3), activation='relu'),
                            layers.MaxPooling2D((2, 2)),
                            layers.Flatten(),
                            layers.Dense(64, activation='relu'),
                            layers.Dense(1, activation='sigmoid')
                        ])

                        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

                        model.fit(X_train, y_train, epochs=5)

                        test_loss, test_acc = model.evaluate(X_test, y_test)
                        

                        model.save("F:\\water_change_detection_model.h5")

                        # Calculate water difference after training
                        water_difference_rate, water_highlighted = obia(img1, img2)

                        if water_difference_rate is not None:
                            st.write(f"Rate of Water Difference: {water_difference_rate:.2f}%")
                            st.image(normalize_image(water_highlighted), caption='Water Difference Image', use_column_width=True)
                            if water_difference_rate == 0:
                                st.write("There might be a low probability of changes in the water body over the 4 years span.")
                        else:
                            st.write("Could not calculate water difference rate.")

                    else:
                        st.write("Data is empty. Please check your dataset.")
                else:
                    st.write("No images found for analysis.")

if __name__ == "__main__":
    main()