

# ==============================================================================================================================================================================
# ==============================================================================================================================================================================


# import spacy
# import pandas as pd
# import re
# import streamlit as st

# # Ensure the model is downloaded
# try:
#     nlp = spacy.load("en_core_web_sm")
# except OSError:
#     import subprocess
#     subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
#     nlp = spacy.load("en_core_web_sm")





import subprocess
import sys
import importlib

def install_packages():
    with open('requirements.txt') as f:
        packages = f.readlines()

    for package in packages:
        package = package.strip()
        try:
            __import__(package.split('==')[0])
        except ImportError:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

def import_library(library_name):
    try:
        return importlib.import_module(library_name)
    except ImportError as e:
        print(f"Error importing {library_name}: {e}")
        return None

# Install missing packages
install_packages()

# Import libraries dynamically
spacy = import_library('spacy')
pandas = import_library('pandas')
streamlit = import_library('streamlit')

# Check if SpaCy is successfully imported
if spacy:
    nlp = spacy.load("en_core_web_sm")
else:
    nlp = None




# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load the dataset from the CSV file
csv_file_path = r"https://raw.githubusercontent.com/itsrudhir/llm_testing/main/synthetic_hotels_dataset_japan.csv"  # Use the correct path
df = pd.read_csv(csv_file_path)

# NLP extraction function
def extract_hotel_attributes(text):
    doc = nlp(text)

    location = []
    max_price = None
    amenities = []
    meal_plan = None
    meal_preference = None
    car_parking_space = None
    standard = None
    duration = None

    amenities_keywords = ['Wi-Fi', 'Spa', 'Parking', 'Bar', 'Restaurant', 'Onsen', 'Fitness', 'Business', 'Alley spots', 'Pool', 'Market']
    meal_plan_keywords = ['Bed & Breakfast', 'Half Board', 'Full Board', 'All-Inclusive', 'Room Only']
    meal_preference_keywords = ['Veg', 'Non Veg', 'Vegan', 'Jain', 'Sushi', 'Tempura', 'Vegetarian']
    standard_keywords = ['Luxury', 'Premium', 'Standard', 'Budget']

    for ent in doc.ents:
        if ent.label_ == 'GPE':
            location.append(ent.text)
        if ent.label_ == 'MONEY':
            price = re.sub(r'[^\d]', '', ent.text)
            if price:
                max_price = int(price)
        if ent.label_ == 'DATE':
            duration = ent.text

    for token in doc:
        for keyword in amenities_keywords:
            if keyword.lower() in token.text.lower():
                amenities.append(keyword)

        for keyword in meal_preference_keywords:
            if keyword.lower() in token.text.lower():
                meal_preference = keyword

        for keyword in standard_keywords:
            if keyword.lower() in token.text.lower():
                standard = keyword

        if 'parking' in token.text.lower():
            car_parking_space = True

    if 'all sorts of food' in text.lower():
        meal_preference = 'Both'

    return {
        'location': location if location else None,
        'max_price': max_price,
        'amenities': ', '.join(amenities) if amenities else None,
        'meal_plan': meal_plan,
        'meal_preference': meal_preference,
        'car_parking_space': car_parking_space,
        'standard': standard,
        'duration': duration
    }

# Hotel recommendation function
def recommend_hotels(df, max_price=None, amenities=None, standard=None, meal_plan=None, meal_preference=None, car_parking_space=None, location=None):
    query = []

    if max_price is not None:
        query.append(f"`price per night` <= {max_price}")

    if amenities:
        amenities_list = [f"`amenities`.str.contains('{re.escape(amenity.strip())}', case=False, na=False)" for amenity in amenities.split(',')]
        query.append(' and '.join(amenities_list))

    if standard:
        query.append(f"`standard` == '{standard}'")

    if meal_plan:
        query.append(f"`type of meal plan` == '{meal_plan}'")

    if meal_preference:
        query.append(f"`meal preference` == '{meal_preference}'")

    if car_parking_space is not None:
        query.append(f"`required car parking space` == {car_parking_space}")

    if location:
        location_filter = ' or '.join([f"`location`.str.contains('{loc.strip()}', case=False, na=False)" for loc in location])
        query.append(f"({location_filter})")

    if query:
        query_string = ' and '.join(query)
        result_df = df.query(query_string)
        return result_df
    else:
        return df

# Function to process user input and recommend hotels
def get_hotel_recommendations(text):
    extracted_attributes = extract_hotel_attributes(text)

    recommendations = recommend_hotels(
        df,
        max_price=extracted_attributes['max_price'],
        amenities=extracted_attributes['amenities'],
        standard=extracted_attributes['standard'],
        meal_plan=extracted_attributes['meal_plan'],
        meal_preference=extracted_attributes['meal_preference'],
        car_parking_space=extracted_attributes['car_parking_space'],
        location=extracted_attributes['location']
    )

    return recommendations

# Streamlit UI
st.title("Japan Hotel Recommendation System")
st.write("Enter your travel preferences, and we'll recommend the best hotels for you!")

# Input from the user
user_input = st.text_area("Enter your preferences (e.g., 'I am planning a trip to Tokyo with a budget of 20,000 yen per night. I want vegan food and a hotel with Wi-Fi and a spa.'):")

# Button to generate recommendations
if st.button("Get Recommendations"):
    if user_input:
        # Get hotel recommendations based on user input
        recommended_hotels = get_hotel_recommendations(user_input)

        if not recommended_hotels.empty:
            st.write("Here are some hotels that match your preferences:")
            # Show all features for the recommended hotels
            st.dataframe(recommended_hotels)
        else:
            st.write("No hotels match your search criteria.")
