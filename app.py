



# # #  import spacy
# # # from collections import defaultdict

# # # nlp = spacy.load("en_core_web_md")

# # # # Expanded travel-related keywords
# # # travel_keywords = {


# # #     "budget": ["budget", "cost", "money", "expenses", "price", "spending", "affordable", "economical", "value", "savings", "expenditure"],
# # #     "duration": ["day", "days", "nights", "duration", "timeframe", "length", "schedule", "period", "time", "itinerary", "stay", "overnight"],
# # #     "cities": ["city", "cities", "destination", "place", "visit", "town", "area", "region", "metropolis", "locale", "urban", "attraction", "site"],
# # #     "activities": ["hiking", "cycling", "explore", "visit", "experience", "sightseeing", "adventure", "tour", "excursion", "activities", "recreation", "leisure", "outdoor", "cultural", "entertainment", "sports", "crafts", "hobbies"],
# # #     "interests": ["temples", "photography", "ceremonies", "festivals", "markets", "culture", "history", "art", "architecture", "nature", "wildlife", "local traditions", "music", "dance", "shopping", "cuisine", "sports", "hobbies", "crafts"],
# # #     "foods": ["food", "sushi", "tempura", "vegetarian", "dishes", "cuisine", "meals", "dining", "restaurants", "street food", "local specialties", "gourmet", "snacks", "beverages", "drinks", "desserts", "vegan", "gluten-free", "organic"],
# # #     "transport": ["public transport", "rental bikes", "buses", "trains", "subway", "metro", "taxi", "ride-sharing", "car rental", "transportation", "ferry", "shuttle", "tram", "flights", "driving", "bike rental", "boat", "walk"],
# # #     "accommodation": ["hotel", "hostel", "guesthouse", "Airbnb", "lodging", "resort", "inn", "motel", "B&B", "accommodation", "stay", "room", "suite", "cabin", "villa", "camping", "holiday park", "apartment"],
# # #     "weather": ["weather", "climate", "temperature", "season", "forecast", "sunny", "rainy", "snowy", "windy", "humidity", "cold", "hot", "warm", "cool", "conditions"],
# # #     "travel_documents": ["passport", "visa", "ID", "travel insurance", "tickets", "booking", "reservation", "itinerary", "confirmation", "documentation", "permits"],
# # #     "safety": ["safety", "security", "health", "precautions", "insurance", "emergency", "risk", "travel advisories", "medical care", "vaccinations", "first aid"],
# # #     "communication": ["language", "translation", "local phrases", "communication", "internet", "Wi-Fi", "SIM card", "contact", "language barrier", "local customs"],
# # #     "packing": ["packing", "luggage", "suitcase", "carry-on", "essentials", "gear", "clothes", "toiletries", "travel accessories", "checklist"],
# # #     "local_experiences": ["local experiences", "cultural immersion", "community events", "workshops", "guided tours", "local guides", "homestays", "unique experiences"],
# # #     "reviews_and_recommendations": ["reviews", "ratings", "feedback", "recommendations", "testimonials", "tips", "advice", "guidebooks", "blogs", "travel forums"],
# # #     "must_see_do_destinations": ["must-see", "must-do", "top attractions", "highlights", "landmarks", "iconic sites", "bucket list", "famous spots", "recommended places"],
# # #     "interests_special": ["sports", "hobbies", "crafts", "fishing", "surfing", "skiing", "rock climbing", "painting", "pottery", "sculpting", "golf", "cycling", "photography", "writing"],
# # #     "food_and_diet_preferences": ["vegan", "vegetarian", "gluten-free", "dairy-free", "halal", "kosher", "organic", "local cuisine", "special diets", "allergies", "dietary restrictions"],
# # #     "fitness": ["fitness", "gym", "workout", "exercise", "yoga", "running", "hiking", "cycling", "swimming", "wellness", "spa", "physical activity", "training"],
# # #     "families_with_kids": ["family-friendly", "kids", "children", "family activities", "kid-friendly", "age of kids", "toddler", "preschool", "school-aged", "teenagers", "family vacation"],
# # #     "season": ["season", "winter", "summer", "spring", "autumn", "holiday season", "peak season", "off-season", "weather conditions"],
# # #     "transportation_preferences": ["transportation preferences", "private car", "public transit", "bike rental", "walking", "ride-sharing", "chauffeur", "carpooling", "bus", "train", "plane", "boat"]



# # # }

# # # # Define a function to extract key information based on travel keywords
# # # def extract_travel_information(doc):
# # #     extracted_info = defaultdict(list)
    
# # #     # Extract entities using spaCy's NER
# # #     for ent in doc.ents:
# # #         if ent.label_ == "MONEY":
# # #             extracted_info["budget"].append(ent.text)
# # #         elif ent.label_ == "DATE" or ent.label_ == "TIME":
# # #             extracted_info["duration"].append(ent.text)
# # #         elif ent.label_ == "GPE":
# # #             extracted_info["cities"].append(ent.text)
    
# # #     # Extract information based on keyword matching
# # #     for token in doc:
# # #         for category, keywords in travel_keywords.items():
# # #             if any(keyword in token.text.lower() for keyword in keywords):
# # #                 extracted_info[category].append(token.text)
    
# # #     return extracted_info

# # # # Example travel corpus
# # # corpus ="""
# # # Hello! I’m planning a family trip and would love your assistance. Our budget is around $5,000 for a 10-day tour. We are particularly interested in visiting must-see destinations like Tokyo, Kyoto, and Mount Fuji. Our interests include exploring historical sites, enjoying local crafts, and attending a traditional tea ceremony. We prefer a mix of Japanese and vegetarian cuisine. Fitness is important to us, so we’d like to include some hiking and walking tours. We have two kids, aged 8 and 12, who are excited about visiting theme parks and interactive museums. We’re planning this trip for the summer season. For transportation, we prefer using public transport like trains and buses to get a more authentic experience. Thank you for helping us plan this exciting adventure!
# # # """


# # # # Process the corpus using spaCy
# # # doc = nlp(corpus)

# # # # Extract the travel information
# # # travel_info = extract_travel_information(doc)


# # # for category, info in travel_info.items():
# # #     print(f"{category.capitalize()}: {', '.join(info)}") can you make it deploy and printt the dataframe in form me a table in the interface
















# # import streamlit as st
# # import spacy
# # from collections import defaultdict
# # import pandas as pd

# # # Load the spaCy model
# # nlp = spacy.load("en_core_web_md")

# # # Expanded travel-related keywords
# # travel_keywords = {
# #     "budget": ["budget", "cost", "money", "expenses", "price", "spending", "affordable", "economical", "value", "savings", "expenditure"],
# #     "duration": ["day", "days", "nights", "duration", "timeframe", "length", "schedule", "period", "time", "itinerary", "stay", "overnight"],
# #     "cities": ["city", "cities", "destination", "place", "visit", "town", "area", "region", "metropolis", "locale", "urban", "attraction", "site"],
# #     "activities": ["hiking", "cycling", "explore", "visit", "experience", "sightseeing", "adventure", "tour", "excursion", "activities", "recreation", "leisure", "outdoor", "cultural", "entertainment", "sports", "crafts", "hobbies"],
# #     "interests": ["temples", "photography", "ceremonies", "festivals", "markets", "culture", "history", "art", "architecture", "nature", "wildlife", "local traditions", "music", "dance", "shopping", "cuisine", "sports", "hobbies", "crafts"],
# #     "foods": ["food", "sushi", "tempura", "vegetarian", "dishes", "cuisine", "meals", "dining", "restaurants", "street food", "local specialties", "gourmet", "snacks", "beverages", "drinks", "desserts", "vegan", "gluten-free", "organic"],
# #     "transport": ["public transport", "rental bikes", "buses", "trains", "subway", "metro", "taxi", "ride-sharing", "car rental", "transportation", "ferry", "shuttle", "tram", "flights", "driving", "bike rental", "boat", "walk"],
# #     "accommodation": ["hotel", "hostel", "guesthouse", "Airbnb", "lodging", "resort", "inn", "motel", "B&B", "accommodation", "stay", "room", "suite", "cabin", "villa", "camping", "holiday park", "apartment"],
# #     "weather": ["weather", "climate", "temperature", "season", "forecast", "sunny", "rainy", "snowy", "windy", "humidity", "cold", "hot", "warm", "cool", "conditions"],
# #     "travel_documents": ["passport", "visa", "ID", "travel insurance", "tickets", "booking", "reservation", "itinerary", "confirmation", "documentation", "permits"],
# #     "safety": ["safety", "security", "health", "precautions", "insurance", "emergency", "risk", "travel advisories", "medical care", "vaccinations", "first aid"],
# #     "communication": ["language", "translation", "local phrases", "communication", "internet", "Wi-Fi", "SIM card", "contact", "language barrier", "local customs"],
# #     "packing": ["packing", "luggage", "suitcase", "carry-on", "essentials", "gear", "clothes", "toiletries", "travel accessories", "checklist"],
# #     "local_experiences": ["local experiences", "cultural immersion", "community events", "workshops", "guided tours", "local guides", "homestays", "unique experiences"],
# #     "reviews_and_recommendations": ["reviews", "ratings", "feedback", "recommendations", "testimonials", "tips", "advice", "guidebooks", "blogs", "travel forums"],
# #     "must_see_do_destinations": ["must-see", "must-do", "top attractions", "highlights", "landmarks", "iconic sites", "bucket list", "famous spots", "recommended places"],
# #     "interests_special": ["sports", "hobbies", "crafts", "fishing", "surfing", "skiing", "rock climbing", "painting", "pottery", "sculpting", "golf", "cycling", "photography", "writing"],
# #     "food_and_diet_preferences": ["vegan", "vegetarian", "gluten-free", "dairy-free", "halal", "kosher", "organic", "local cuisine", "special diets", "allergies", "dietary restrictions"],
# #     "fitness": ["fitness", "gym", "workout", "exercise", "yoga", "running", "hiking", "cycling", "swimming", "wellness", "spa", "physical activity", "training"],
# #     "families_with_kids": ["family-friendly", "kids", "children", "family activities", "kid-friendly", "age of kids", "toddler", "preschool", "school-aged", "teenagers", "family vacation"],
# #     "season": ["season", "winter", "summer", "spring", "autumn", "holiday season", "peak season", "off-season", "weather conditions"],
# #     "transportation_preferences": ["transportation preferences", "private car", "public transit", "bike rental", "walking", "ride-sharing", "chauffeur", "carpooling", "bus", "train", "plane", "boat"]
# # }

# # # Sample Travel Dataset for Recommendations
# # travel_data = {
# #     'Destination': ['Tokyo', 'Kyoto', 'Mount Fuji', 'Osaka', 'Hokkaido'],
# #     'Activities': ['Historical Sites, Temples', 'Cultural Tours, Food Tours', 'Hiking, Sightseeing', 'Theme Parks, Shopping', 'Skiing, Nature'],
# #     'Budget': ['medium', 'low', 'medium', 'medium', 'high'],
# #     'Suitable For': ['Families, Cultural Travelers', 'Backpackers, Cultural Enthusiasts', 'Hikers, Nature Lovers', 'Families, Shoppers', 'Skiers, Nature Enthusiasts']
# # }

# # df_travel = pd.DataFrame(travel_data)

# # # Function to extract key information based on travel keywords
# # def extract_travel_information(doc):
# #     extracted_info = defaultdict(list)
    
# #     # Extract entities using spaCy's NER
# #     for ent in doc.ents:
# #         if ent.label_ == "MONEY":
# #             extracted_info["budget"].append(ent.text)
# #         elif ent.label_ == "DATE" or ent.label_ == "TIME":
# #             extracted_info["duration"].append(ent.text)
# #         elif ent.label_ == "GPE":
# #             extracted_info["cities"].append(ent.text)
    
# #     # Extract information based on keyword matching
# #     for token in doc:
# #         for category, keywords in travel_keywords.items():
# #             if any(keyword in token.text.lower() for keyword in keywords):
# #                 extracted_info[category].append(token.text)
    
# #     return extracted_info

# # # Streamlit app
# # st.title('Travel Planner NLP Extractor & Recommendation System')

# # # User input text
# # corpus = st.text_area("Enter your travel description:", """
# # Hello! I’m planning a family trip and would love your assistance. Our budget is around $5,000 for a 10-day tour. We are particularly interested in visiting must-see destinations like Tokyo, Kyoto, and Mount Fuji. Our interests include exploring historical sites, enjoying local crafts, and attending a traditional tea ceremony. We prefer a mix of Japanese and vegetarian cuisine. Fitness is important to us, so we’d like to include some hiking and walking tours. We have two kids, aged 8 and 12, who are excited about visiting theme parks and interactive museums. We’re planning this trip for the summer season. For transportation, we prefer using public transport like trains and buses to get a more authentic experience. Thank you for helping us plan this exciting adventure!
# # """)

# # # Process the input text using spaCy
# # doc = nlp(corpus)

# # # Extract the travel information
# # travel_info = extract_travel_information(doc)

# # # Display extracted information as a DataFrame
# # df_extracted = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in travel_info.items()]))
# # st.write("## Extracted Travel Information")
# # st.dataframe(df_extracted)

# # # Recommendation System Logic
# # st.write("## Recommended Destinations and Activities")

# # user_budget = "medium"  # Assuming user budget is medium
# # recommendations = df_travel[df_travel['Budget'].str.contains(user_budget, case=False)]

# # st.write("Based on your preferences, we recommend the following destinations:")
# # st.dataframe(recommendations)






















# import streamlit as st
# import pandas as pd
# import spacy
# import random
# import re
# from tabulate import tabulate

# # Load the spaCy model
# nlp = spacy.load("en_core_web_sm")

# # Synthetic dataset creation (using the code you provided)
# names = ["Hotel Sakura", "Kyoto Serenity", "Tokyo Tower Inn", "Mount Fuji Retreat", "Osaka Bay Resort", "Nara Zen Lodge", "Shibuya Plaza"]
# locations = ["Tokyo", "Kyoto", "Osaka", "Sapporo", "Nara", "Nagoya", "Fukuoka", "Hiroshima", "Yokohama", "Kobe"]
# number_of_rooms = list(range(5, 301))
# price_per_night = list(range(5000, 60001, 1000))
# good_review_count = list(range(10, 5001))
# bad_review_count = list(range(0, 1001))
# amenities_list = ["Wi-Fi", "Onsen", "Free Breakfast", "Spa", "Parking", "Pet-friendly", "Room Service", "Bar", "Restaurant", "Fitness Center", "Business Center"]
# meal_plan = ["Bed & Breakfast", "Half Board", "Full Board", "All-Inclusive", "Room Only"]
# meal_preferences = ["Veg", "Non Veg", "Both", "Vegan", "Jain", "Non veg without Beef"]
# car_parking_space = [True, False]
# standards = ["Luxury", "Premium", "Standard", "Budget"]

# # Helper functions to generate the dataset
# def generate_id(index):
#     return f"JPN{str(index).zfill(4)}"

# def generate_random_amenities():
#     return ", ".join(random.sample(amenities_list, random.randint(3, 6)))

# def calculate_rating(good_reviews, bad_reviews):
#     total_reviews = good_reviews + bad_reviews
#     if total_reviews == 0:
#         return 1
#     good_ratio = good_reviews / total_reviews
#     if good_ratio <= 1/5:
#         return 1
#     elif good_ratio <= 2/5:
#         return 2
#     elif good_ratio <= 3/5:
#         return 3
#     elif good_ratio <= 4/5:
#         return 4
#     else:
#         return 5

# def assign_meal_preference(meal_plan):
#     if meal_plan == "Room Only":
#         return None
#     else:
#         return random.choice(meal_preferences)

# def assign_standard_and_price():
#     standard = random.choices(standards, weights=[10, 20, 30, 40])[0]
    
#     if standard == "Luxury":
#         price = random.choice(range(30000, 60001, 1000))
#     elif standard == "Premium":
#         price = random.choice(range(20000, 30001, 1000))
#     elif standard == "Standard":
#         price = random.choice(range(12000, 20001, 1000))
#     else:  # Budget
#         price = random.choice(range(5000, 12001, 1000))
    
#     return standard, price

# def generate_random_entry(index):
#     good_reviews = random.choice(good_review_count)
#     bad_reviews = random.choice(bad_review_count)
#     rating = calculate_rating(good_reviews, bad_reviews)
    
#     selected_standard, price_per_night = assign_standard_and_price()
#     selected_meal_plan = random.choice(meal_plan)
#     meal_preference = assign_meal_preference(selected_meal_plan)  
    
#     return {
#         "id": generate_id(index),
#         "name": random.choice(names),
#         "location": random.choice(locations),
#         "available number of rooms": random.choice(number_of_rooms),
#         "standard": selected_standard,
#         "price per night": price_per_night,
#         "amenities": generate_random_amenities(),
#         "type of meal plan": selected_meal_plan,
#         "meal preference": meal_preference,
#         "required car parking space": random.choice(car_parking_space),
#         "good review count": good_reviews,
#         "bad review count": bad_reviews,
#         "rating": rating,
#         "hotel_description": "A comfortable and convenient hotel offering excellent services and amenities.",
#         "distance_from_city_center": f"{random.randint(1, 20)} km",
#         "check_in_time": f"{random.randint(14, 16)}:00",
#         "check_out_time": f"{random.randint(10, 12)}:00"
#     }

# # Generate the dataset
# dataset = [generate_random_entry(i) for i in range(1000)]
# df = pd.DataFrame(dataset)

# # Function to extract hotel attributes from the input text using NLP
# def extract_hotel_attributes(text):
#     doc = nlp(text)
    
#     location = []
#     max_price = None
#     amenities = []
#     meal_plan = None
#     meal_preference = None
#     car_parking_space = None
#     standard = None
#     duration = None
    
#     amenities_keywords = ['Wi-Fi', 'Spa', 'Parking', 'Bar', 'Restaurant', 'Onsen', 'Fitness', 'Business', 'Pool', 'Market']
#     meal_plan_keywords = ['Bed & Breakfast', 'Half Board', 'Full Board', 'All-Inclusive', 'Room Only']
#     meal_preference_keywords = ['Veg', 'Non Veg', 'Vegan', 'Jain']
#     standard_keywords = ['Luxury', 'Premium', 'Standard', 'Budget']
    
#     for ent in doc.ents:
#         if ent.label_ == 'GPE':
#             location.append(ent.text)
#         if ent.label_ == 'MONEY':
#             price = re.sub(r'[^\d]', '', ent.text)
#             if price:
#                 max_price = int(price)
#         if ent.label_ == 'DATE':
#             duration = ent.text
    
#     for token in doc:
#         for keyword in amenities_keywords:
#             if keyword.lower() in token.text.lower():
#                 amenities.append(keyword)
#         for keyword in meal_preference_keywords:
#             if keyword.lower() in token.text.lower():
#                 meal_preference = keyword
#         for keyword in standard_keywords:
#             if keyword.lower() in token.text.lower():
#                 standard = keyword
#         if 'parking' in token.text.lower():
#             car_parking_space = True
    
#     return {
#         'location': location if location else None,
#         'max_price': max_price,
#         'amenities': ', '.join(amenities) if amenities else None,
#         'meal_plan': meal_plan,
#         'meal_preference': meal_preference,
#         'car_parking_space': car_parking_space,
#         'standard': standard,
#         'duration': duration
#     }

# # Function to recommend hotels based on extracted attributes
# def recommend_hotels(df, max_price=None, amenities=None, standard=None, meal_plan=None, meal_preference=None, car_parking_space=None, location=None):
#     query = []
    
#     if max_price is not None:
#         query.append(f"`price per night` <= {max_price}")
    
#     if amenities:
#         amenities_list = [f"`amenities`.str.contains('{re.escape(amenity.strip())}', case=False, na=False)" for amenity in amenities.split(',')]
#         query.append(' and '.join(amenities_list))
    
#     if standard:
#         query.append(f"`standard` == '{standard}'")
    
#     if meal_plan:
#         query.append(f"`type of meal plan` == '{meal_plan}'")
    
#     if meal_preference:
#         query.append(f"`meal preference` == '{meal_preference}'")
    
#     if car_parking_space is not None:
#         query.append(f"`required car parking space` == {car_parking_space}")
    
#     if location:
#         location_filter = ' or '.join([f"`location`.str.contains('{loc.strip()}', case=False, na=False)" for loc in location])
#         query.append(f"({location_filter})")
    
#     if query:
#         query_string = ' and '.join(query)
#         result_df = df.query(query_string)
#         return result_df
#     else:
#         return df

# def get_hotel_recommendations(text):
#     extracted_attributes = extract_hotel_attributes(text)
#     recommendations = recommend_hotels(
#         df,
#         max_price=extracted_attributes['max_price'],
#         amenities=extracted_attributes['amenities'],
#         standard=extracted_attributes['standard'],
#         meal_plan=extracted_attributes['meal_plan'],
#         meal_preference=extracted_attributes['meal_preference'],
#         car_parking_space=extracted_attributes['car_parking_space'],
#         location=extracted_attributes['location']
#     )
#     return recommendations

# # Streamlit UI
# st.title('Hotel Recommendation System')

# # Input section
# user_input = st.text_area("Enter your travel preferences", "I am planning a trip to Tokyo with a budget of 20,000 yen per night. I want vegan food and a hotel with Wi-Fi and a spa.")

# if st.button('Find Hotels'):
#     # Get recommendations based on user input
#     recommended_hotels = get_hotel_recommendations(user_input)
    
#     if not recommended_hotels.empty:
#         st.write(f"### Found {len(recommended_hotels)} hotel(s) matching your preferences:")
#         st.dataframe(recommended_hotels[['name', 'location', 'price per night', 'amenities', 'rating', 'meal preference', 'required car parking space']].head(10))
#     else:
#         st.write("No hotels match your search criteria.")













# ==============================================================================================================================================================================
# ==============================================================================================================================================================================









import spacy
import pandas as pd
import re
import streamlit as st

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
