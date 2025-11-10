# -*- coding: utf-8 -*-
"""Smart Swachh Portal – Fixed Version"""

import streamlit as st
import pandas as pd
from datetime import datetime, date, time
import os
import re
import base64
import json

# Try to import OpenAI, handle if not installed
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Configuration
CSV_FILE = "smart_swachh_reports.csv"

# Initialize session state
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False
if 'classified_category' not in st.session_state:
    st.session_state.classified_category = None
if 'classified_subcategory' not in st.session_state:
    st.session_state.classified_subcategory = None

# Category mappings
CATEGORIES = {
    "Cars": [],
    "Bikes": ["Motorcycles", "Scooters", "Spare Parts", "Bicycles"],
    "Electronics & Appliances": ["TVs, Video - Audio", "Kitchen & Other Appliances",
                                  "Computers & Laptops", "Cameras & Lenses"],
    "Mobiles": ["Mobile Phones", "Accessories", "Tablets"],
    "Commercial Vehicles & Spares": ["Commercial & Other Vehicles", "Spare Parts"],
    "Furniture": ["Sofa & Dining", "Beds & Wardrobes", "Home Decor & Garden",
                  "Kids Furniture", "Other Household Items"],
    "Fashion": ["Men", "Women", "Kids"],
    "Books, Sports & Hobbies": ["Books", "Gym & Fitness", "Musical Instruments",
                                 "Sports Equipment", "Other Hobbies"]
}

# ---------------------------------------------------------
# Validation & Helper Functions
# ---------------------------------------------------------

def validate_phone(phone):
    """Validate Indian phone number format"""
    pattern = r'^[6-9]\d{9}$'
    return bool(re.match(pattern, phone))

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_coordinates(lat, lon):
    """Validate latitude and longitude"""
    try:
        if lat and lon:
            lat_f = float(lat)
            lon_f = float(lon)
            return (-90 <= lat_f <= 90) and (-180 <= lon_f <= 180)
        return True
    except ValueError:
        return False

def encode_image_to_base64(image_file):
    """Convert uploaded image to base64"""
    return base64.b64encode(image_file.read()).decode('utf-8')

def classify_image_with_openai(image_file, api_key):
    """Classify image using OpenAI Vision API"""
    try:
        # Reset file pointer
        image_file.seek(0)

        # Encode image
        base64_image = encode_image_to_base64(image_file)
        image_file.seek(0)

        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        # Prepare category list
        category_list = "\n".join([f"- {cat}: {', '.join(subs) if subs else 'No subcategories'}"
                                   for cat, subs in CATEGORIES.items()])

        # Prompt
        prompt = f"""Analyze this image and classify it into ONE of the following categories and subcategories.

Available Categories and Subcategories:
{category_list}

Instructions:
1. Identify the object in the image.
2. Match it to the MOST APPROPRIATE category and subcategory.
3. Return ONLY a JSON object:
{{"category": "Category Name", "subcategory": "Subcategory Name"}}

If no subcategory applies, use "General".
"""

        # API Call
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )

        result_text = response.choices[0].message.content.strip()

        # Remove markdown formatting if any
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        result_text = result_text.strip()

        classification = json.loads(result_text)
        return classification.get("category"), classification.get("subcategory"), None

    except Exception as e:
        return None, None, str(e)

def save_report(report):
    """Save report to CSV safely"""
    try:
        df = pd.DataFrame([report])
        if os.path.exists(CSV_FILE):
            df_existing = pd.read_csv(CSV_FILE)
            df = pd.concat([df_existing, df], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving report: {str(e)}")
        return False

def save_uploaded_file(uploaded_file):
    """Save uploaded file to local directory"""
    try:
        if uploaded_file:
            if not os.path.exists("uploads"):
                os.makedirs("uploads")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{uploaded_file.name}"
            filepath = os.path.join("uploads", filename)

            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())

            return filename
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart Swachh Portal",
    page_icon="♻️",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    st.markdown("---")
    st.info("Upload an image to auto-classify it into waste category.")
    st.caption("Your API key and images are not stored permanently.")

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("♻️ Smart Swachh – Citizen Waste Reporting Portal")
st.markdown("### 🧾 Submit Waste Collection Request")
st.markdown("---")

# ---------------------------------------------------------
# STEP 1: AI Classification (Outside Form)
# ---------------------------------------------------------
st.header("📸 Upload Image for Auto-Classification")

uploaded_photo = st.file_uploader(
    "Upload an image of the item *",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image for automatic classification"
)

if uploaded_photo:
    col_img1, col_img2 = st.columns([1, 2])
    with col_img1:
        st.image(uploaded_photo, caption="Uploaded Image", use_container_width=True)
    with col_img2:
        if api_key:
            if st.button("🤖 Classify Image with AI"):
                with st.spinner("Analyzing image..."):
                    category, subcategory, error = classify_image_with_openai(uploaded_photo, api_key)
                    if error:
                        st.error(f"Classification error: {error}")
                    else:
                        st.session_state.classified_category = category
                        st.session_state.classified_subcategory = subcategory
                        st.success(f"✅ Classified as: **{category} → {subcategory}**")
        else:
            st.warning("⚠️ Please enter your OpenAI API key in the sidebar.")

st.markdown("---")

# ---------------------------------------------------------
# STEP 2: Waste Report Form
# ---------------------------------------------------------
with st.form(key="waste_report_form", clear_on_submit=True):
    # Section 1: Personal
    st.header("👤 Personal Details")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name *")
        phone = st.text_input("Phone Number *", placeholder="10-digit number")
    with col2:
        email = st.text_input("Email ID *")

    st.markdown("---")

    # Section 2: Location
    st.header("📍 Location Details")
    col3, col4 = st.columns(2)
    with col3:
        lat = st.text_input("Latitude (optional)")
    with col4:
        lon = st.text_input("Longitude (optional)")

    address = st.text_area("Full Address *", height=80)
    landmark = st.text_input("Landmark (optional)")

    st.markdown("---")

    # Section 3: Waste Details
    st.header("🗑️ Item Details")

    if st.session_state.classified_category:
        st.success(f"🤖 AI Classification: {st.session_state.classified_category} → {st.session_state.classified_subcategory}")

    col5, col6 = st.columns(2)
    with col5:
        category_list = ["Select Category"] + list(CATEGORIES.keys())
        default_category_idx = 0
        if st.session_state.classified_category and st.session_state.classified_category in CATEGORIES:
            default_category_idx = category_list.index(st.session_state.classified_category)

        category = st.selectbox("Category *", category_list, index=default_category_idx)

        if category != "Select Category":
            subcategories = CATEGORIES.get(category, [])
            if not subcategories:
                subcategory = "General"
                st.info("No subcategories available.")
            else:
                default_subcat_idx = 0
                if (st.session_state.classified_subcategory and
                    st.session_state.classified_subcategory in subcategories):
                    default_subcat_idx = subcategories.index(st.session_state.classified_subcategory)
                subcategory = st.selectbox("Subcategory *", subcategories, index=default_subcat_idx)
        else:
            subcategory = "N/A"

    with col6:
        volume = st.slider("Approx. Volume (L) *", 1, 500, 10)
        weight = st.slider("Approx. Weight (Kg) *", 1, 200, 5)

    st.markdown("---")

    # Collection Window
    st.header("🕒 Collection Window")
    col7, col8 = st.columns(2)
    with col7:
        collection_date = st.date_input("Preferred Date *", min_value=date.today())
    with col8:
        collection_time = st.time_input("Preferred Time *", value=time(9, 0))

    st.markdown("---")

    # Notes
    st.header("📝 Additional Notes")
    notes = st.text_area("Any special instructions?", height=70)

    st.markdown("---")
    submit_button = st.form_submit_button("📤 Submit Report", use_container_width=True)

# ---------------------------------------------------------
# Submission Handling
# ---------------------------------------------------------
if submit_button:
    errors = []
    if not name.strip():
        errors.append("❌ Full Name is required")
    if not phone or not validate_phone(phone):
        errors.append("❌ Invalid phone number")
    if not email or not validate_email(email):
        errors.append("❌ Invalid email ID")
    if not address.strip():
        errors.append("❌ Address is required")
    if not uploaded_photo:
        errors.append("❌ Image upload required")
    if category == "Select Category":
        errors.append("❌ Please select a valid Category")
    if (lat or lon) and not validate_coordinates(lat, lon):
        errors.append("❌ Invalid coordinates")

    if errors:
        st.error("### ⚠️ Please fix the following:")
        for e in errors:
            st.error(e)
    else:
        photo_filename = save_uploaded_file(uploaded_photo)
        report = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Name": name.strip(),
            "Phone": phone,
            "Email": email.strip(),
            "Latitude": lat or "N/A",
            "Longitude": lon or "N/A",
            "Address": address.strip(),
            "Landmark": landmark or "N/A",
            "Category": category,
            "Subcategory": subcategory,
            "AI_Classified": "Yes" if st.session_state.classified_category else "No",
            "Volume(L)": volume,
            "Weight(Kg)": weight,
            "Collection_Date": collection_date.strftime("%Y-%m-%d"),
            "Collection_Time": collection_time.strftime("%H:%M"),
            "Photo": photo_filename or "N/A",
            "Notes": notes.strip() or "N/A"
        }

        if save_report(report):
            st.success("### ✅ Report submitted successfully!")
            st.balloons()
            st.write("#### 📋 Summary:")
            st.json(report)

            # Reset classification
            st.session_state.classified_category = None
            st.session_state.classified_subcategory = None
            st.session_state.form_submitted = True

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>♻️ Smart Swachh Initiative | Making cities cleaner, one report at a time</p>
        <p>For queries: support@smartswachh.gov.in | Helpline: 1800-XXX-XXXX</p>
    </div>
    """,
    unsafe_allow_html=True
)
