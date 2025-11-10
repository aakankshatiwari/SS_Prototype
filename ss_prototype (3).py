# -*- coding: utf-8 -*-
"""
Smart Swachh Streamlit App
Uses Euri AI to auto-classify waste images into categories and subcategories.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import base64
import os
import json
from euriai.langchain import create_chat_model

# =========================
# CONFIGURATION
# =========================
CSV_FILE = "smart_swachh_reports.csv"

# Define categories and subcategories
CATEGORIES = {
    "Plastic Waste": ["Bottles", "Cups", "Packaging", "Bags", "Straws"],
    "Paper Waste": ["Newspapers", "Cardboard", "Books", "Tissues"],
    "Metal Waste": ["Cans", "Utensils", "Scrap", "Foil"],
    "Organic Waste": ["Food", "Leaves", "Garden", "Compostable"],
    "Glass Waste": ["Bottles", "Jars", "Broken Glass"],
    "E-Waste": ["Mobile", "Laptop", "Battery", "Cables"],
    "Textile Waste": ["Clothes", "Shoes", "Bags"],
    "Hazardous Waste": ["Paint", "Chemicals", "Medical", "Batteries"],
    "Construction Waste": ["Concrete", "Bricks", "Tiles", "Wood"],
    "Other": ["General", "Mixed", "Uncategorized"]
}

# =========================
# HELPER FUNCTIONS
# =========================

def save_report(data):
    """Append a single report row to the CSV file."""
    df_new = pd.DataFrame([data])
    if not os.path.exists(CSV_FILE):
        df_new.to_csv(CSV_FILE, index=False)
    else:
        df_existing = pd.read_csv(CSV_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(CSV_FILE, index=False)


def classify_image_with_euri(image_file, api_key):
    """
    Classify uploaded image into category/subcategory using Euri AI.
    Returns (category, subcategory, error_message)
    """
    try:
        # Convert image to base64
        image_file.seek(0)
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
        image_file.seek(0)

        # Create chat model
        chat_model = create_chat_model(
            api_key=api_key,
            model="gpt-4.1-nano",
            temperature=0.2
        )

        # Build prompt with category/subcategory structure
        category_list = "\n".join(
            [f"- {cat}: {', '.join(subs)}" for cat, subs in CATEGORIES.items()]
        )

        prompt = f"""
        You are a waste classification AI.
        Analyze the attached image (base64 format provided below) and determine the most suitable category and subcategory.

        Categories and Subcategories:
        {category_list}

        Respond only in JSON format:
        {{
            "category": "Category Name",
            "subcategory": "Subcategory Name"
        }}

        Base64 Image (truncated): {image_base64[:200]}...
        """

        # Send to Euri AI
        response = chat_model.invoke(prompt)
        result_text = response.content.strip()

        # Extract JSON safely
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.strip().startswith("json"):
                result_text = result_text[4:].strip()

        result_json = json.loads(result_text)
        category = result_json.get("category", "Other")
        subcategory = result_json.get("subcategory", "General")
        return category, subcategory, None

    except Exception as e:
        return None, None, str(e)


# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="Smart Swachh Report", page_icon="♻️", layout="wide")
st.title("♻️ Smart Swachh Report System")
st.caption("Upload an image and let Euri AI classify the waste category and subcategory automatically.")

# Sidebar for API Key
st.sidebar.header("🔐 API Configuration")
api_key = st.sidebar.text_input("Enter your Euri API Key", type="password")

# Report Form
with st.form("waste_form", clear_on_submit=True):
    st.subheader("🧾 Waste Report Form")

    col1, col2 = st.columns(2)
    with col1:
        reporter_name = st.text_input("Reporter Name")
        location = st.text_input("Location (City/Area)")
        date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with col2:
        uploaded_photo = st.file_uploader("Upload Waste Image", type=["jpg", "jpeg", "png"])

        auto_classify = st.checkbox("Auto-classify using Euri AI")

        category = st.selectbox("Waste Category", [""] + list(CATEGORIES.keys()))
        subcategory = st.selectbox("Subcategory", [""])

    # Auto-classification logic
    if auto_classify and uploaded_photo is not None:
        if not api_key:
            st.warning("Please enter your Euri API key in the sidebar to enable classification.")
        else:
            with st.spinner("Classifying image with Euri AI..."):
                category, subcategory, error = classify_image_with_euri(uploaded_photo, api_key)
                if error:
                    st.error(f"Classification failed: {error}")
                else:
                    st.success(f"✅ Classified as {category} → {subcategory}")
                    # Update UI fields automatically (for display only)
                    st.write(f"**Category:** {category}")
                    st.write(f"**Subcategory:** {subcategory}")

    # Description field
    description = st.text_area("Additional Description")

    # Submit button
    submitted = st.form_submit_button("Submit Report")

    if submitted:
        if not (reporter_name and location and uploaded_photo):
            st.error("Please fill all mandatory fields and upload an image.")
        else:
            # Prepare data row
            report_data = {
                "Reporter": reporter_name,
                "Location": location,
                "DateTime": date_time,
                "Category": category or "Unspecified",
                "Subcategory": subcategory or "Unspecified",
                "Description": description
            }
            save_report(report_data)
            st.success("✅ Report submitted successfully and saved to CSV!")


# =========================
# DATA DISPLAY SECTION
# =========================
st.subheader("📊 Submitted Reports")
if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
    st.dataframe(df, use_container_width=True)
else:
    st.info("No reports submitted yet.")
