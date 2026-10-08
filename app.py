import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Land Price Prediction",
    page_icon="🌾",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 20px;
    }

    .prediction-box {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #ddd;
        text-align: center;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    import requests
    from io import BytesIO

    url = "https://github.com/Vinayyadav0510/land-price-prediction/releases/download/v1.0/land_price_model.pkl"
    response = requests.get(url)
    response.raise_for_status()

    return joblib.load(BytesIO(response.content))


model = load_model()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "Land_data.csv"
    )


df = load_data()



# ============================================================
# DATASET COLUMNS
# ============================================================

numerical_columns = [
    "City Dist",
    "Road Dist",
    "Area",
    "Town Dist",
    "Market Dist"
]

categorical_columns = [
    "Land Type",
    "Soil",
    "Water",
    "Road",
    "Electricity",
    "Irrigation",
    "Crop",
    "Ownership"
]


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🌾 Land Price Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered land price prediction and analytics dashboard'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌾 Land Price AI")

st.sidebar.markdown(
    "### Navigation"
)

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Prediction",
        "📊 Dashboard",
        "🔎 Data Explorer"
    ]
)


# ============================================================
# ============================================================
# PAGE 1 — PREDICTION
# ============================================================
# ============================================================

if page == "🏠 Prediction":

    st.header("🔮 Predict Land Price")

    st.write(
        "Enter the land details below and the machine learning "
        "model will estimate the land price."
    )

    st.divider()


    # ========================================================
    # INPUT FORM
    # ========================================================

    left, right = st.columns(2)


    # --------------------------------------------------------
    # LEFT SIDE
    # --------------------------------------------------------

    with left:

        st.subheader("📍 Location & Land")

        city = st.number_input(
            "City Distance (km)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.1
        )

        road_dist = st.number_input(
            "Road Distance (km)",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.1
        )

        land = st.selectbox(
            "Land Type",
            [
                "Agricultural",
                "Residential",
                "Commercial"
            ]
        )

        soil = st.selectbox(
            "Soil Quality",
            [
                "Low",
                "Medium",
                "High"
            ]
        )

        water = st.selectbox(
            "Water Source",
            [
                "Borewell",
                "Canal",
                "Municipal",
                "None"
            ]
        )

        road = st.selectbox(
            "Road Access",
            [
                "Yes",
                "No"
            ]
        )

        electricity = st.selectbox(
            "Electricity",
            [
                "Yes",
                "No"
            ]
        )


    # --------------------------------------------------------
    # RIGHT SIDE
    # --------------------------------------------------------

    with right:

        st.subheader("🌱 Agricultural Details")

        irrigation = st.selectbox(
            "Irrigation",
            [
                "Yes",
                "No"
            ]
        )

        area = st.number_input(
            "Area (Acres)",
            min_value=0.1,
            max_value=100.0,
            value=5.0,
            step=0.1
        )

        town = st.number_input(
            "Town Distance (km)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.1
        )

        market = st.number_input(
            "Market Distance (km)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.1
        )

        crop = st.selectbox(
            "Crop",
            [
                "Rice",
                "Cotton",
                "Vegetables",
                "Maize",
                "Groundnut",
                "Millets",
                "None"
            ]
        )

        ownership = st.selectbox(
            "Ownership",
            [
                "Yes",
                "No"
            ]
        )


    st.divider()


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    predict_button = st.button(
        "🔮 Predict Land Price",
        type="primary",
        use_container_width=True
    )


    if predict_button:

        # ====================================================
        # CREATE INPUT DATAFRAME
        # ====================================================

        input_df = pd.DataFrame({

            "City Dist": [city],

            "Road Dist": [road_dist],

            "Land Type": [land],

            "Soil": [soil],

            "Water": [water],

            "Road": [road],

            "Electricity": [electricity],

            "Irrigation": [irrigation],

            "Area": [area],

            "Town Dist": [town],

            "Market Dist": [market],

            "Crop": [crop],

            "Ownership": [ownership]

        })


        try:

            # =================================================
            # STEP 1
            # BUILD MODEL INPUT
            # =================================================
            # The saved model is a full scikit-learn Pipeline
            # (OneHotEncoder + RandomForestRegressor), so the
            # raw input row is passed straight through — the
            # pipeline handles categorical encoding internally.
            # If an older, non-pipeline model is loaded instead
            # (one that only has feature_names_in_ from a
            # manually one-hot-encoded training set), fall back
            # to encoding the input the same way here.

            if hasattr(model, "named_steps"):

                input_encoded = input_df.copy()

            elif hasattr(model, "feature_names_in_"):

                input_encoded = pd.get_dummies(
                    input_df,
                    columns=categorical_columns
                )

                model_features = list(
                    model.feature_names_in_
                )

                input_encoded = input_encoded.reindex(
                    columns=model_features,
                    fill_value=0
                ).astype(float)

            else:

                st.error(
                    "❌ The saved model does not contain "
                    "training feature names."
                )

                st.info(
                    "Please retrain the model and save it "
                    "again using the same preprocessing."
                )

                st.stop()


            # =================================================
            # STEP 2
            # PREDICT
            # =================================================

            prediction = model.predict(
                input_encoded
            )[0]


            # =================================================
            # RESULT
            # =================================================

            st.markdown(
                f"""
                <div class="prediction-box">

                <h2>💰 Estimated Land Price</h2>

                <h1>₹ {prediction:,.2f} Lakhs</h1>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.write("")


            # =================================================
            # METRICS
            # =================================================

            metric1, metric2, metric3 = st.columns(3)


            with metric1:

                st.metric(
                    "Estimated Price",
                    f"₹ {prediction:,.2f} L"
                )


            with metric2:

                st.metric(
                    "Land Area",
                    f"{area:.2f} Acres"
                )


            with metric3:

                price_per_acre = (
                    prediction / area
                )

                st.metric(
                    "Price / Acre",
                    f"₹ {price_per_acre:,.2f} L"
                )


            st.divider()


            # =================================================
            # INPUT SUMMARY
            # =================================================

            st.subheader(
                "📋 Property Details"
            )

            st.dataframe(
                input_df,
                use_container_width=True,
                hide_index=True
            )


            # =================================================
            # MODEL INPUT
            # =================================================

            with st.expander(
                "🔧 View Encoded Model Input"
            ):

                st.dataframe(
                    input_encoded,
                    use_container_width=True
                )


        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.error(
                f"Error: {str(e)}"
            )

            st.info(
                "This usually means that the saved model "
                "was trained with a different preprocessing "
                "method."
            )


# ============================================================
# ============================================================
# PAGE 2 — DASHBOARD
# ============================================================
# ============================================================

elif page == "📊 Dashboard":

    st.header(
        "📊 Land Price Analytics Dashboard"
    )

    st.write(
        "Explore land prices based on location, land type, "
        "soil, crops, water availability and other factors."
    )

    st.divider()


    # ========================================================
    # KPI SECTION
    # ========================================================

    total_records = len(df)

    average_price = df["Price"].mean()

    maximum_price = df["Price"].max()

    minimum_price = df["Price"].min()

    median_price = df["Price"].median()


    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)


    with kpi1:

        st.metric(
            "📑 Records",
            f"{total_records:,}"
        )


    with kpi2:

        st.metric(
            "💰 Average Price",
            f"₹ {average_price:,.2f} L"
        )


    with kpi3:

        st.metric(
            "📈 Highest Price",
            f"₹ {maximum_price:,.2f} L"
        )


    with kpi4:

        st.metric(
            "📉 Lowest Price",
            f"₹ {minimum_price:,.2f} L"
        )


    with kpi5:

        st.metric(
            "📊 Median Price",
            f"₹ {median_price:,.2f} L"
        )


    st.divider()


    # ========================================================
    # FILTERS
    # ========================================================

    st.subheader(
        "🔎 Dashboard Filters"
    )


    filter1, filter2, filter3 = st.columns(3)


    with filter1:

        land_filter = st.multiselect(
            "Land Type",
            sorted(
                df["Land Type"].dropna().unique()
            ),
            default=list(
                df["Land Type"].dropna().unique()
            )
        )


    with filter2:

        soil_filter = st.multiselect(
            "Soil",
            sorted(
                df["Soil"].dropna().unique()
            ),
            default=list(
                df["Soil"].dropna().unique()
            )
        )


    with filter3:

        crop_filter = st.multiselect(
            "Crop",
            sorted(
                df["Crop"].dropna().unique()
            ),
            default=list(
                df["Crop"].dropna().unique()
            )
        )


    # ========================================================
    # FILTER DATA
    # ========================================================

    filtered_df = df[
        df["Land Type"].isin(
            land_filter
        )
        &
        df["Soil"].isin(
            soil_filter
        )
        &
        df["Crop"].isin(
            crop_filter
        )
    ].copy()


    st.caption(
        f"Showing {len(filtered_df):,} records"
    )


    # ========================================================
    # CHECK EMPTY DATA
    # ========================================================

    if len(filtered_df) == 0:

        st.warning(
            "No data available for the selected filters."
        )

        st.stop()


    # ========================================================
    # PRICE DISTRIBUTION
    # ========================================================

    st.subheader(
        "💰 Land Price Distribution"
    )


    fig, ax = plt.subplots(
        figsize=(12, 5)
    )


    ax.hist(
        filtered_df["Price"],
        bins=30,
        edgecolor="black"
    )


    ax.set_title(
        "Distribution of Land Prices"
    )

    ax.set_xlabel(
        "Price (Lakhs)"
    )

    ax.set_ylabel(
        "Number of Properties"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )


    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    # ========================================================
    # LAND TYPE + CROP
    # ========================================================

    chart1, chart2 = st.columns(2)


    # --------------------------------------------------------
    # LAND TYPE
    # --------------------------------------------------------

    with chart1:

        st.subheader(
            "🏠 Average Price by Land Type"
        )


        land_price = (
            filtered_df
            .groupby("Land Type")["Price"]
            .mean()
            .sort_values(
                ascending=False
            )
        )


        fig, ax = plt.subplots(
            figsize=(7, 5)
        )


        land_price.plot(
            kind="bar",
            ax=ax
        )


        ax.set_title(
            "Average Price by Land Type"
        )

        ax.set_xlabel(
            "Land Type"
        )

        ax.set_ylabel(
            "Average Price (Lakhs)"
        )

        ax.tick_params(
            axis="x",
            rotation=0
        )

        ax.grid(
            axis="y",
            alpha=0.3
        )


        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    # --------------------------------------------------------
    # CROP
    # --------------------------------------------------------

    with chart2:

        st.subheader(
            "🌱 Average Price by Crop"
        )


        crop_price = (
            filtered_df
            .groupby("Crop")["Price"]
            .mean()
            .sort_values(
                ascending=False
            )
        )


        fig, ax = plt.subplots(
            figsize=(7, 5)
        )


        crop_price.plot(
            kind="bar",
            ax=ax
        )


        ax.set_title(
            "Average Price by Crop"
        )

        ax.set_xlabel(
            "Crop"
        )

        ax.set_ylabel(
            "Average Price (Lakhs)"
        )

        ax.tick_params(
            axis="x",
            rotation=45
        )

        ax.grid(
            axis="y",
            alpha=0.3
        )


        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    # ========================================================
    # SOIL + WATER
    # ========================================================

    chart3, chart4 = st.columns(2)


    # --------------------------------------------------------
    # SOIL
    # --------------------------------------------------------

    with chart3:

        st.subheader(
            "🌍 Average Price by Soil"
        )


        soil_price = (
            filtered_df
            .groupby("Soil")["Price"]
            .mean()
            .sort_values(
                ascending=False
            )
        )


        fig, ax = plt.subplots(
            figsize=(7, 5)
        )


        soil_price.plot(
            kind="bar",
            ax=ax
        )


        ax.set_title(
            "Average Price by Soil Quality"
        )

        ax.set_xlabel(
            "Soil Quality"
        )

        ax.set_ylabel(
            "Average Price (Lakhs)"
        )

        ax.tick_params(
            axis="x",
            rotation=0
        )

        ax.grid(
            axis="y",
            alpha=0.3
        )


        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    # --------------------------------------------------------
    # WATER
    # --------------------------------------------------------

    with chart4:

        st.subheader(
            "💧 Average Price by Water Source"
        )


        water_price = (
            filtered_df
            .groupby("Water")["Price"]
            .mean()
            .sort_values(
                ascending=False
            )
        )


        fig, ax = plt.subplots(
            figsize=(7, 5)
        )


        water_price.plot(
            kind="bar",
            ax=ax
        )


        ax.set_title(
            "Average Price by Water Source"
        )

        ax.set_xlabel(
            "Water Source"
        )

        ax.set_ylabel(
            "Average Price (Lakhs)"
        )

        ax.tick_params(
            axis="x",
            rotation=45
        )

        ax.grid(
            axis="y",
            alpha=0.3
        )


        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    # ========================================================
    # AREA VS PRICE
    # ========================================================

    st.subheader(
        "📐 Area vs Price"
    )


    fig, ax = plt.subplots(
        figsize=(12, 5)
    )


    ax.scatter(
        filtered_df["Area"],
        filtered_df["Price"],
        alpha=0.6
    )


    ax.set_title(
        "Land Area vs Price"
    )

    ax.set_xlabel(
        "Area (Acres)"
    )

    ax.set_ylabel(
        "Price (Lakhs)"
    )

    ax.grid(
        alpha=0.3
    )


    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    # ========================================================
    # CITY DISTANCE VS PRICE
    # ========================================================

    st.subheader(
        "🏙️ City Distance vs Price"
    )


    fig, ax = plt.subplots(
        figsize=(12, 5)
    )


    ax.scatter(
        filtered_df["City Dist"],
        filtered_df["Price"],
        alpha=0.6
    )


    ax.set_title(
        "City Distance vs Land Price"
    )

    ax.set_xlabel(
        "City Distance (km)"
    )

    ax.set_ylabel(
        "Price (Lakhs)"
    )

    ax.grid(
        alpha=0.3
    )


    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    # ========================================================
    # MARKET DISTANCE VS PRICE
    # ========================================================

    st.subheader(
        "🛒 Market Distance vs Price"
    )


    fig, ax = plt.subplots(
        figsize=(12, 5)
    )


    ax.scatter(
        filtered_df["Market Dist"],
        filtered_df["Price"],
        alpha=0.6
    )


    ax.set_title(
        "Market Distance vs Land Price"
    )

    ax.set_xlabel(
        "Market Distance (km)"
    )

    ax.set_ylabel(
        "Price (Lakhs)"
    )

    ax.grid(
        alpha=0.3
    )


    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    # ========================================================
    # ROAD + IRRIGATION
    # ========================================================

    chart5, chart6 = st.columns(2)


    # --------------------------------------------------------
    # ROAD
    # --------------------------------------------------------

    with chart5:

        st.subheader(
            "🛣️ Road Access"
        )


        road_price = (
            filtered_df
            .groupby("Road")["Price"]
            .mean()
        )


        fig, ax = plt.subplots(
            figsize=(7, 5)
        )


        road_price.plot(
            kind="bar",
            ax=ax
        )


        ax.set_title(
            "Average Price by Road Access"
        )

        ax.set_xlabel(
            "Road Access"
        )

        ax.set_ylabel(
            "Average Price (Lakhs)"
        )

        ax.tick_params(
            axis="x",
            rotation=0
        )

        ax.grid(
            axis="y",
            alpha=0.3
        )


        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    # --------------------------------------------------------
    # IRRIGATION
    # --------------------------------------------------------

    with chart6:

        st.subheader(
            "💦 Irrigation"
        )


        irrigation_price = (
            filtered_df
            .groupby("Irrigation")["Price"]
            .mean()
        )


        fig, ax = plt.subplots(
            figsize=(7, 5)
        )


        irrigation_price.plot(
            kind="bar",
            ax=ax
        )


        ax.set_title(
            "Average Price by Irrigation"
        )

        ax.set_xlabel(
            "Irrigation"
        )

        ax.set_ylabel(
            "Average Price (Lakhs)"
        )

        ax.tick_params(
            axis="x",
            rotation=0
        )

        ax.grid(
            axis="y",
            alpha=0.3
        )


        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    # ========================================================
    # TOP 10 EXPENSIVE LAND
    # ========================================================

    st.divider()

    st.subheader(
        "🏆 Top 10 Highest-Priced Properties"
    )


    top_lands = (
        filtered_df
        .sort_values(
            by="Price",
            ascending=False
        )
        .head(10)
    )


    st.dataframe(
        top_lands,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ============================================================
# PAGE 3 — DATA EXPLORER
# ============================================================
# ============================================================

elif page == "🔎 Data Explorer":

    st.header(
        "🔎 Land Dataset Explorer"
    )

    st.write(
        "Explore the dataset used by the application."
    )

    st.divider()


    # ========================================================
    # DATASET KPIs
    # ========================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Rows",
            f"{df.shape[0]:,}"
        )


    with col2:

        st.metric(
            "Columns",
            f"{df.shape[1]:,}"
        )


    with col3:

        missing_values = int(
            df.isnull().sum().sum()
        )

        st.metric(
            "Missing Values",
            f"{missing_values:,}"
        )


    st.divider()


    # ========================================================
    # DATASET
    # ========================================================

    st.subheader(
        "📋 Complete Dataset"
    )


    st.dataframe(
        df,
        use_container_width=True,
        height=500
    )


    # ========================================================
    # STATISTICS
    # ========================================================

    st.subheader(
        "📊 Statistical Summary"
    )


    st.dataframe(
        df.describe(),
        use_container_width=True
    )


    # ========================================================
    # COLUMN INFORMATION
    # ========================================================

    st.subheader(
        "🧾 Column Information"
    )


    column_info = pd.DataFrame({

        "Column": df.columns,

        "Data Type": [
            str(dtype)
            for dtype in df.dtypes
        ],

        "Missing Values": [
            df[column].isnull().sum()
            for column in df.columns
        ],

        "Unique Values": [
            df[column].nunique()
            for column in df.columns
        ]

    })


    st.dataframe(
        column_info,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🌾 Land Price Prediction | "
    "Machine Learning + Streamlit | "
    "Developed with Python"
)
