# complete_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Set page config
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)


@st.cache_resource
def load_feature_names():
    """Load feature names from the pickle file"""
    try:
        with open('best_churn_model.pkl', 'rb') as file:
            feature_names = pickle.load(file)
        return feature_names
    except Exception as e:
        st.error(f"Error loading feature names: {e}")
        return None


def create_demo_model(feature_names):
    """Create a demo model that matches your feature structure"""
    st.info("🔧 Using intelligent demo model based on your feature structure")

    # Create a realistic demo model
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    # Create training data that matches your feature structure
    n_samples = 1000
    X_demo = np.zeros((n_samples, len(feature_names)))

    # Add some realistic patterns based on feature names
    for i, feature in enumerate(feature_names):
        if 'tenure' in feature:
            X_demo[:, i] = np.random.randint(0, 72, n_samples)
        elif 'charge' in feature.lower():
            X_demo[:, i] = np.random.uniform(20, 120, n_samples)
        elif any(term in feature.lower() for term in ['senior', 'partner', 'dependents', 'phoneservice']):
            X_demo[:, i] = np.random.randint(0, 2, n_samples)
        else:
            # For categorical encoded features
            X_demo[:, i] = np.random.randint(0, 2, n_samples)

    # Create realistic target variable
    y_demo = np.zeros(n_samples)

    # Realistic churn rules based on common patterns
    for i in range(n_samples):
        tenure = X_demo[i, list(feature_names).index('tenure')] if 'tenure' in feature_names else 12
        monthly_charges = X_demo[
            i, list(feature_names).index('MonthlyCharges')] if 'MonthlyCharges' in feature_names else 65

        churn_prob = 0.0
        if tenure < 6:
            churn_prob += 0.4
        if monthly_charges > 80:
            churn_prob += 0.3
        if 'Contract_Month-to-month' in feature_names and X_demo[
            i, list(feature_names).index('Contract_Month-to-month')] == 1:
            churn_prob += 0.3

        y_demo[i] = 1 if np.random.random() < churn_prob else 0

    model.fit(X_demo, y_demo)
    return model


# Load feature names
feature_names = load_feature_names()

if feature_names is not None:
    st.success(f"✅ Loaded feature structure with {len(feature_names)} features")
    st.write("**Features detected:**", ", ".join(feature_names[:10]) + "...")

    # Create model based on feature structure
    model = create_demo_model(feature_names)
else:
    st.error("❌ Could not load feature names")
    st.stop()

# Main dashboard
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("---")

# Navigation
page = st.sidebar.selectbox("Navigation", ["Single Prediction", "Batch Prediction", "Feature Analysis"])

if page == "Single Prediction":
    st.header("🔍 Single Customer Prediction")

    # Create input form based on actual feature names
    st.info(f"Based on your model's {len(feature_names)} features")

    # Group features by type for better organization
    demographic_features = [f for f in feature_names if f in ['gender', 'SeniorCitizen', 'Partner', 'Dependents']]
    service_features = [f for f in feature_names if any(term in f for term in ['Service', 'Contract', 'Payment'])]
    charge_features = [f for f in feature_names if any(term in f for term in ['Charge', 'charge'])]
    tenure_features = [f for f in feature_names if 'tenure' in f]
    other_features = [f for f in feature_names if
                      f not in demographic_features + service_features + charge_features + tenure_features]

    # Create input data dictionary
    input_data = {}

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographics & Basic Info")

        # Demographic features
        if 'gender' in feature_names:
            gender = st.selectbox("Gender", ["Female", "Male"])
            input_data['gender'] = 1 if gender == "Male" else 0

        if 'SeniorCitizen' in feature_names:
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            input_data['SeniorCitizen'] = 1 if senior == "Yes" else 0

        if 'Partner' in feature_names:
            partner = st.selectbox("Partner", ["No", "Yes"])
            input_data['Partner'] = 1 if partner == "Yes" else 0

        if 'Dependents' in feature_names:
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            input_data['Dependents'] = 1 if dependents == "Yes" else 0

        # Tenure
        if 'tenure' in feature_names:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            input_data['tenure'] = tenure

    with col2:
        st.subheader("Services & Contract")

        # Service features
        if 'PhoneService' in feature_names:
            phone = st.selectbox("Phone Service", ["No", "Yes"])
            input_data['PhoneService'] = 1 if phone == "Yes" else 0

        # Contract type
        contract_options = ["Month-to-month", "One year", "Two year"]
        contract = st.selectbox("Contract Type", contract_options)

        for opt in contract_options:
            feature_name = f"Contract_{opt.replace(' ', '')}"
            if feature_name in feature_names:
                input_data[feature_name] = 1 if contract == opt else 0

        # Payment method
        payment_options = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        payment = st.selectbox("Payment Method", payment_options)

        for opt in payment_options:
            feature_name = f"PaymentMethod_{opt.split(' ')[0]}"
            if feature_name in feature_names:
                input_data[feature_name] = 1 if payment == opt else 0

        if 'PaperlessBilling' in feature_names:
            paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
            input_data['PaperlessBilling'] = 1 if paperless == "Yes" else 0

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Charges & Internet Services")

        # Charge features
        if 'MonthlyCharges' in feature_names:
            monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
            input_data['MonthlyCharges'] = monthly

        if 'TotalCharges' in feature_names:
            total = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)
            input_data['TotalCharges'] = total

        # Internet service
        internet_options = ["DSL", "Fiber optic", "No"]
        internet = st.selectbox("Internet Service", internet_options)

        for opt in internet_options:
            feature_name = f"InternetService_{opt.replace(' ', '')}"
            if feature_name in feature_names:
                input_data[feature_name] = 1 if internet == opt else 0

    with col4:
        st.subheader("Additional Services")

        # Additional services
        service_categories = {
            'MultipleLines': ["No", "No phone service", "Yes"],
            'OnlineSecurity': ["No", "No internet service", "Yes"],
            'OnlineBackup': ["No", "No internet service", "Yes"],
            'DeviceProtection': ["No", "No internet service", "Yes"],
            'TechSupport': ["No", "No internet service", "Yes"],
            'StreamingTV': ["No", "No internet service", "Yes"],
            'StreamingMovies': ["No", "No internet service", "Yes"]
        }

        for service, options in service_categories.items():
            for opt in options:
                feature_name = f"{service}_{opt.replace(' ', '')}"
                if feature_name in feature_names:
                    selection = st.selectbox(service, options, key=service)
                    input_data[feature_name] = 1 if selection == opt else 0
                    break

    # Derived features
    st.subheader("Derived Features")
    col5, col6 = st.columns(2)

    with col5:
        if 'charge_per_month' in feature_names:
            charge_pm = st.number_input("Charge per Month", 0.0, 200.0, 65.0)
            input_data['charge_per_month'] = charge_pm

        if 'total_services' in feature_names:
            total_services = st.slider("Total Services", 0, 10, 3)
            input_data['total_services'] = total_services

    with col6:
        if 'high_spender' in feature_names:
            high_spender = st.selectbox("High Spender", ["No", "Yes"])
            input_data['high_spender'] = 1 if high_spender == "Yes" else 0

        if 'new_customer' in feature_names:
            new_customer = st.selectbox("New Customer", ["No", "Yes"])
            input_data['new_customer'] = 1 if new_customer == "Yes" else 0

        if 'simple_contract' in feature_names:
            simple_contract = st.selectbox("Simple Contract", ["No", "Yes"])
            input_data['simple_contract'] = 1 if simple_contract == "Yes" else 0

    # Tenure groups
    if any('tenure_group' in f for f in feature_names):
        st.write("Tenure Groups (auto-calculated):")
        if 'tenure_group_1-2_years' in feature_names:
            input_data['tenure_group_1-2_years'] = 1 if 12 <= tenure < 24 else 0
        if 'tenure_group_2-3_years' in feature_names:
            input_data['tenure_group_2-3_years'] = 1 if 24 <= tenure < 36 else 0
        if 'tenure_group_3-4_years' in feature_names:
            input_data['tenure_group_3-4_years'] = 1 if 36 <= tenure < 48 else 0
        if 'tenure_group_4+_years' in feature_names:
            input_data['tenure_group_4+_years'] = 1 if tenure >= 48 else 0

    # Prediction
    if st.button("Predict Churn", type="primary", use_container_width=True):
        # Create feature vector in correct order
        feature_vector = np.zeros(len(feature_names))
        for i, feature in enumerate(feature_names):
            feature_vector[i] = input_data.get(feature, 0)

        # Make prediction
        prediction = model.predict([feature_vector])[0]
        probabilities = model.predict_proba([feature_vector])[0]

        # Display results
        st.markdown("---")
        st.subheader("🎯 Prediction Results")

        col7, col8 = st.columns(2)

        with col7:
            if prediction == 1:
                st.error("🚨 **Prediction: HIGH RISK OF CHURN**")
            else:
                st.success("✅ **Prediction: LOW RISK OF CHURN**")

            st.metric("Churn Probability", f"{probabilities[1]:.1%}")
            st.metric("Retention Probability", f"{probabilities[0]:.1%}")

            # Risk level
            risk_level = "High" if probabilities[1] > 0.7 else "Medium" if probabilities[1] > 0.3 else "Low"
            st.metric("Risk Level", risk_level)

        with col8:
            # Visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Probability bars
            ax1.bar(['Retention', 'Churn'], [probabilities[0], probabilities[1]],
                    color=['green', 'red'], alpha=0.7)
            ax1.set_ylabel('Probability')
            ax1.set_ylim(0, 1)
            ax1.set_title('Churn Probability')

            # Risk gauge
            risk_angle = probabilities[1] * 180  # Convert to gauge angle
            ax2.set_xlim(-1, 1)
            ax2.set_ylim(0, 1)
            ax2.add_patch(plt.Circle((0, 0), 0.8, fill=False, color='black', linewidth=2))
            ax2.plot([0, 0.8 * np.sin(np.radians(risk_angle))],
                     [0, 0.8 * np.cos(np.radians(risk_angle))], 'r-', linewidth=3)
            ax2.set_title('Risk Gauge')
            ax2.axis('off')

            plt.tight_layout()
            st.pyplot(fig)

        # Risk factors
        st.subheader("📊 Risk Factors Analysis")
        risk_factors = []

        if tenure < 6:
            risk_factors.append("🆕 **New customer** (tenure < 6 months)")
        if contract == "Month-to-month":
            risk_factors.append("📝 **Month-to-month contract** (higher churn risk)")
        if monthly > 80:
            risk_factors.append("💰 **High monthly charges** (> $80)")
        if internet == "Fiber optic" and monthly > 70:
            risk_factors.append("🌐 **Premium internet service** with high cost")

        if risk_factors:
            st.warning("**Identified Risk Factors:**")
            for factor in risk_factors:
                st.write(f"- {factor}")
        else:
            st.success("✅ No major risk factors identified")

elif page == "Batch Prediction":
    st.header("📁 Batch Prediction")
    st.info("Upload a CSV file with customer data for batch predictions")

    uploaded_file = st.file_uploader("Choose CSV file", type="csv")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("### Uploaded Data Preview")
            st.dataframe(data.head())

            if st.button("Process Batch Prediction"):
                with st.spinner("Processing..."):
                    # Process each row
                    predictions = []
                    probabilities = []

                    for _, row in data.iterrows():
                        # Create feature vector (simplified)
                        feature_vector = np.zeros(len(feature_names))
                        for i, feature in enumerate(feature_names):
                            feature_vector[i] = row.get(feature, 0)

                        pred = model.predict([feature_vector])[0]
                        proba = model.predict_proba([feature_vector])[0]

                        predictions.append(pred)
                        probabilities.append(proba[1])

                    # Add results to data
                    results = data.copy()
                    results['Churn_Prediction'] = predictions
                    results['Churn_Probability'] = probabilities

                    # Display results
                    st.success(f"✅ Processed {len(results)} customers")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Customers", len(results))
                    col2.metric("Predicted Churn", sum(predictions))
                    col3.metric("Churn Rate", f"{(sum(predictions) / len(results)):.1%}")

                    # Download
                    csv = results.to_csv(index=False)
                    st.download_button(
                        "📥 Download Predictions",
                        csv,
                        "churn_predictions.csv",
                        "text/csv"
                    )

        except Exception as e:
            st.error(f"Error: {e}")

else:  # Feature Analysis
    st.header("🔬 Feature Analysis")

    st.write("### Feature Structure")
    st.write(f"Your model uses **{len(feature_names)}** features:")

    # Group features
    feature_groups = {
        "Demographic": [f for f in feature_names if f in ['gender', 'SeniorCitizen', 'Partner', 'Dependents']],
        "Contract & Payment": [f for f in feature_names if any(term in f for term in ['Contract', 'Payment'])],
        "Internet Services": [f for f in feature_names if 'InternetService' in f],
        "Additional Services": [f for f in feature_names if any(term in f for term in
                                                                ['MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                                                                 'DeviceProtection', 'TechSupport', 'Streaming'])],
        "Charges & Tenure": [f for f in feature_names if any(term in f for term in ['charge', 'tenure'])]
    }

    for group, features in feature_groups.items():
        if features:
            with st.expander(f"{group} ({len(features)} features)"):
                st.write(", ".join(features))

    st.info(
        "💡 **Next Steps:** To use your actual trained model, you need the model file (.pkl) that contains the trained GradientBoostingClassifier, not just the feature names.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Using feature structure from your model")