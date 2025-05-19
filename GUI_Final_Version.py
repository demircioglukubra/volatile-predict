import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# === Load Models ===
@st.cache_resource
def load_models():
    ea_model, ea_scaler = joblib.load("Ea_model_LGB_best.pkl")
    A_model, A_scaler = joblib.load("A_model_LGB_best.pkl")
    final_model, final_scaler = joblib.load("FinalModel_baseline_Ea_A_LGB.pkl")
    return ea_model, ea_scaler, A_model, A_scaler, final_model, final_scaler


ea_model, ea_scaler, A_model, A_scaler, final_model, final_scaler = load_models()

# === Prediction Logic ===
def preprocess_and_predict(df, custom_category=None, custom_therm=None):
    eps = 1e-6
    cat2_map = {
        'Digestate': 'Biomass', 'Sewage': 'Biomass', 'HTC-MSW': 'Mix', 'Cellulose': 'Biomass',
        'Lignin': 'Biomass', 'Hemicellulose': 'Biomass', 'Wood': 'Biomass', 'Torr-Wood': 'Biomass',
        'Lignite': 'Coal', 'Torr-Coal': 'Coal', 'Rubber': 'Plastic', 'RDF1': 'Plastic',
        'RDF2': 'Mix', 'Digestate_PP': 'Mix1', 'Digestate_SCP': 'Mix1', 'Digestate_PE': 'Mix1'
    }
    therm_map = {
        'Digestate': 'Yes', 'Sewage': 'No', 'HTC-MSW': 'Yes', 'Cellulose': 'No', 'Lignin': 'No',
        'Hemicellulose': 'No', 'Wood': 'No', 'Torr-Wood': 'Yes', 'Lignite': 'No', 'Torr-Coal': 'Yes',
        'Rubber': 'No', 'RDF1': 'No', 'RDF2': 'No', 'Digestate_PP': 'No',
        'Digestate_SCP': 'No', 'Digestate_PE': 'No'
    }

    if custom_category is not None:
        df['fuel_category2'] = custom_category
        df['thermal_treatment'] = custom_therm
    else:
        df['fuel_category2'] = df['fuel_type'].map(cat2_map).fillna('Unknown')
        df['thermal_treatment'] = df['fuel_type'].map(therm_map).fillna('No')

    df = pd.get_dummies(df, columns=['fuel_category2', 'thermal_treatment'],
                        prefix=['fuel_cat2', 'th_treat'], drop_first=False)

    for col in ['fuel_cat2_Biomass', 'fuel_cat2_Coal', 'fuel_cat2_Mix', 'fuel_cat2_Plastic',
                'th_treat_No', 'th_treat_Yes']:
        if col not in df.columns:
            df[col] = 0

    df['vm_fc'] = df['vm'] / (df['fc'] + eps)
    df['oh_w'] = df['o'] / (df['h'] + eps)
    df['ac_fc'] = df['ac'] / (df['fc'] + eps)
    df['t_to_T'] = df['temperature'] / (df['heat_rate'] + eps)
    df['cl_ac'] = df['cl'] / (df['ac'] + eps)
    df['n_ac'] = df['n'] / (df['ac'] + eps)

    for col in ['vm_fc', 'ac_fc', 'cl_ac', 'n_ac', 't_to_T', 'oh_w', 'oc', 'hc']:
        df[f'log_{col}'] = np.log(df[col] + eps)

    # === Ea prediction ===
    X_ea = df[[f for f in ea_scaler.feature_names_in_ if f in df.columns]]
    df['EA_Arr'] = ea_model.predict(ea_scaler.transform(X_ea))

    # === A prediction ===
    X_A = df[[f for f in A_scaler.feature_names_in_ if f in df.columns]]
    df['A_Arr'] = A_model.predict(A_scaler.transform(X_A))

    # === Final prediction ===
    final_fs = ['hc', 'oc', 'vm_fc', 'temperature', 'heat_rate',
                'residence_time', 'pressure', 'EA_Arr', 'A_Arr']
    X_final = final_scaler.transform(df[final_fs])
    df['predicted_devol_yield'] = final_model.predict(X_final)

    return df[['temperature', 'EA_Arr', 'A_Arr', 'predicted_devol_yield']]


# === Streamlit UI ===
st.title("🔥 Volatile Release Predictor")
st.markdown("## Fuel & Operational Conditions Inputs")

col1, col2 = st.columns(2)

existing_fuels = [
    'Wood', 'Digestate', 'Sewage', 'HTC-MSW', 'Cellulose', 'Lignin',
    'Hemicellulose', 'Torr-Wood', 'Lignite', 'Torr-Coal', 'Rubber',
    'RDF1', 'RDF2', 'Digestate_PP', 'Digestate_SCP', 'Digestate_PE'
]

with col1:
    fuel_choice = st.selectbox("Fuel Type", existing_fuels + ["Other (New Fuel)"])
    if fuel_choice == "Other (New Fuel)":
        fuel_type = st.text_input("Enter New Fuel Name")
        custom_category = st.selectbox("Select Fuel Category", ["Biomass", "Coal", "Mix", "Plastic"])
        custom_therm = st.selectbox("Thermal Treatment?", ["Yes", "No"])
    else:
        fuel_type = fuel_choice
        custom_category = None
        custom_therm = None

    hc = st.number_input("H/C Ratio", value=1.5)
    oc = st.number_input("O/C Ratio", value=0.7)
    h = st.number_input("H (%)", value=6.0)
    o = st.number_input("O (%)", value=40.0)
    vm = st.number_input("Volatile Matter (%)", value=70.0)
    fc = st.number_input("Fixed Carbon (%)", value=20.0)

with col2:
    ac = st.number_input("Ash (%)", value=10.0)
    cl = st.number_input("Cl (%)", value=0.1)
    n = st.number_input("N (%)", value=0.5)
    heat_rate_input = st.text_input("Heating Rates (K/s, comma-separated)", value="10, 100, 1000")
    res_time = st.number_input("Residence Time (s)", value=2.0)
    pressure = st.number_input("Pressure (bar)", value=1.0)

st.markdown("### Temperature Range")
temp_min = st.number_input("Min Temp (°C)", value=400)
temp_max = st.number_input("Max Temp (°C)", value=800)
temp_steps = st.slider("Steps", 2, 20, 5)

submitted = st.button("Predict")
exp_file = st.file_uploader("Upload Experimental Data (CSV)", type=["csv"])

# === Plot Results ===
if submitted:
    try:
        heating_rates = [float(hr.strip()) for hr in heat_rate_input.split(',') if hr.strip()]
    except ValueError:
        st.error("Please enter numeric heating rates separated by commas.")
        st.stop()

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(heating_rates)))
    markers = ['s', 'o', 'v', '^', 'd', '<', '>']

    # Prepare experimental data (if provided)
    exp_df = None
    if exp_file is not None:
        exp_df = pd.read_csv(exp_file, delimiter=',')
        exp_df['fuel_type'] = exp_df['fuel_type'].astype(str).str.lower()
        fuel_type_lower = fuel_type.lower()

    for i, hr in enumerate(heating_rates):
        # Simulated Prediction
        rows = []
        for T in np.linspace(temp_min, temp_max, temp_steps):
            rows.append({
                'fuel_type': fuel_type, 'hc': hc, 'oc': oc, 'h': h, 'o': o,
                'vm': vm, 'fc': fc, 'ac': ac, 'cl': cl, 'n': n,
                'heat_rate': hr, 'residence_time': res_time,
                'pressure': pressure, 'temperature': T
            })
        df_hr = pd.DataFrame(rows)
        result_df = preprocess_and_predict(df_hr, custom_category, custom_therm)

        x = result_df['temperature']
        y = result_df['predicted_devol_yield'] * (1 - df_hr['ac'] / 100) * 100  # ash-free %

        k_spline = 2
        x_smooth = np.linspace(x.min(), x.max(), 50)
        spl = make_interp_spline(x, y, k=k_spline)
        y_smooth = spl(x_smooth)

        ax.plot(x_smooth, y_smooth,
                label=f"Sim {int(hr)} K/s",
                color=colors[i], linestyle='--',
                marker=markers[i % len(markers)], markevery=10, markersize=5)

        # === Overlay Experimental Data ===
        if exp_df is not None:
            subset = exp_df[(exp_df['fuel_type'] == fuel_type_lower) & (exp_df['heat_rate'] == hr)]
            if not subset.empty:
                x_exp = subset['temperature']
                y_exp = subset['devol_yield'] * (1 - subset['ac'] / 100) * 100
                ax.scatter(x_exp, y_exp, color=colors[i], marker='x', s=50, label=f"Exp {int(hr)} K/s")

    ax.set_xlabel("Temperature / °C")
    ax.set_ylabel("Mass loss / wt.%")
    ax.set_title(f"Simulated vs. Experimental — {fuel_type}")
    ax.set_xlim(temp_min - 10, temp_max + 10)
    ax.set_ylim(0, 100)
    ax.grid(True)
    ax.legend(title="Legend", fontsize=10)

    st.pyplot(fig)
