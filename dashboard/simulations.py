# Simulation Lab Logic
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
import os

def render_what_if_analysis(model_b_module, resources, data_bundle, ticker):
    """
    Interactive 'What-If' scenario builder.
    """
    st.markdown(f"### 🧪 What-If Analysis: {ticker}")
    st.caption("adjust the sliders to see how changes in fundamentals impact the probability distribution of returns.")

    # 1. Get latest feature row
    stmts = data_bundle.get(ticker, {}).get("stmts")
    if not stmts:
        st.error("No data found for ticker.")
        return

    # Generate features using the passed module
    df_features = model_b_module._feature_frame(ticker, stmts)
    if df_features.empty:
        st.error("Could not generate features.")
        return
        
    # Take the latest row
    latest_row = df_features.sort_values("asof_date").iloc[-1].copy()
    
    # 2. Sliders for Key Drivers
    # We identify a few interpretable features to tweak.
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Profitability**")
        # Net Margin (Log scale or absolute? The model uses absolute ratio)
        curr_nm = float(latest_row.get("netMargin", 0.10))
        new_nm = st.slider("Net Margin", min_value=-0.50, max_value=0.50, value=curr_nm, step=0.01, format="%.2f")
        
        # Revenue Growth (YoY)
        curr_rev_yoy = float(latest_row.get("revYoY", 0.0))
        new_rev_yoy = st.slider("Revenue Growth (YoY)", min_value=-1.0, max_value=2.0, value=curr_rev_yoy, step=0.05, format="%.2f")

    with col2:
        st.markdown("**Leverage & Efficiency**")
        # Debt to Assets
        curr_d2a = float(latest_row.get("debtToAssets", 0.0))
        new_d2a = st.slider("Debt / Assets", min_value=0.0, max_value=1.5, value=curr_d2a, step=0.05, format="%.2f")
        
        # Asset Turnover
        curr_at = float(latest_row.get("assetTurnover", 1.0))
        new_at = st.slider("Asset Turnover", min_value=0.0, max_value=3.0, value=curr_at, step=0.1, format="%.2f")

    with col3:
        st.markdown("**Market Sentiment**")
        # Momentum (12m)
        curr_mom = float(latest_row.get("mom_12m", 0.0))
        new_mom = st.slider("Momentum (12m)", min_value=-0.8, max_value=3.0, value=curr_mom, step=0.1, format="%.2f")
        
        # Volatility (6m)
        curr_vol = float(latest_row.get("vol_6m", 0.02))
        new_vol = st.slider("Volatility (6m)", min_value=0.01, max_value=0.10, value=curr_vol, step=0.005, format="%.3f")

    # 3. Apply overrides
    scenario_row = latest_row.copy()
    scenario_row["netMargin"] = new_nm
    scenario_row["revYoY"] = new_rev_yoy
    scenario_row["debtToAssets"] = new_d2a
    scenario_row["assetTurnover"] = new_at
    scenario_row["mom_12m"] = new_mom
    scenario_row["vol_6m"] = new_vol
    
    # Recalculate derived features if necessary (simple dependency update)
    # e.g. netMargin changes -> netMargin_roll8_mean? 
    # For this V1 simulation, we just update the specific point-in-time features.
    # The model is robust enough to handle single inconsistencies, or we assume this is a "regime shift".
    
    # 4. Predict
    # We construct a DataFrame with 2 rows: [Original, Scenario]
    df_input = pd.DataFrame([latest_row, scenario_row])
    # Ensure all required features are present (model_b.predict_on_features handles cleaning)
    # We might need to fill NaNs for rolling cols if they are missing in 'latest_row' (shouldn't be if it came from _feature_frame)
    
    preds = model_b_module.predict_on_features(resources, df_input)
    
    # 5. Visualize (Comparison Cone)
    fig = go.Figure()
    
    # Helper to plot a single cone
    def plot_cone(row_idx, name, color):
        p = preds.iloc[row_idx]
        x_vals = ["Bear (q5)", "Low (q25)", "Median (q50)", "High (q75)", "Bull (q95)"]
        y_vals = [p["q5"], p["q25"], p["q50"], p["q75"], p["q95"]]
        
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='lines+markers',
            name=name,
            line=dict(color=color, width=3, shape='spline'),
            marker=dict(size=8)
        ))
        
        # Fill area for IQR
        fig.add_trace(go.Scatter(
            x=["Low (q25)", "Median (q50)", "High (q75)"],
            y=[p["q25"], p["q50"], p["q75"]],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor=color.replace('rgb', 'rgba').replace(')', ', 0.2)'),
            showlegend=False
        ))

    # Original
    plot_cone(0, "Original Forecast", "rgb(150, 150, 150)")
    
    # Scenario
    plot_cone(1, "Simulated Scenario", "rgb(0, 123, 255)")
    
    fig.update_layout(
        title="Forecast Distribution Shift",
        yaxis_title="Predicted 12m Return",
        yaxis=dict(tickformat=".1%"),
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", y=1.05)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Delta metrics
    orig_med = preds.iloc[0]["q50"]
    new_med = preds.iloc[1]["q50"]
    delta = new_med - orig_med
    
    st.metric("Median Return Impact", f"{delta:+.2%}", delta_color="normal")

    # 6. Data Trace (Equation Visualization)
    render_feature_trace(latest_row, scenario_row)

def render_feature_trace(original, scenario):
    """
    Visualizes the transformation of raw inputs to features.
    """
    with st.expander("🔍 Feature Engineering Trace (Under the Hood)"):
        st.markdown("How raw numbers become model inputs:")
        
        # Example 1: Net Margin
        st.markdown("### 1. Net Margin")
        st.latex(r"\text{NetMargin} = \frac{\text{NetIncome}}{\text{TotalRevenue}}")
        
        # We don't have the raw values for the SCENARIO (since we adjusted the ratio directly),
        # but we can show the Original derivation if we had the raw components in 'original'.
        # Since 'original' is the FEATURE row, it doesn't have 'NetIncome' raw unless we kept it.
        # model_b._feature_frame DOES keep raw cols! (lines 309 in model_b.py)
        
        ni = original.get("netIncome", np.nan)
        rev = original.get("totalRevenue", np.nan)
        mar = original.get("netMargin", np.nan)
        
        if pd.notna(ni) and pd.notna(rev):
            st.code(f"Original: {ni:,.0f} / {rev:,.0f} = {mar:.4f} ({mar:.2%})", language="python")
        
        st.markdown(f"**Scenario Input:** `{scenario['netMargin']:.4f}`")

        # Example 2: Log Assets (Compression)
        st.markdown("### 2. Signed Log Assets (Tail Compression)")
        st.latex(r"f(x) = \text{sign}(x) \cdot \ln(1 + |x|)")
        
        assets = original.get("totalAssets", np.nan)
        log_a = original.get("logAssets", np.nan)
        if pd.notna(assets):
            st.code(f"Original: ln(1 + {assets:,.0f}) ≈ {log_a:.4f}", language="python")
            
        # Example 3: Seasonality (YoY)
        st.markdown("### 3. Year-Over-Year Delta")
        st.latex(r"\Delta_{YoY} = \frac{X_t - X_{t-4}}{X_{t-4}}")
        st.caption("(Used for Revenue, Net Income, Margins to remove seasonality)")


