"""
Zero-Fail Logistics - Interactive Demo Dashboard
Junction 2025 Challenge - Valio Aimo

Streamlit dashboard to demonstrate the stockout prediction system.
Run with: streamlit run 05_demo_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Valio Aimo - Zero-Fail Logistics",
    page_icon="📦",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .high-risk {
        background-color: #ff4b4b;
        color: white;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    .medium-risk {
        background-color: #ffa500;
        color: white;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    .low-risk {
        background-color: #00cc00;
        color: white;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">📦 Valio Aimo - Zero-Fail Logistics</div>', unsafe_allow_html=True)
st.markdown("### Pre-Picking Stockout Prediction System")
st.markdown("---")

# Load model and data
@st.cache_resource
def load_model():
    DATA_DIR = "/Users/otsov/Programming/Random experiments/Junction2025"
    try:
        model = joblib.load(f"{DATA_DIR}/stockout_prediction_model.pkl")
        return model
    except:
        return None

@st.cache_data
def load_data():
    DATA_DIR = "/Users/otsov/Programming/Random experiments/Junction2025"
    try:
        # Load feature importance
        feature_importance = pd.read_csv(f"{DATA_DIR}/feature_importance.csv")
        
        # Load test predictions
        test_predictions = pd.read_csv(f"{DATA_DIR}/test_predictions.csv")
        
        # Load product stats
        product_stats = pd.read_csv(f"{DATA_DIR}/product_stockout_stats.csv")
        
        return feature_importance, test_predictions, product_stats
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

model = load_model()
feature_importance, test_predictions, product_stats = load_data()

# Sidebar - Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Overview",
    "🔮 Live Prediction",
    "📊 Model Performance",
    "🎯 Feature Importance",
    "💼 Business Impact"
])

# ==============================================================================
# PAGE 1: Overview
# ==============================================================================
if page == "🏠 Overview":
    st.header("Challenge Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("The Problem")
        st.write("""
        **Current State:**
        - Customers discover missing items at delivery or next-day call
        - Manual process is time-consuming and reactive
        - Poor customer experience and operational inefficiency
        
        **Impact:**
        - 20,000+ professional kitchens affected
        - Manual calls take significant operational resources
        - Late discovery means no time for corrective action
        """)
    
    with col2:
        st.subheader("Our Solution")
        st.write("""
        **AI-Powered Pre-Picking Prediction:**
        - 🎯 Predict stockout probability BEFORE picking
        - 📞 Proactive customer contact with replacements
        - 🤖 Automated, reducing manual workload
        - 📈 Improve customer satisfaction and efficiency
        
        **Technology Stack:**
        - LightGBM for stockout prediction
        - Real-time feature engineering
        - REST API for integration
        """)
    
    st.markdown("---")
    
    # Key metrics (if data available)
    if test_predictions is not None:
        st.subheader("Model Performance Snapshot")
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate metrics
        high_risk = (test_predictions['predicted_proba'] > 0.7).sum()
        actual_stockouts = test_predictions['actual'].sum()
        caught_stockouts = ((test_predictions['predicted_proba'] > 0.7) & 
                          (test_predictions['actual'] == 1)).sum()
        precision = caught_stockouts / high_risk if high_risk > 0 else 0
        
        with col1:
            st.metric("Test Orders", f"{len(test_predictions):,}")
        with col2:
            st.metric("Actual Stockouts", f"{actual_stockouts:,}")
        with col3:
            st.metric("High-Risk Predictions", f"{high_risk:,}")
        with col4:
            st.metric("Precision @ 70%", f"{precision:.1%}")

# ==============================================================================
# PAGE 2: Live Prediction
# ==============================================================================
elif page == "🔮 Live Prediction":
    st.header("Real-Time Stockout Prediction")
    st.write("Enter order details to get instant stockout probability prediction")
    
    if model is None:
        st.error("⚠️ Model not loaded. Please run the training script first.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Order Information")
            
            product_code = st.text_input("Product Code", "410397")
            order_qty = st.number_input("Order Quantity", min_value=1.0, value=5.0, step=1.0)
            customer_number = st.text_input("Customer Number", "33345")
            sales_unit = st.selectbox("Sales Unit", ["ST", "BOT", "PAK", "BAG", "RAS", "PRK"])
            
            order_date = st.date_input("Order Date", datetime.now())
            order_hour = st.slider("Order Hour", 0, 23, 8)
            lead_time = st.slider("Lead Time (days)", 0, 7, 1)
            
        with col2:
            st.subheader("Additional Context")
            
            plant = st.text_input("Plant", "30588")
            supplier_reliability = st.slider("Supplier Reliability", 0.0, 1.0, 0.95, 0.01)
            product_hist_stockout = st.slider("Historical Stockout Rate", 0.0, 1.0, 0.05, 0.01)
            
        if st.button("🔮 Predict Stockout Probability", type="primary"):
            # Create feature dictionary
            order_datetime = datetime.combine(order_date, datetime.min.time()).replace(hour=order_hour)
            
            features = pd.DataFrame([{
                'order_month': order_datetime.month,
                'order_day_of_week': order_datetime.weekday(),
                'order_hour': order_hour,
                'order_week': order_datetime.isocalendar()[1],
                'lead_time_days': lead_time,
                'is_weekend': int(order_datetime.weekday() in [5, 6]),
                'is_monday': int(order_datetime.weekday() == 0),
                'is_friday': int(order_datetime.weekday() == 4),
                'quarter': (order_datetime.month - 1) // 3 + 1,
                'is_holiday_season': int(order_datetime.month in [11, 12, 1]),
                'is_rush_hour': int(order_hour in [7, 8, 9, 10, 11]),
                'product_historical_stockout_rate': product_hist_stockout,
                'product_order_frequency': 100,
                'product_avg_order_qty': 10,
                'order_qty': order_qty,
                'is_large_order': int(order_qty > 10),
                'customer_order_frequency': 50,
                'customer_avg_lead_time': 1.5,
                'plant_stockout_rate': 0.05,
                'storage_stockout_rate': 0.05,
                'warehouse_stockout_rate': 0.05,
                'supplier_reliability': supplier_reliability,
                'sales_unit': sales_unit,
            }])
            
            # Make prediction
            try:
                prob = model.predict(features)[0]
                
                st.markdown("---")
                st.subheader("Prediction Results")
                
                # Display probability with color coding
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Stockout Probability", f"{prob:.1%}")
                
                with col2:
                    if prob > 0.7:
                        st.markdown('<div class="high-risk">⚠️ HIGH RISK</div>', unsafe_allow_html=True)
                        risk_level = "HIGH"
                    elif prob > 0.3:
                        st.markdown('<div class="medium-risk">⚡ MEDIUM RISK</div>', unsafe_allow_html=True)
                        risk_level = "MEDIUM"
                    else:
                        st.markdown('<div class="low-risk">✅ LOW RISK</div>', unsafe_allow_html=True)
                        risk_level = "LOW"
                
                with col3:
                    st.metric("Confidence", f"{max(prob, 1-prob):.1%}")
                
                # Recommendations
                st.subheader("Recommended Actions")
                if prob > 0.7:
                    st.error("""
                    🚨 **HIGH RISK - Immediate Action Required:**
                    1. Contact customer proactively via AI agent
                    2. Propose alternative products
                    3. Confirm replacement or delay acceptance
                    4. Flag for inventory team review
                    """)
                elif prob > 0.3:
                    st.warning("""
                    ⚠️ **MEDIUM RISK - Monitor Closely:**
                    1. Check current inventory levels
                    2. Prepare backup options
                    3. Alert picking team
                    4. Consider priority handling
                    """)
                else:
                    st.success("""
                    ✅ **LOW RISK - Proceed Normally:**
                    1. Process order as usual
                    2. Standard picking workflow
                    3. No special action needed
                    """)
                
                # Visual gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Stockout Risk"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "lightyellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"Prediction error: {e}")

# ==============================================================================
# PAGE 3: Model Performance
# ==============================================================================
elif page == "📊 Model Performance":
    st.header("Model Performance Analysis")
    
    if test_predictions is not None:
        # ROC Curve
        from sklearn.metrics import roc_curve, auc, precision_recall_curve
        
        fpr, tpr, _ = roc_curve(test_predictions['actual'], test_predictions['predicted_proba'])
        roc_auc = auc(fpr, tpr)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ROC Curve")
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={roc_auc:.3f})'))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
            fig_roc.update_layout(
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                title='ROC Curve'
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        
        with col2:
            st.subheader("Precision-Recall Curve")
            precision, recall, _ = precision_recall_curve(test_predictions['actual'], test_predictions['predicted_proba'])
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines'))
            fig_pr.update_layout(
                xaxis_title='Recall',
                yaxis_title='Precision',
                title='Precision-Recall Curve'
            )
            st.plotly_chart(fig_pr, use_container_width=True)
        
        # Prediction distribution
        st.subheader("Prediction Distribution")
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=test_predictions[test_predictions['actual']==0]['predicted_proba'],
            name='Fulfilled',
            opacity=0.7,
            nbinsx=50
        ))
        fig_dist.add_trace(go.Histogram(
            x=test_predictions[test_predictions['actual']==1]['predicted_proba'],
            name='Stockout',
            opacity=0.7,
            nbinsx=50
        ))
        fig_dist.update_layout(
            xaxis_title='Predicted Probability',
            yaxis_title='Count',
            title='Distribution of Predictions by Actual Class',
            barmode='overlay'
        )
        st.plotly_chart(fig_dist, use_container_width=True)

# ==============================================================================
# PAGE 4: Feature Importance
# ==============================================================================
elif page == "🎯 Feature Importance":
    st.header("Feature Importance Analysis")
    
    if feature_importance is not None:
        st.write("Understanding which factors drive stockout predictions")
        
        # Top features bar chart
        top_n = st.slider("Number of top features to display", 5, 30, 15)
        top_features = feature_importance.head(top_n)
        
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title=f'Top {top_n} Most Important Features'
        )
        fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature interpretation
        st.subheader("Feature Categories")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Time-Based Features:**")
            time_features = feature_importance[feature_importance['feature'].str.contains('hour|day|week|month|weekend|season|quarter', case=False)]
            st.dataframe(time_features.head(10), use_container_width=True)
        
        with col2:
            st.write("**Product & Supplier Features:**")
            product_features = feature_importance[feature_importance['feature'].str.contains('product|supplier|plant|warehouse|storage', case=False)]
            st.dataframe(product_features.head(10), use_container_width=True)

# ==============================================================================
# PAGE 5: Business Impact
# ==============================================================================
elif page == "💼 Business Impact":
    st.header("Business Impact & ROI")
    
    if test_predictions is not None:
        st.subheader("Operational Metrics")
        
        # Calculate various metrics at different thresholds
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        metrics_data = []
        
        for threshold in thresholds:
            predictions = (test_predictions['predicted_proba'] > threshold).astype(int)
            tp = ((predictions == 1) & (test_predictions['actual'] == 1)).sum()
            fp = ((predictions == 1) & (test_predictions['actual'] == 0)).sum()
            tn = ((predictions == 0) & (test_predictions['actual'] == 0)).sum()
            fn = ((predictions == 0) & (test_predictions['actual'] == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            metrics_data.append({
                'threshold': threshold,
                'proactive_calls': tp + fp,
                'true_positives': tp,
                'false_positives': fp,
                'precision': precision,
                'recall': recall
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.line(metrics_df, x='threshold', y=['precision', 'recall'], 
                          title='Precision vs Recall by Threshold',
                          labels={'value': 'Score', 'variable': 'Metric'})
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.bar(metrics_df, x='threshold', y='proactive_calls',
                         title='Number of Proactive Calls by Threshold')
            st.plotly_chart(fig2, use_container_width=True)
        
        # ROI Calculation
        st.subheader("ROI Estimation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Cost Assumptions (per order):**")
            manual_call_cost = st.number_input("Manual call cost (€)", value=5.0, step=0.5)
            ai_call_cost = st.number_input("AI call cost (€)", value=0.5, step=0.1)
            
        with col2:
            st.write("**Customer Impact (per order):**")
            stockout_penalty = st.number_input("Stockout penalty (€)", value=50.0, step=5.0)
            prevented_benefit = st.number_input("Prevention benefit (€)", value=40.0, step=5.0)
        
        with col3:
            st.write("**Scale:**")
            annual_orders = st.number_input("Annual orders", value=7_000_000, step=100_000)
            stockout_rate = st.number_input("Stockout rate (%)", value=5.0, step=0.5) / 100
        
        # Calculate ROI
        selected_threshold = 0.7
        selected_metrics = metrics_df[metrics_df['threshold'] == selected_threshold].iloc[0]
        
        annual_stockouts = annual_orders * stockout_rate
        prevented_stockouts = annual_stockouts * selected_metrics['recall']
        ai_calls_made = selected_metrics['proactive_calls'] / len(test_predictions) * annual_orders
        
        # Costs
        current_cost = annual_stockouts * manual_call_cost
        ai_cost = ai_calls_made * ai_call_cost
        
        # Benefits
        prevented_penalties = prevented_stockouts * prevented_benefit
        
        # Net benefit
        net_benefit = prevented_penalties - ai_cost
        roi = (net_benefit / ai_cost) * 100 if ai_cost > 0 else 0
        
        st.markdown("---")
        st.subheader(f"📊 Annual Impact (@ {selected_threshold:.0%} threshold)")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Prevented Stockouts", f"{prevented_stockouts:,.0f}")
        with col2:
            st.metric("AI Calls Made", f"{ai_calls_made:,.0f}")
        with col3:
            st.metric("Net Annual Benefit", f"€{net_benefit:,.0f}")
        with col4:
            st.metric("ROI", f"{roi:.0f}%")
        
        st.success(f"""
        💰 **Financial Summary:**
        - Current annual cost of reactive calls: €{current_cost:,.0f}
        - New AI system cost: €{ai_cost:,.0f}
        - Customer satisfaction benefit: €{prevented_penalties:,.0f}
        - **Net annual benefit: €{net_benefit:,.0f}**
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏆 Junction 2025 - Zero-Fail Logistics Challenge | Valio Aimo</p>
    <p>Built with ❤️ using LightGBM, Streamlit, and Plotly</p>
</div>
""", unsafe_allow_html=True)

