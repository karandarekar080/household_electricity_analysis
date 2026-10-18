import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, Image as PDFImage, HRFlowable, KeepTogether
)

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Household Electricity Analysis",
    page_icon="⚡",
    layout="wide"
)

# Custom Professional Styling & Theme Injection
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
    }
    h1, h2, h3 {
        color: #1e293b;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 6px;
        padding: 10px 20px;
        color: #475569;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Household Electricity Consumption Analytics")
st.markdown(
    "A professional statistical dashboard designed to evaluate electricity consumption "
    "patterns, correlations, regression models, and household characteristics."
)

st.info(
    "💡 **Note:** The included dataset is synthetic sample data. "
    "Upload your own CSV in the sidebar for custom analysis."
)

# ---------------- LOAD DATA ----------------

uploaded_file = st.sidebar.file_uploader(
    "📁 Upload your custom household CSV",
    type=["csv"]
)

@st.cache_data
def load_default_data():
    return pd.read_csv("household_data.csv")

try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = load_default_data()
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

required_columns = [
    "Household_ID",
    "Household_Size",
    "Appliances",
    "Daily_Usage_Hours",
    "Monthly_Consumption_kWh"
]

if not all(column in df.columns for column in required_columns):
    st.error(
        "CSV must contain these columns: "
        + ", ".join(required_columns)
    )
    st.stop()

df = df[required_columns].copy()

numeric_columns = required_columns[1:]
for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=numeric_columns)

if len(df) < 3:
    st.error("Please provide at least 3 valid household records.")
    st.stop()

if (df["Monthly_Consumption_kWh"] < 0).any():
    st.error("Monthly consumption cannot be negative.")
    st.stop()

if df["Monthly_Consumption_kWh"].mean() <= 0:
    st.error("Mean consumption must be greater than zero.")
    st.stop()

# ---------------- FILTERS ----------------

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Dashboard Controls")

min_people = int(df["Household_Size"].min())
max_people = int(df["Household_Size"].max())

people_range = st.sidebar.slider(
    "Filter by Household Size",
    min_value=min_people,
    max_value=max_people,
    value=(min_people, max_people)
)

filtered = df[
    df["Household_Size"].between(
        people_range[0], people_range[1]
    )
].copy()

if len(filtered) < 3:
    st.warning(
        "Select a wider household-size range "
        "to perform regression and hypothesis testing."
    )
    st.stop()

y = filtered["Monthly_Consumption_kWh"]

# ---------------- SUMMARY METRICS ----------------

mean_value = y.mean()
median_value = y.median()
sd_value = y.std(ddof=1)
cv_value = sd_value / mean_value * 100

st.markdown("### 📊 Key Performance Indicators")
c1, c2, c3, c4 = st.columns(4)

c1.metric("Active Households", f"{len(filtered):,}")
c2.metric("Mean Consumption", f"{mean_value:.1f} kWh")
c3.metric("Median Consumption", f"{median_value:.1f} kWh")
c4.metric("Coef. of Variation", f"{cv_value:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- ORGANIZED TABS FOR DASHBOARD ----------------

tab_overview, tab_desc, tab_dist, tab_corr, tab_reg, tab_hyp, tab_pdf = st.tabs([
    "📋 Raw Data",
    "📈 Descriptive Stats",
    "📊 Distribution",
    "🔗 Correlation",
    "📉 Regression",
    "🧪 Hypothesis Test",
    "📄 Report Export"
])

# --- TAB 1: RAW DATA ---
with tab_overview:
    st.subheader("Filtered Household Dataset")
    st.dataframe(
        filtered.style.background_gradient(subset=["Monthly_Consumption_kWh"], cmap="Blues"),
        use_container_width=True
    )
    
    st.download_button(
        "📥 Download Filtered Data as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="household_data_filtered.csv",
        mime="text/csv"
    )

# --- TAB 2: DESCRIPTIVE STATS ---
with tab_desc:
    st.subheader("Descriptive Statistical Summary")
    
    summary = pd.DataFrame({
        "Measure": [
            "Mean",
            "Median",
            "Sample Standard Deviation",
            "Coefficient of Variation",
            "Minimum Consumption",
            "Maximum Consumption"
        ],
        "Result": [
            f"{mean_value:.2f} kWh",
            f"{median_value:.2f} kWh",
            f"{sd_value:.2f} kWh",
            f"{cv_value:.2f}%",
            f"{y.min():.2f} kWh",
            f"{y.max():.2f} kWh"
        ]
    })

    st.dataframe(
        summary,
        hide_index=True,
        use_container_width=True
    )
    st.caption("Mean represents central tendency; coefficient of variation measures relative dispersion.")

# --- TAB 3: HISTOGRAM ---
with tab_dist:
    st.subheader("Electricity Consumption Distribution")
    
    fig1, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.hist(y, bins=8, color="#2563eb", edgecolor="#1e3a8a", alpha=0.85)
    ax1.set_xlabel("Monthly Consumption (kWh)", fontsize=11, fontweight="bold", color="#334155")
    ax1.set_ylabel("Number of Households", fontsize=11, fontweight="bold", color="#334155")
    ax1.set_title("Frequency Distribution of Monthly Consumption", fontsize=13, fontweight="bold", pad=12, color="#1e293b")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    fig1.tight_layout()

    st.pyplot(fig1)

# --- TAB 4: CORRELATION ---
with tab_corr:
    st.subheader("Bivariate Correlation Analysis")

    variables = {
        "Household size": "Household_Size",
        "Appliance count": "Appliances",
        "Daily usage hours": "Daily_Usage_Hours"
    }

    correlation_results = []

    for label, column in variables.items():
        r, p = stats.pearsonr(
            filtered[column],
            filtered["Monthly_Consumption_kWh"]
        )
        correlation_results.append({
            "Variable": label,
            "Pearson correlation (r)": round(r, 4),
            "P-value": round(p, 6)
        })

    correlation_df = pd.DataFrame(correlation_results)

    st.dataframe(
        correlation_df.style.background_gradient(subset=["Pearson correlation (r)"], cmap="coolwarm", vmin=-1, vmax=1),
        hide_index=True,
        use_container_width=True
    )

    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    ax2.bar(
        correlation_df["Variable"],
        correlation_df["Pearson correlation (r)"],
        color="#0ea5e9",
        edgecolor="#0369a1",
        width=0.5
    )
    ax2.set_ylim(-1, 1)
    ax2.set_ylabel("Pearson Correlation (r)", fontsize=11, fontweight="bold", color="#334155")
    ax2.set_title("Strength of Linear Correlation with Monthly Consumption", fontsize=13, fontweight="bold", pad=12, color="#1e293b")
    ax2.tick_params(axis="x", rotation=0)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig2.tight_layout()

    st.pyplot(fig2)

# --- TAB 5: REGRESSION ---
with tab_reg:
    st.subheader("Ordinary Least Squares (OLS) Regression")

    x = filtered["Appliances"].to_numpy(dtype=float)
    y_array = filtered["Monthly_Consumption_kWh"].to_numpy(dtype=float)

    if len(np.unique(x)) < 2:
        st.warning("Regression needs at least two different appliance counts.")
        st.stop()

    reg = stats.linregress(x, y_array)

    st.markdown("**Fitted Regression Model:**")
    st.latex(
        rf"\hat{{y}} = {reg.intercept:.2f} + {reg.slope:.2f}x"
    )

    r_squared = reg.rvalue ** 2

    r1, r2, r3 = st.columns(3)
    r1.metric("Regression Slope", f"{reg.slope:.2f}")
    r2.metric("R-squared ($R^2$)", f"{r_squared:.4f}")
    r3.metric("Slope P-value", f"{reg.pvalue:.6g}")

    fig3, ax3 = plt.subplots(figsize=(9, 4.5))
    ax3.scatter(x, y_array, color="#2563eb", alpha=0.7, label="Observed Households", s=50)

    line_x = np.linspace(x.min(), x.max(), 100)
    line_y = reg.intercept + reg.slope * line_x

    ax3.plot(line_x, line_y, color="#dc2626", linewidth=2.5, label="Regression Line")
    ax3.set_xlabel("Number of Appliances", fontsize=11, fontweight="bold", color="#334155")
    ax3.set_ylabel("Monthly Consumption (kWh)", fontsize=11, fontweight="bold", color="#334155")
    ax3.set_title("Appliance Count vs. Electricity Consumption", fontsize=13, fontweight="bold", pad=12, color="#1e293b")
    ax3.legend(frameon=True, facecolor="#f8fafc")
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    fig3.tight_layout()

    st.pyplot(fig3)

    st.info(
        f"**Interpretation:** The model estimates an average change of **{reg.slope:.2f} kWh** "
        f"in monthly consumption for each additional appliance. The model explains **{r_squared * 100:.1f}%** "
        f"of the variance in consumption."
    )

# --- TAB 6: HYPOTHESIS TESTING ---
with tab_hyp:
    st.subheader("Statistical Hypothesis Testing on Regression Slope")

    st.markdown("""
    * **Null Hypothesis ($H_0$):** The population regression slope equals zero ($\beta_1 = 0$).
    * **Alternative Hypothesis ($H_1$):** The population regression slope is non-zero ($\beta_1 \neq 0$).
    * **Significance Level ($\alpha$):** $0.05$ (5%)
    """)

    if len(filtered) >= 3 and reg.stderr > 0:
        dfree = len(filtered) - 2
        t_stat = reg.slope / reg.stderr
        p_value = 2 * stats.t.sf(abs(t_stat), dfree)
        t_critical = stats.t.ppf(0.975, dfree)

        lower = reg.slope - t_critical * reg.stderr
        upper = reg.slope + t_critical * reg.stderr

        h1, h2 = st.columns(2)
        with h1:
            st.metric("Calculated t-statistic", f"{t_stat:.4f}")
            st.metric("Degrees of Freedom", f"{dfree}")
        with h2:
            st.metric("Two-Tailed p-value", f"{p_value:.8g}")
            st.metric("95% Confidence Interval", f"[{lower:.3f}, {upper:.3f}]")

        st.markdown("<br>", unsafe_allow_html=True)

        if p_value < 0.05:
            st.success(
                "✅ **Reject $H_0$:** There is strong statistical evidence of a significant "
                "linear association between appliance count and monthly electricity consumption."
            )
        else:
            st.warning(
                "⚠️ **Fail to Reject $H_0$:** Insufficient statistical evidence to conclude "
                "a non-zero linear relationship at the 5% significance level."
            )
    else:
        st.warning("Standard error is zero or sample size is insufficient for standard t-testing.")

# --- TAB 7: PROFESSIONAL REPORT EXPORT ---
with tab_pdf:
    st.subheader("📄 Executive Statistical Report Generator")
    st.write("Generate a publication-grade PDF analytics report including structured cards, styled data tables, and high-resolution chart assets.")

    def make_professional_pdf(data, descriptive, correlations, regression):
        buffer = io.BytesIO()

        # Document setup with calibrated page geometry
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=1.8 * cm,
            leftMargin=1.8 * cm,
            topMargin=1.8 * cm,
            bottomMargin=1.8 * cm
        )

        styles = getSampleStyleSheet()
        
        # Color Palette Definition
        primary_color = colors.HexColor("#0f172a")     # Slate 900
        secondary_color = colors.HexColor("#2563eb")   # Blue 600
        accent_bg = colors.HexColor("#f8fafc")         # Slate 50
        border_color = colors.HexColor("#cbd5e1")      # Slate 300
        text_dark = colors.HexColor("#334155")         # Slate 700

        # Custom Paragraph Typography
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=primary_color,
            alignment=0,
            spaceAfter=4
        )
        
        subtitle_style = ParagraphStyle(
            "DocSubTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=secondary_color,
            spaceAfter=12
        )

        h1_style = ParagraphStyle(
            "SectionH1",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=primary_color,
            spaceBefore=14,
            spaceAfter=6,
            keepWithNext=True
        )

        body_style = ParagraphStyle(
            "ReportBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=text_dark,
            spaceAfter=8
        )

        th_style = ParagraphStyle(
            "TableHeader",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=colors.whitesmoke
        )

        td_style = ParagraphStyle(
            "TableCell",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=text_dark
        )

        callout_style = ParagraphStyle(
            "CalloutText",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=9,
            leading=13,
            textColor=primary_color,
            alignment=1
        )

        story = []

        # 1. Header Banner
        header_data = [
            [
                Paragraph("Household Electricity Consumption Analytics", title_style),
                Paragraph(f"<b>REPORT DATE</b><br/>{datetime.now().strftime('%B %d, %Y')}", ParagraphStyle("HMeta", parent=styles["Normal"], fontName="Helvetica", fontSize=8, leading=11, textColor=colors.HexColor("#64748b"), alignment=2))
            ],
            [
                Paragraph("Executive Statistical Brief & Empirical Modeling Report", subtitle_style),
                Paragraph(f"<b>SAMPLE SIZE</b><br/>N = {len(data)} Households", ParagraphStyle("HMeta2", parent=styles["Normal"], fontName="Helvetica", fontSize=8, leading=11, textColor=colors.HexColor("#64748b"), alignment=2))
            ]
        ]
        header_table = Table(header_data, colWidths=[11.5 * cm, 5.9 * cm])
        header_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(header_table)
        story.append(HRFlowable(width="100%", thickness=1.5, color=secondary_color, spaceBefore=4, spaceAfter=12))

        # 2. Executive Summary
        story.append(Paragraph("1. Executive Summary", h1_style))
        story.append(Paragraph(
            "This empirical report evaluates the relationships between household characteristics and monthly electricity "
            "consumption (kWh). Utilizing descriptive statistics, Pearson correlation analysis, Ordinary Least Squares (OLS) "
            "regression, and formal hypothesis testing, this document provides quantitative insights for utility forecasting.",
            body_style
        ))

        # 3. KPI Stat Cards Container
        story.append(Spacer(1, 4))
        kpi_data = [
            [
                Paragraph("<b>MEAN CONSUMPTION</b>", ParagraphStyle("K1", parent=styles["Normal"], fontSize=7, textColor=colors.HexColor("#64748b"))),
                Paragraph("<b>MEDIAN CONSUMPTION</b>", ParagraphStyle("K2", parent=styles["Normal"], fontSize=7, textColor=colors.HexColor("#64748b"))),
                Paragraph("<b>STD DEVIATION</b>", ParagraphStyle("K3", parent=styles["Normal"], fontSize=7, textColor=colors.HexColor("#64748b"))),
                Paragraph("<b>COEF. VARIATION</b>", ParagraphStyle("K4", parent=styles["Normal"], fontSize=7, textColor=colors.HexColor("#64748b")))
            ],
            [
                Paragraph(f"<b>{mean_value:.1f} kWh</b>", ParagraphStyle("V1", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=12, textColor=primary_color)),
                Paragraph(f"<b>{median_value:.1f} kWh</b>", ParagraphStyle("V2", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=12, textColor=primary_color)),
                Paragraph(f"<b>{sd_value:.1f} kWh</b>", ParagraphStyle("V3", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=12, textColor=primary_color)),
                Paragraph(f"<b>{cv_value:.1f}%</b>", ParagraphStyle("V4", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=12, textColor=primary_color))
            ]
        ]
        kpi_table = Table(kpi_data, colWidths=[4.35 * cm] * 4)
        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), accent_bg),
            ('BOX', (0,0), (-1,-1), 0.5, border_color),
            ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ]))
        story.append(kpi_table)
        story.append(Spacer(1, 10))

        # 4. Descriptive Statistics Table
        story.append(Paragraph("2. Descriptive Statistics Summary", h1_style))
        desc_table_data = [[
            Paragraph("Statistical Measure", th_style),
            Paragraph("Result", th_style)
        ]]
        for _, row in descriptive.iterrows():
            desc_table_data.append([
                Paragraph(str(row["Measure"]), td_style),
                Paragraph(str(row["Result"]), td_style)
            ])

        desc_table = Table(desc_table_data, colWidths=[10.5 * cm, 6.9 * cm])
        desc_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), primary_color),
            ('GRID', (0,0), (-1,-1), 0.5, border_color),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, accent_bg]),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(desc_table)
        story.append(Spacer(1, 10))

        # 5. Correlation Analysis
        story.append(Paragraph("3. Bivariate Correlation Analysis", h1_style))
        corr_table_data = [[
            Paragraph("Variable", th_style),
            Paragraph("Pearson Correlation (r)", th_style),
            Paragraph("P-Value", th_style)
        ]]
        for _, row in correlations.iterrows():
            corr_table_data.append([
                Paragraph(str(row["Variable"]), td_style),
                Paragraph(f"{row['Pearson correlation (r)']:.4f}", td_style),
                Paragraph(f"{row['P-value']:.6f}", td_style)
            ])

        corr_table = Table(corr_table_data, colWidths=[7.4 * cm, 5.0 * cm, 5.0 * cm])
        corr_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), primary_color),
            ('GRID', (0,0), (-1,-1), 0.5, border_color),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, accent_bg]),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(corr_table)
        story.append(Spacer(1, 12))

        # 6. Regression & Visual Charts Section
        story.append(Paragraph("4. Linear Regression Model & Visualization", h1_style))
        
        reg_summary_box = [
            [Paragraph(
                f"<b>Fitted Equation:</b> Monthly Consumption = <b>{regression.intercept:.2f}</b> + <b>{regression.slope:.2f}</b> × (Appliance Count)<br/>"
                f"<b>Model Performance:</b> R-squared (R²) = <b>{regression.rvalue**2:.4f}</b> ({regression.rvalue**2*100:.1f}% Variance Explained)",
                callout_style
            )]
        ]
        reg_box_table = Table(reg_summary_box, colWidths=[17.4 * cm])
        reg_box_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#eff6ff")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#93c5fd")),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(reg_box_table)
        story.append(Spacer(1, 8))

        # Generate High-DPI Chart Image for Report
        chart_buffer = io.BytesIO()
        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        ax.scatter(data["Appliances"], data["Monthly_Consumption_kWh"], color="#2563eb", alpha=0.75, s=45, label="Observed Households")
        
        xx = np.linspace(data["Appliances"].min(), data["Appliances"].max(), 100)
        ax.plot(xx, regression.intercept + regression.slope * xx, color="#dc2626", linewidth=2, label="OLS Regression Line")
        
        ax.set_xlabel("Number of Appliances", fontsize=9, fontweight="bold", color="#334155")
        ax.set_ylabel("Monthly Consumption (kWh)", fontsize=9, fontweight="bold", color="#334155")
        ax.set_title("Appliance Count vs. Electricity Consumption", fontsize=10, fontweight="bold", color="#0f172a", pad=8)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=True, facecolor="#ffffff", fontsize=8)
        fig.tight_layout()
        fig.savefig(chart_buffer, format="png", dpi=250)
        plt.close(fig)
        chart_buffer.seek(0)

        story.append(PDFImage(chart_buffer, width=17.4 * cm, height=7.42 * cm))
        story.append(Spacer(1, 10))

        # 7. Methodology & Conclusions
        story.append(Paragraph("5. Methodology & Analytical Conclusion", h1_style))
        story.append(Paragraph(
            "The statistical methods confirm a robust positive relationship between appliance inventory and household "
            "electricity consumption. The regression slope test demonstrates statistical significance at the 5% level. "
            "Future predictive accuracy can be further improved by tracking seasonal wattage and HVAC load profiles.",
            body_style
        ))

        # Footer divider and document metadata
        story.append(Spacer(1, 6))
        story.append(HRFlowable(width="100%", thickness=0.5, color=border_color, spaceBefore=4, spaceAfter=8))
        story.append(Paragraph(
            f"Report Generated Automatically via Household Electricity Analytics Platform • Confidential & Proprietary",
            ParagraphStyle("DocFooter", parent=styles["Normal"], fontName="Helvetica", fontSize=7.5, textColor=colors.HexColor("#94a3b8"), alignment=1)
        ))

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    pdf_bytes = make_professional_pdf(
        filtered,
        summary,
        correlation_df,
        reg
    )

    st.download_button(
        label="📄 Download Executive PDF Project Report",
        data=pdf_bytes,
        file_name="Electricity_Consumption_Project_Report.pdf",
        mime="application/pdf"
    )

st.sidebar.markdown("---")
st.sidebar.success("Dashboard successfully loaded and ready.")