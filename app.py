# =========================================================
# AMMC-AGENT
# Interface Streamlit Professionnelle
# Master IA & Recherche Opérationnelle
# =========================================================

# =========================================================
# INSTALLATION
# =========================================================
# pip install streamlit pandas matplotlib seaborn
# pip install scikit-learn scipy openpyxl
# pip install langchain langchain-groq
# pip install fpdf
# pip install streamlit-authenticator

# =========================================================
# EXECUTION
# =========================================================
# streamlit run app.py

# =========================================================
# IMPORTS
# =========================================================

import sqlite3
import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.optimize import minimize

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from fpdf import FPDF

import streamlit_authenticator as stauth

# =========================================================
# CONFIGURATION PAGE
# =========================================================

st.set_page_config(

    page_title="AMMC-Agent",

    page_icon="📊",

    layout="wide",

    initial_sidebar_state="expanded"
)

# =========================================================
# CSS
# =========================================================

st.markdown("""

<style>

.main {
    background-color: #f8fafc;
}

h1, h2, h3 {
    color: #0f172a;
}

.stMetric {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
}

.block-container {
    padding-top: 2rem;
}

</style>

""", unsafe_allow_html=True)

# =========================================================
# TITRE
# =========================================================

st.title("📊 AMMC-Agent")

st.markdown("""
### Système IA Multi-Agents Collaboratif  
Analyse Automatique des États Financiers Marocains
""")

# =========================================================
# AUTHENTIFICATION
# =========================================================

credentials = {

    "usernames": {

        "admin": {

            "name": "Admin",

            "password": stauth.Hasher.hash("1234")
        }
    }
}

authenticator = stauth.Authenticate(

    credentials,

    "ammc_cookie",

    "ammc_agent_super_secure_key_2026",

    cookie_expiry_days=1
)

# =========================================================
# LOGIN
# =========================================================

authenticator.login(
    location="main"
)

# =========================================================
# SESSION
# =========================================================

authentication_status = st.session_state.get(
    "authentication_status"
)

name = st.session_state.get(
    "name"
)

username = st.session_state.get(
    "username"
)

# =========================================================
# LOGIN CHECK
# =========================================================

if authentication_status == False:

    st.error(
        "❌ Username ou Password incorrect"
    )

elif authentication_status == None:

    st.warning(
        "⚠️ Entrez vos informations"
    )

elif authentication_status:

    authenticator.logout(
        location="sidebar"
    )

    st.sidebar.success(
        f"Bienvenue {name}"
    )

    # =====================================================
    # CLE API GROQ
    # =====================================================

    MY_GROQ_KEY = "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"

    # =====================================================
    # INITIALISATION LLM
    # =====================================================

    llm = ChatGroq(

        model_name="llama-3.3-70b-versatile",

        groq_api_key=MY_GROQ_KEY,

        temperature=0
    )

    output_parser = StrOutputParser()

    # =====================================================
    # SQLITE
    # =====================================================

    @st.cache_data
    def load_data():

        conn = sqlite3.connect(
            "finance_maroc_pro.db"
        )

        df = pd.read_sql(
            "SELECT * FROM financial_data",
            conn
        )

        conn.close()

        return df

    df = load_data()

    # =====================================================
    # CLEAN DATA
    # =====================================================

    df = df.dropna(
        subset=["valeur_normalisee"]
    )

    df = df[
        df["valeur_normalisee"] > 0
    ]

    # =====================================================
    # CLEAN PDF TEXT
    # =====================================================

    def clean_text(text):

        return str(text).encode(
            'latin-1',
            'replace'
        ).decode('latin-1')

    # =====================================================
    # SIDEBAR
    # =====================================================

    st.sidebar.title("📌 Navigation")

    page = st.sidebar.radio(

        "Choisir une page",

        [
            "🏠 Dashboard",

            "🏢 Analyse Entreprise",

            "📈 Visualisations",

            "📊 Optimisation",

            "🤖 Chat IA",

            "📥 Export"
        ]
    )

    # =====================================================
    # DASHBOARD
    # =====================================================

    if page == "🏠 Dashboard":

        st.header("📊 Dashboard Général")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Documents",
            len(df)
        )

        col2.metric(
            "Entreprises",
            df["emetteur"].nunique()
        )

        col3.metric(
            "Métriques",
            df["metrique"].nunique()
        )

        col4.metric(
            "Années",
            df["annee"].nunique()
        )

        st.subheader("📄 Aperçu des données")

        st.dataframe(
            df.head(20),
            width='stretch'
        )

    # =====================================================
    # ANALYSE ENTREPRISE
    # =====================================================

    elif page == "🏢 Analyse Entreprise":

        st.header("🏢 Analyse Entreprise")

        entreprises = sorted(
            df["emetteur"].unique()
        )

        entreprise = st.selectbox(

            "Sélectionner une entreprise",

            entreprises
        )

        df_ent = df[
            df["emetteur"] == entreprise
        ]

        st.dataframe(
            df_ent,
            width='stretch'
        )

        pivot = df_ent.pivot_table(

            index="annee",

            columns="metrique",

            values="valeur_normalisee",

            aggfunc="mean"
        )

        st.subheader("📈 Évolution Financière")

        fig, ax = plt.subplots(
            figsize=(12,6)
        )

        pivot.plot(
            ax=ax,
            marker="o"
        )

        plt.grid(True)

        st.pyplot(fig)

    # =====================================================
    # VISUALISATIONS
    # =====================================================

    elif page == "📈 Visualisations":

        st.header("📈 Visualisations IA")

        pivot_df = df.pivot_table(

            index='emetteur',

            columns='metrique',

            values='valeur_normalisee',

            aggfunc='mean'
        )

        pivot_df = pivot_df.fillna(0)

        X = pivot_df.values

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(

            n_clusters=3,

            random_state=42
        )

        clusters = kmeans.fit_predict(
            X_scaled
        )

        pca = PCA(
            n_components=2
        )

        reduced = pca.fit_transform(
            X_scaled
        )

        plot_df = pd.DataFrame({

            "Entreprise": pivot_df.index,

            "PCA1": reduced[:,0],

            "PCA2": reduced[:,1],

            "Cluster": clusters
        })

        st.subheader(
            " Clustering Entreprises"
        )

        fig, ax = plt.subplots(
            figsize=(12,8)
        )

        scatter = ax.scatter(

            plot_df["PCA1"],

            plot_df["PCA2"],

            c=plot_df["Cluster"],

            s=150,

            cmap='viridis'
        )

        for i in range(len(plot_df)):

            ax.text(

                plot_df["PCA1"].iloc[i],

                plot_df["PCA2"].iloc[i],

                plot_df["Entreprise"].iloc[i],

                fontsize=8
            )

        ax.grid(True)

        st.pyplot(fig)

        # =================================================
        # HEATMAP
        # =================================================

        st.subheader("Heatmap")

        corr = pivot_df.corr()

        fig2, ax2 = plt.subplots(
            figsize=(10,8)
        )

        sns.heatmap(

            corr,

            annot=True,

            cmap="Blues",

            ax=ax2
        )

        st.pyplot(fig2)

    # =====================================================
    # OPTIMISATION
    # =====================================================

    elif page == "📊 Optimisation":

        st.header(
            "📊 Optimisation Portefeuille"
        )

        pivot_df = df.pivot_table(

            index=['emetteur', 'annee'],

            columns='metrique',

            values='valeur_normalisee',

            aggfunc='mean'
        ).reset_index()

        pivot_df = pivot_df.fillna(0)

        pivot_df["ROE"] = (

            pivot_df.get(
                "resultat_net",
                0
            )

            /

            pivot_df.get(
                "capitaux_propres",
                1
            )
        )

        portfolio_df = pivot_df[
            [
                "emetteur",
                "ROE"
            ]
        ]

        portfolio_df = portfolio_df.groupby(
            "emetteur"
        ).mean().reset_index()

        portfolio_df = portfolio_df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        portfolio_df = portfolio_df.dropna()

        returns = portfolio_df["ROE"].values

        companies = portfolio_df[
            "emetteur"
        ].values

        n_assets = len(companies)

        # =================================================
        # MATRICE COVARIANCE
        # =================================================

        returns_matrix = np.random.randn(
            500,
            n_assets
        )

        cov_matrix = np.cov(
            returns_matrix.T
        )

        # =================================================
        # FONCTIONS
        # =================================================

        def portfolio_return(weights):

            return np.sum(
                returns * weights
            )

        def portfolio_risk(weights):

            return np.sqrt(

                np.dot(
                    weights.T,

                    np.dot(
                        cov_matrix,
                        weights
                    )
                )
            )

        def objective(weights):

            return -(
                portfolio_return(weights)

                -

                0.5 * portfolio_risk(weights)
            )

        # =================================================
        # CONTRAINTES
        # =================================================

        constraints = [

            {
                'type': 'eq',
                'fun': lambda w: np.sum(w) - 1
            }
        ]

        bounds = tuple(

            (0, 0.5)

            for i in range(n_assets)
        )

        # =================================================
        # INITIALISATION
        # =================================================

        init_weights = np.array(

            n_assets * [1/n_assets]
        )

        # =================================================
        # OPTIMISATION
        # =================================================

        optimal = minimize(

            objective,

            init_weights,

            method='SLSQP',

            bounds=bounds,

            constraints=constraints
        )

        weights = optimal.x

        # =================================================
        # RESULTATS
        # =================================================

        result_df = pd.DataFrame({

            "Entreprise": companies,

            "Poids Optimal": weights
        })

        st.subheader(
            "Portefeuille Optimal"
        )

        st.dataframe(
            result_df,
            width='stretch'
        )

        # =================================================
        # FRONTIERE EFFICIENTE
        # =================================================

        st.subheader(
            "📈 Frontière Efficiente"
        )

        portfolio_returns = []

        portfolio_risks = []

        for i in range(5000):

            random_weights = np.random.random(
                n_assets
            )

            random_weights /= np.sum(
                random_weights
            )

            ret = np.sum(
                returns * random_weights
            )

            risk = np.sqrt(

                np.dot(
                    random_weights.T,

                    np.dot(
                        cov_matrix,
                        random_weights
                    )
                )
            )

            portfolio_returns.append(ret)

            portfolio_risks.append(risk)

        optimal_return = np.sum(
            returns * weights
        )

        optimal_risk = np.sqrt(

            np.dot(
                weights.T,

                np.dot(
                    cov_matrix,
                    weights
                )
            )
        )

        fig, ax = plt.subplots(
            figsize=(12,7)
        )

        scatter = ax.scatter(

            portfolio_risks,

            portfolio_returns,

            c=portfolio_returns,

            cmap='viridis',

            alpha=0.5
        )

        ax.scatter(

            optimal_risk,

            optimal_return,

            color='red',

            s=250,

            label='Portefeuille Optimal'
        )

        ax.set_title(
            "Frontière Efficiente - Markowitz"
        )

        ax.set_xlabel(
            "Risque"
        )

        ax.set_ylabel(
            "Rendement"
        )

        ax.grid(True)

        ax.legend()

        fig.colorbar(
            scatter,
            ax=ax,
            label="Rendement"
        )

        st.pyplot(fig)

    # =====================================================
    # CHAT IA
    # =====================================================

    elif page == " Chat IA":

        st.header(" Chat IA Collaboratif")

        question = st.text_area(
            "Posez votre question financière"
        )

        if st.button("Analyser"):

            if question:

                data_context = df.head(
                    50
                ).to_string()

                prompt = ChatPromptTemplate.from_template("""

Tu es un analyste financier expert marocain.

Base financière :

{data}

Question utilisateur :

{question}

""")

                chain = (

                    prompt

                    | llm

                    | output_parser
                )

                try:

                    response = chain.invoke({

                        "data": data_context,

                        "question": question
                    })

                    st.success(
                        " Réponse IA"
                    )

                    st.write(response)

                except Exception as e:

                    st.error(
                        f"Erreur LLM : {e}"
                    )

    # =====================================================
    # EXPORT
    # =====================================================

    elif page == "📥 Export":

        st.header("📥 Export Données")

        st.dataframe(
            df.head(20),
            width='stretch'
        )

        # CSV

        csv = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(

            "⬇️ Télécharger CSV",

            csv,

            "finance_export.csv",

            "text/csv"
        )

        # EXCEL

        excel_file = "finance_export.xlsx"

        with pd.ExcelWriter(
            excel_file,
            engine="openpyxl"
        ) as writer:

            df.to_excel(

                writer,

                index=False,

                sheet_name="Financial_Data"
            )

        with open(
            excel_file,
            "rb"
        ) as f:

            st.download_button(

                "⬇️ Télécharger Excel",

                f,

                file_name="finance_export.xlsx"
            )

        # PDF

        st.subheader("📄 Export PDF")

        if st.button("📄 Générer Rapport PDF"):

            try:

                pdf = FPDF()

                pdf.add_page()

                pdf.set_font(
                    "Arial",
                    "B",
                    16
                )

                pdf.cell(

                    200,

                    10,

                    txt=clean_text(
                        "AMMC-Agent Financial Report"
                    ),

                    ln=True,

                    align='C'
                )

                pdf.ln(10)

                pdf.set_font(
                    "Arial",
                    size=11
                )

                pdf.multi_cell(

                    0,

                    8,

                    txt=clean_text("""
Système IA Multi-Agents Collaboratif
Analyse Automatique des États Financiers Marocains
""")
                )

                pdf_file = "finance_report.pdf"

                pdf.output(pdf_file)

                with open(
                    pdf_file,
                    "rb"
                ) as f:

                    st.download_button(

                        "⬇️ Télécharger PDF",

                        f,

                        file_name="finance_report.pdf",

                        mime="application/pdf"
                    )

                st.success(
                    "✅ PDF généré avec succès"
                )

            except Exception as e:

                st.error(
                    f"Erreur PDF : {e}"
                )

    # =====================================================
    # FOOTER
    # =====================================================

    st.markdown("---")

    st.caption("""
AMMC-Agent © Master Intelligence Artificielle & Recherche Opérationnelle
""")