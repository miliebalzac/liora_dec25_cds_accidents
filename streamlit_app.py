import time

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
import streamlit.components.v1 as components
from pathlib import Path
import streamlit.components.v1 as components
    

@st.cache_data
def lire_csv_avec_index(dossier, nom_fichier):
    return pd.read_csv(dossier + nom_fichier, sep=";", encoding="utf-8", index_col=0)

@st.cache_data
def lire_csv_sans_index(dossier, nom_fichier):
        return pd.read_csv(dossier + nom_fichier, sep=";", encoding="utf-8")

@st.cache_data
def lire_texte(dossier, nom_fichier):
    with open(dossier + nom_fichier, encoding="utf-8") as f:
        return f.read()

st.title("Prédiction de la gravité des accidents de la route")
st.sidebar.title("Sommaire")
pages = ["Exploration", "DataVizualization", "FeaturesSelection", "Modélisation", "Carte"]
page = st.sidebar.radio("Aller vers", pages)

en_ligne = True  # True si l'application est en ligne, False si elle est en local

dossier_exploration = "C:/Env_Python/Projet DS accidents/FOR_STREAMLIT/EXPLORATION/"
dossier_visualisation = "C:/Env_Python/Projet DS accidents/FOR_STREAMLIT/DATAVISUALIZATION/"
dossier_featureselection = "C:/Env_Python/Projet DS accidents/FOR_STREAMLIT/FEATURESELECTION/"
dossier_modelisation = "C:/Env_Python/Projet DS accidents/FOR_STREAMLIT/MODELISATION/"
dossier_map = "C:/Env_Python/Projet DS accidents/FOR_STREAMLIT/MODELISATION/"

if en_ligne :
    dossier_exploration = "./FOR_STREAMLIT/EXPLORATION/"
    dossier_visualisation = "./FOR_STREAMLIT/DATAVISUALIZATION/"
    dossier_featureselection = "./FOR_STREAMLIT/FEATURESELECTION/"
    dossier_modelisation = "./FOR_STREAMLIT/MODELISATION/"
    dossier_map = "./FOR_STREAMLIT/MODELISATION/"


# ==========================================================
# Streamlit - Partie EXPLORATION
# ==========================================================

if page == pages[0] :

    st.header("Exploration des tables annuelles")
    st.write("Téléchargées de https://www.data.gouv.fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2024")

    # affiche le Warning de dtype detecté au moment de la lecture des bases années
    st.write("### TABLES CARACTERISTIQUES")
    st.write("décrit les circonstances générales de l'accident : colonnes et types des tables **caracteristiques** par année")
    st.dataframe(lire_csv_avec_index(dossier_exploration,"carac_types.csv"))
    st.write("shapes par année")
    st.dataframe(lire_csv_avec_index(dossier_exploration,"carac_shapes.csv"))
    st.write("### TABLES LIEUX")
    st.write("décrit le lieu principal de l'accident  : colonnes et types des tables **lieux** par année ")
    st.write("(Warning de dtype detecté au moment de la lecture des bases années :)")
    st.code(lire_texte(dossier_exploration,"DtypeWarning.txt"))
    st.dataframe(lire_csv_avec_index(dossier_exploration,"lieux_types.csv"))
    st.write("shapes par année")
    st.dataframe(lire_csv_avec_index(dossier_exploration,"lieux_shapes.csv"))
    st.write("### TABLES VEHICULES ")
    st.write("description du ou des véhicules impliqués : colonnes et types des bases **vehicule** par année")
    st.dataframe(lire_csv_avec_index(dossier_exploration,"vehicule_types.csv"))
    st.write("shapes par année")
    st.dataframe(lire_csv_avec_index(dossier_exploration,"vehicule_shapes.csv"))
    st.write("### TABLES USAGERS ")
    st.write("description des usagers impliqués et de la gravité de la blessure corporelle  : colonnes et types des bases **usagers** par année")
    st.dataframe(lire_csv_avec_index(dossier_exploration,"usagers_types.csv"))
    st.write("shapes par année")
    st.dataframe(lire_csv_avec_index(dossier_exploration,"usagers_shapes.csv"))
    st.divider()
    st.write("## Dimension du dataset assemblé")
    st.code(lire_texte(dossier_exploration, "final_shape.txt"))
    st.write("## Echantillon random du dataset assemblé")
    st.dataframe(lire_csv_avec_index(dossier_exploration,"sample.csv").head(10))
    st.write("## Describe du dataset assemblé ")
    st.write("(avant transformation de -1 en NaN)")
    st.dataframe(lire_csv_avec_index(dossier_exploration,"describe.csv"))


# ==========================================================
# Streamlit - Partie DATAVISUALIZATION
# ==========================================================

if page == pages[1] :
    st.header("Visualisation des données")
    st.write("A noter : latitude et longitudes sont les 2 seules variables numériques qui seront conservées. On onglet spécifique carte est porposé pour ces variables")
    
    # SELECTION DE LA VARIABLE QU'ON VEUT ANALYSER------------------------------------------------------------------------------------
    suivi_analyse = lire_csv_avec_index(dossier_visualisation,"suivi_analyse.csv")
    colonnes_disponibles = suivi_analyse["colonne"].tolist()
    colonne_lib = st.selectbox( "Choisissez une variable à analyser", sorted(colonnes_disponibles, key=str.lower))
    colonne = colonne_lib.split(" ")[0]

    # Graphique 1 : répartition des modalités----------------------------
    st.subheader(f"Répartition de la variable « {colonne} »")
    value_counts = lire_csv_sans_index(dossier_visualisation,f"{colonne}_value_counts.csv")
    if colonne in value_counts.columns :
        colonne_label = colonne
    else :
        colonne_label = colonne + "_label"

    # tri décroissant sur les effectifs (équivalent de order=value_counts().index)
    value_counts = value_counts.sort_values("count", ascending=False)

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=value_counts,
        x=colonne_label,
        y="count",
        ax=ax1
    )
    ax1.set_title(f'Répartition de {colonne}')
    ax1.set_xlabel(colonne)
    ax1.set_ylabel("Nombre d'occurrences")
    ax1.tick_params(axis="x", rotation=45, labelsize=8)
    plt.setp(ax1.get_xticklabels(), ha="right")
    fig1.tight_layout()
    st.pyplot(fig1)


    # Graphique 2 : heatmap gravité / modalités------------------------------

    st.subheader(f'Répartition de la gravité par {colonne}')

    contingency_table = lire_csv_sans_index(dossier_visualisation,f"{colonne}_contingency_table.csv")

    # la première colonne contient les libellés : elle devient l'index
    contingency_table = contingency_table.set_index(colonne_label)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        contingency_table,
        annot=True,
        fmt=".1%",
        cmap="YlGnBu",
        linewidths=0.5,
        ax=ax2
    )
    ax2.set_title("Répartition de la gravité en fonction de la variable (en %)")
    ax2.set_xlabel("Gravité")
    ax2.set_ylabel(colonne)
    fig2.tight_layout()
    st.pyplot(fig2)

    # AFFICHAGE DES VARIABLES SUPPRIMEES------------------------------------------------------------------------------------
    st.divider()
    st.subheader("Variables supprimées et leurs motifs")
    df = lire_csv_avec_index(dossier_visualisation,"suivi_suppressions.csv")
    df["taux de na"] = df["taux de na"] / 1000000  
    st.dataframe( df.style.format({"taux de na": "{:.3%}"}),  hide_index=True)


# ==========================================================
# Streamlit - Partie Features Selection
# ==========================================================

if page == pages[2] : 
    st.header("Sélection des features")
    st.write("Tables de V de Cramer / Chi2 par split, puis variables retenues.")

    st.write("\nSplit aléatoire : Test du V de Cramer / Chi2 (resultat trié par V de Cramer décroissant)")
    st.dataframe(lire_csv_avec_index(dossier_featureselection,"resultats_chi2_sal.csv"))
    st.write("\nSplit temporel : Test du V de Cramer / Chi2 (resultat trié par V de Cramer décroissant)")
    st.dataframe(lire_csv_avec_index(dossier_featureselection,"resultats_chi2_temp.csv"))
    st.write("\nSplit groupé par accident : Test du V de Cramer / Chi2 (resultat trié par V de Cramer décroissant)")
    st.dataframe(lire_csv_avec_index(dossier_featureselection,"resultats_chi2_grp.csv"))

    st.write("### Conclusion : aucune différence entre les différents splits concernant les variables qui passent le test de sélection du khi2 et du V de Cramer (p < 0.05) and (V_Cramer >= 0.10)")
    st.write("Toutes les pvalues sont < 0.05 donc on utilise le V de Cramer pour identifier les variables les plus influençantes")
    
    #affichage du graphique de final étape5
    resultats_chi2_temp = lire_csv_sans_index(dossier_featureselection,"resultats_chi2.csv")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(resultats_chi2_temp["Variable"], resultats_chi2_temp["V_Cramer"])
    ax.axvline(0.1, linestyle="--", color = 'red')
    ax.set_xlabel("V de Cramer")
    ax.set_title("Variables retenues par le seuil V de Cramer >= 0,10")
    fig.tight_layout()
    st.pyplot(fig)

# ==========================================================
# Streamlit - Partie MODELISATION
# ==========================================================

if page == pages[3] : 
    st.header("Modélisation")
    # A AJOUTER ???
    # afficher pour chacun des 2 modèles
    #    o	les indicateurs que nous avons retenus
    #    o	la matrice de confusion

    import pickle

    @st.cache_resource
    def charger_modele(fichier):
        with open(dossier_modelisation+fichier, "rb") as f:
            modele = pickle.load(f)
        return modele
    
    st.write("vous souhaitez prédire la gravité des blessures pour un usager impliqué dans un accident de la route ?")
    st.write("Voici un mini modèle uniquement basé sur les 5 facteurs les plus influançants :")
    # secu1 : L'équipement de sécurité principal,
    secu1_mapping = lire_csv_avec_index(dossier_modelisation,"secu1_mapping.csv")
    secu1_possibles = secu1_mapping["description"].tolist()
    secu1_choisi = st.selectbox( "Choisissez secu1 : L'équipement de sécurité principal ", secu1_possibles)
    # catv : la catégorie du véhicule,
    catv_mapping = lire_csv_avec_index(dossier_modelisation,"catv_mapping.csv")
    catv_possibles = catv_mapping["description"].tolist()
    catv_choisi = st.selectbox( "Choisissez catv : La catégorie du véhicule ", catv_possibles)
    # DENS : la densité de population,
    DENS_mapping = lire_csv_avec_index(dossier_modelisation,"DENS_mapping.csv")
    DENS_possibles = DENS_mapping["description"].tolist()
    DENS_choisi = st.selectbox( "Choisissez DENS : La densité de population ", DENS_possibles)
    # manv : la manoeuvre au moment de l'accident,
    manv_mapping = lire_csv_avec_index(dossier_modelisation,"manv_mapping.csv")
    manv_possibles = manv_mapping["description"].tolist()
    manv_choisi = st.selectbox("Choisissez manv : La manoeuvre au moment de l'accident ", manv_possibles)
    # obsm : présence d'un obstacle mobile
    obsm_mapping = lire_csv_avec_index(dossier_modelisation,"obsm_mapping.csv")
    obsm_possibles = obsm_mapping["description"].tolist()
    obsm_choisi = st.selectbox( "Choisissez obsm : La présence d'un obstacle mobile ", obsm_possibles)

    # colonnes attendues par le modèle (ordre strict) 
    colonnes_pour_modele = ['secu1_1.0', 'secu1_2.0', 'secu1_3.0', 'secu1_4.0', 'secu1_5.0',
       'secu1_6.0', 'secu1_7.0', 'secu1_8.0', 'secu1_9.0', 'catv_1.0',
       'catv_10.0', 'catv_13.0', 'catv_14.0', 'catv_15.0', 'catv_16.0',
       'catv_17.0', 'catv_2.0', 'catv_20.0', 'catv_21.0', 'catv_3.0',
       'catv_30.0', 'catv_31.0', 'catv_32.0', 'catv_33.0', 'catv_34.0',
       'catv_35.0', 'catv_36.0', 'catv_37.0', 'catv_38.0', 'catv_39.0',
       'catv_40.0', 'catv_41.0', 'catv_42.0', 'catv_43.0', 'catv_50.0',
       'catv_60.0', 'catv_7.0', 'catv_80.0', 'catv_99.0', 'DENS_2.0',
       'DENS_3.0', 'manv_1.0', 'manv_10.0', 'manv_11.0', 'manv_12.0',
       'manv_13.0', 'manv_14.0', 'manv_15.0', 'manv_16.0', 'manv_17.0',
       'manv_18.0', 'manv_19.0', 'manv_2.0', 'manv_20.0', 'manv_21.0',
       'manv_22.0', 'manv_23.0', 'manv_24.0', 'manv_25.0', 'manv_26.0',
       'manv_3.0', 'manv_4.0', 'manv_5.0', 'manv_6.0', 'manv_7.0', 'manv_8.0',
       'manv_9.0', 'obsm_1.0', 'obsm_2.0', 'obsm_4.0', 'obsm_5.0', 'obsm_6.0',
       'obsm_9.0']

    # traduction description > code ---------------------------------------------------------------
    
    def code_depuis_description(mapping_df, nom_col_code, description_selectionnee):
        # mapping_df a 2 colonnes : 
        #           1 - par exemple secu1 qui contient le code 
        #           2 - description
        # l'utilsateur a selectionné une description et on veut retrouver le code correspondant
        mapping_df_selected = mapping_df.loc[mapping_df["description"]==description_selectionnee]
        return mapping_df_selected[nom_col_code].iloc[0]

    valeurs_selectionnees = {
        "secu1": code_depuis_description(secu1_mapping, "secu1", secu1_choisi),
        "catv":  code_depuis_description(catv_mapping,  "catv",  catv_choisi),
        "DENS":  code_depuis_description(DENS_mapping,  "DENS",  DENS_choisi),
        "manv":  code_depuis_description(manv_mapping,  "manv",  manv_choisi),
        "obsm":  code_depuis_description(obsm_mapping,  "obsm",  obsm_choisi),
    }

    # construction du dataframe X_a_tester avec les colonnes attendues par le modèle---------------------------------------------------------------

    colonnes_a_passer_a_1 = []
    for feature, code in valeurs_selectionnees.items():
        colonne = f"{feature}_{code}"
        colonnes_a_passer_a_1.append(colonne)
        #st.write(f"Variable : {feature} | Code sélectionné : {code} | Colonne one-hot à créer : {colonne}")

    # intialise le dataframe avec toutes les colonnes à 0
    X_a_tester = pd.DataFrame(0, index=[0], columns=colonnes_pour_modele, dtype="int")

    # met à 1 les colonnes existantes dans le modèle et selctionnées par l'utilisateur
    colonnes_a_passer_a_1_et_connues = [c for c in colonnes_a_passer_a_1 if c in colonnes_pour_modele]
    X_a_tester.loc[0, colonnes_a_passer_a_1_et_connues] = 1

    # nota : il est possible l'utilisateur ait selectionné une modalité supprimée par le drop='first' de l'encodeur
    # elle sera donc absente de colonnes_a_passer_a_1_et_connues et donc tout restera à 0 pour cette feature


    # Prédiction--------------------------------------------------------------- ---------------------------------------------------------
    if st.button("Prédire la gravité"):
        modele = charger_modele('best_xgb_model_pickle.pkl')
        pred = modele.predict(X_a_tester)[0]

        # interprétation selon le remapping fait à l'étape 4.2 SPLIT
        # 1 (Indemne) → 0 | 4 (Blessé léger) → 1 | 3 (Blessé hospitalisé) → 2 | 2 (Tué) → 3
        grav_labels = {0: "Indemne", 1 :"Blessé léger", 2: "Blessé hospitalisé", 3:"Tué"}
        libelle = grav_labels[pred]
        # afficha dans un cadre vert (st.success) la gravité pédite en gras (la paire de **)
        st.success(f"Gravité prédite : **{libelle}**")

        proba = modele.predict_proba(X_a_tester)[0]
        st.dataframe(
            pd.DataFrame({
                 "classe": [grav_labels.get(i, i) for i in range(len(proba))],
                "probabilité": proba.round(2)
            }),
            hide_index=True
            )


# ==========================================================
# Streamlit - Partie MAP (cartes)
# ==========================================================

if page == pages[4]:
    st.header("Carte des accidents")
    #dossier_map = "./FOR_STREAMLIT/MAP/"
    dossier_map = Path(__file__).parent / "FOR_STREAMLIT" / "MAP"

    st.write("### Heatmap")
    chemin_carte = dossier_map.joinpath("heatmap_accidents.html")
    contenu_html = chemin_carte.read_text(encoding="utf-8")
    components.html(contenu_html, height=700, scrolling=True)

    st.write("### Maillage (tuiles h3)")
    st.markdown("Chaque tuile fait environ 450m de côté. La tuile sera noire s'il y a plusieurs accidents impliquant au moins un usager tué ou hospitalisé, rouge s'il y a 1 accident impliquant au moins un usager tué ou hospitalisé, jaune s'il y a 1 ou plusieurs accidents mais aucun usager tué ou hospitalisé.")
    chemin_carte = dossier_map.joinpath("carte_risque.html")
    contenu_html = chemin_carte.read_text(encoding="utf-8")
    components.html(contenu_html, height=700, scrolling=True)