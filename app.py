# on écrit le code ci-dessous dans le fichier app.py
# qui va ensuite être exécuté pour lire le streamlit

#######################
##### LES IMPORTS #####
#######################

import sre_compile
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.api
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
import matplotlib.pyplot as plt



################################################
##### INSTANCIATION DES VARIABLES GLOBALES #####
################################################

df_raw = pd.read_csv('weatherAUS_raw.csv')



#########################################
##### PAGE DE GARDE  / INTRODUCTION #####
#########################################

def introduction():
  st.title('Prévisions météo en Australie')
  st.caption('projet ML pour formation bootcamp data scientist')

  st.image('rainfallAUS.jpg')
  st.caption('source : Australian Government Bureau of Meteorology Climate Data Online; copyright Commonwealth of Australia')

  st.markdown("# Prédire s'il va pleuvoir le lendemain")
  st.markdown('''---''')
  
  st.markdown("#### Yassine HAMDAOUI, JingYi LIU, Julien PROST, Mohamed TOUMI")
  st.caption('Avril 2022 - Bootcamp Data scientist')
  st.markdown('''---''')
  
  st.markdown("### Contexte et objectifs :")
  st.markdown("""
    Ce streamlit vise à traiter une problématique de classification dans le cadre d’un projet réalisé au cours de la formation bootcamp data scientist (avr 2022) avec Datascientest.  \n
    L’objectif de ce projet est de prédire s'il va pleuvoir le lendemain (j+1) sur la base des indicateurs météorologiques du jour actuel (j) en Australie.  \n
    L'ensemble de données contient des informations sur les observations météorologiques recueillies pendant dix ans auprès de diverses stations météorologiques australiennes. Le temps est défini comme « pluvieux » si les précipitations sont de 1 mm ou plus. Les données sont recueillies auprès du Bureau de météorologie du gouvernement australien. 
    """)



####################################################
##### DATASET / PRESENTATION DU JEU DE DONNEES #####
####################################################

def dataset(df_dataset):
  st.title('Présentation du jeu de données')

  st.markdown("## Description du jeu de données et des variables")
  st.write("Le dataset utilisé pour ce projet provient du site Kaggle.com. Le lien est indiqué ci-dessous :")
  st.write("[lien vers le dataset](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)")
  st.write("L'ensemble de données comprend 23 colonnes et 145 460 lignes.")
  st.write("Il existe des caractéristiques quantitatives telles que la température maximale et minimale, l'évaporation, la durée d'ensoleillement et la vitesse du vent, des caractéristiques qualitatives telles que les dates, les lieux, la direction du vent et deux caractéristiques catégorielles binaires (Yes, No) indiquant s'il a plu ce jour-là et s'il va pleuvoir le lendemain.")

  st.markdown("## Lecture du jeu de données")
  ## Lecture du jeu de données
  with st.echo():       # pour afficher une partie du code
    df = pd.read_csv('weatherAUS_raw.csv')

  st.dataframe(df_dataset)

  st.markdown("## Description statistique des variables")
  st.dataframe(df_dataset.describe())

  st.write("On observe que les variables ont dans l’ensemble une répartition équilibrée, hormis les variables Pressure9am, Pressure3pm, RainToday et RainTomorrow qui sont très focalisées (rapport std/mean <15%).")



############################################
##### ANALYSE EXPLORATOIRE DES DONNEES #####
############################################

def data_explore(df_explore):
  st.title('Analyse exploratoire des données')
  st.write("Ci-dessous, quelques premières visualisations pour quelques variables. Des visualisations complémentaires et plus poussées sont présentées dans le reste du document, notamment dans la partie sur la sélection des variables.")
  st.write("Nous présentons ensuite le nettoyage et le preprocessing des données.")  

  st.markdown("<style>.streamlit-expanderHeader {font-size: xx-large;}</style>",unsafe_allow_html=True)   # pour configurer la police de st.expander
  with st.expander("Visualisations"):
  #  st.markdown("## Visualisations")
    graph_choisi = st.selectbox(label = 'choix du graphique', 
                                options = ['Distribution de fréquence de la variable RainTomorrow', 
                                          'Corrélation entre ensoleillement et précipitation en mm', 
                                          'Corrélation entre ensoleillement et évaporation',
                                          'Corrélation entre nuages et pluie le lendemain',
                                          'corrélation entre humidité et RainTomorrow', 
                                          'Influence des saisons', 
                                          'Influence de la pression atomosphérique',])
    
    if graph_choisi == 'Corrélation entre nuages et pluie le lendemain':
      st.image('Cloud9am_RainTomorrow.jpg')
      st.write("Le pourcentage de nuages à 9h du matin est concentré sur les valeurs élevées lorsqu'il a plu le lendemain (yes) - boite Orange.")
      st.write("En revanche, on voit s'agissant de la boite bleue (No) indiquant qu'il n'a pas plu le lendemain, que la distribution de la densité de nuages à 9h du matin est bien plus large, et donc moins porteuse d'information")  
    
    if graph_choisi == 'Corrélation entre ensoleillement et précipitation en mm':
      fig = plt.figure(figsize=(10, 4))
      sns.lineplot(data= df_raw,x="Sunshine",y="Rainfall",color = "green");
      st.pyplot(fig)
      st.write("Nous pouvons voir que l'ensoleillement est inversement proportionnel aux précipitations:")
    
    if graph_choisi == 'Corrélation entre ensoleillement et évaporation':
      fig = plt.figure(figsize=(10, 4))
      sns.lineplot(data= df_raw,x="Sunshine",y="Evaporation",color = "blue");
      st.pyplot(fig) 
      st.write("Phénomène inverse, nous pouvons voir que l'ensoleillement (heures) est proportionnel avec l’évaporation (mm)")

    if graph_choisi == 'Distribution de fréquence de la variable RainTomorrow':
      st.image('RainTommorow_count.jpg')
      st.write("Notre première expression à la vue de cette figure est que l'Australie n'est apparemment pas un territoire où il pleut beaucoup. Comme l'indiquent les chiffres, la majorité des journées ne connaissent pas de précipitation. Une répartition totalement déséquilibrée susceptible d'influencer notre modélisation ")
      st.write("Dans le langage de la science des données, c'est ce qu'on appelle un Dataset déséquilibré (78%,  22%)")
    
    if graph_choisi == 'corrélation entre humidité et RainTomorrow':
      fig = plt.figure(figsize=(10, 4)) 
      sns.boxplot(data= df_raw,x='RainTomorrow',y = "Humidity3pm");
      st.pyplot(fig)
      st.write("Le pourcentage d'humidité à 15 heures est élevé les jours où il a plu le lendemain.")
      
    if graph_choisi == 'Influence des saisons':
      st.image('serie_temp_date.jpg')
      st.image('influence_month.jpg')
      st.write("Les deux graphiques ci-dessus indiquent clairement l’influence (significative) des mois/saisons sur les variables (exemple ici avec les variables MinTemp et MaxTemp). Nous observons bien une tendance stationnaire ainsi qu’une saisonnalité à l’échelle de l’année .")
    
    if graph_choisi == 'Influence de la pression atomosphérique':
      st.write("Pression moyenne : 1013.25hPa au niveau de la mer.")
      st.write("Le pression change en fonction de l'altitude : environ 900hPa à 1000m, 700hPa à 3000m.")
      st.write("Les météorologues analysent donc les variations de pression (dépressions et anticyclones) pour ne pas avoir à intégrer l'altitude dans les calculs.")
      st.write("Nous allons créer une nouvelle colonne 'Pressur_diff' qui prendra en compte la différence de pression entre 9am et 3pm.")
      with st.echo():       # pour afficher une partie du code
        df_raw['Pressure_diff']=df_raw['Pressure3pm']-df_raw['Pressure9am']
      
      fig = plt.figure(figsize=(10,6))
      sns.boxenplot(x='RainTomorrow',y='Pressure_diff',data=df_raw);
      st.pyplot(fig)

      st.write("Bien que le résultat visuel n'indique pas de manière évidente l'impact devariation de pression sur la précipitation, celle-ci est bien démontrée par un test statistique :")
      result=statsmodels.formula.api.ols('Pressure_diff~RainTomorrow',data=df_raw).fit()
      table=statsmodels.api.stats.anova_lm(result)
        
      st.write("la p-value est très inférieure à 5% ==> l'hypothèse H0 est donc rejetée")
  
  with st.expander("Nettoyage"):
  #st.markdown("## Nettoyage")

    ## checkbox
    if st.checkbox('Afficher valeurs manquantes'):
      st.dataframe(df_explore.isna().sum())

      st.write("Avant de réaliser la modélisation , il était nécessaire de procéder à quelques traitements des données. Nous avons commencé par faire un regroupement des villes par Etat (State).\
      Ensuite, Nous avons diviser la  colonne date en jour, mois et année car elle a un impact sur les prédictions")

    with st.echo():       
      df_raw['Year']=pd.to_datetime(df_raw['Date']).dt.year
      df_raw['Month']=pd.to_datetime(df_raw['Date']).dt.month
      df_raw['Day']=pd.to_datetime(df_raw['Date']).dt.day
      
    st.write("Afin de traiter les treize colonnes avec NA de notre ensemble des données, et comme nous l’avons vu précédemment, il y a une saisonnalité à l’échelle de l’année sur l’ensemble des données recueillies" 
            " Nous avons donc décidé de :")

    st.write("1 - Faire un regroupement des variables par mois ( groupby('Month'))")
    st.write("2 - Remplacer les Na par la moyenne observée durant le mois ")
    st.write("3 - Supprimer les colonne avec un taux qui avoisine 50'%' de Na (insignifiante pour les calculs)")
    st.write("4 - Supprimer l'année 2007 et 2008 (Relevés fortement minoritaires, et présentants des valeurs abbérantes en surnombre)")
  
    if st.checkbox("Afficher les Relevés par année"):
      st.image('Afficher les Relevés par année.jpg')
      
    st.write("5 - Supprimer toutes les observations compartant des Na pour notre variable cible")
              
  with st.expander("Preprocessing"):
    #st.markdown("## Preprocessing")
    st.write("Il est impératif de bien préparer nos données avant leur passage dans la machine.")         
    st.write("**Tout d'abord nous examinons les colonnes catégorielles** de notre ensemble de données")   
    if st.checkbox("Afficher les variables catégorielles"):
      st.image('Variables.jpg')
    st.caption("Dichotomisation des variables qualitatives")

    with st.echo():       
      list_dichotom=['WindGustDir','WindDir9am','WindDir3pm','Cloud9am','Cloud3pm','State','Year','Month','Day']
      #df=pd.get_dummies(data=df,columns=list_dichotom)
    st.write("Nous avons choisi la méthode **get_dummies** pour adresser la problématqiue de hiérarchie.")
    
    '''
    st.caption("Graphe d'évolution de pluviométrie")
    st.image('evolution.jpg')
    st.write("Le graphe ci-dessus confirme la saisonnalité et notre choix d'extraction de l'année du mois")
    '''
    
    st.caption("Standarisation")
    st.write("Nous pouvons  clôturer cette étape de preprocessing par une **normalisation** qui permet de mettre sur\
    une même échelle toutes les variables quantitatives en utilisant la méthode **StandardScaler()**")
    st.image("scaler.jpg")



##############################################
##### SELECTION DES VARIABLES INFLUENTES #####
##############################################

def feature_select():
  st.title('Sélection des variables influentes')
  
  df = pd.read_csv('weatherAUS_raw.csv')
  st.markdown("<style>.streamlit-expanderHeader {font-size: x-large;}</style>",unsafe_allow_html=True)   # pour configurer la police de st.expander
  with st.expander("1. Rapel du taux de valeurs manquantes:"):
    st.dataframe( ((df.isna().sum())/df.shape[0]).sort_values(ascending=False))
    st.write("Le premier volet de la sélection des variable porte sur la prise en compte ou non des variables comportant des taux importants de NA.")
    st.write("Il est communément observé qu’un seuil de 25 à 30 % de valeurs manquantes peut être acceptable. Au-delà, les caractéristiques concernées doivent être supprimées de l'analyse. Il s’agit dans notre cas de :")
    st.image("cycle de l'eau.PNG")
    st.write(" ### - Sunshine")
    st.write("L'ensoleillement est lié au début du cycle de l'eau.")
    st.write("**avis métier :** a priori pas d'impact immédiat sur la pluie du lendemain > **supprimer** ")
    st.write(" ### - Evaporation")
    st.write("L'évaporation est l'une des premières étapes du cycle de l'eau ou de la pluie.")
    st.write("**avis métier :** a priori pas d'impact direct sur la pluie du lendemain > **supprimer**")
    st.write(" ### - Cloud")    
    st.write("La formation de nuages précède la précipitation dans le cycle de l'eau ou de la pluie.")
    st.write("**avis métier :** il y a donc un lien  direct entre nuages et pluie > **garder**")

  with st.expander("2. La distribution des variables numériques:"):
    st.write("Le second volet abordant la sélection des variables se base sur l’analyse de la distribution des variables")
    df_quantitative = [column for column in df.columns if df[column].dtype == float]
    df_quantitative_not_Pressure = [column for column in df_quantitative if column not in ['Pressure9am', 'Pressure3pm']]
    if st.checkbox('Distribution de **toutes** les variables quantitatives:', value = True):
      fig = plt.figure(figsize=(10,10))
      df.boxplot(column = df_quantitative)
      plt.xticks(rotation=90)
      st.pyplot(fig)
      st.write("les valeurs de pression sont beaucoup plus grandes que les autres variables, et ont donc tendance à les écraser en affichage. Par conséquent, afin de mieux visualiser la distribution, nous présentons les variables (cf. figure ci-dessous) sans les 2 variables de pression")
    if st.checkbox('Distribution des variables quantitatives **sans** les variables Pression:', value = False):
      fig = plt.figure(figsize=(10,10))
      df.boxplot(column = df_quantitative_not_Pressure)
      plt.xticks(rotation=90)
      st.pyplot(fig)
      st.write("De nombreux points semblent atypiques dans la variable Rainfall. Mais sont-ils des outliers pour autant ?")

    st.write("### Les outliers?")
    if st.checkbox('Afficher les détails de RainFall'):
      fig, axs = plt.subplots(1,2,figsize=(18, 5))
      sns.histplot(x='Rainfall',data = df,bins=20, kde=True,ax=axs[0])
      sns.boxplot(x='Rainfall', data = df,ax = axs[1], color='#99befd', fliersize=1)
      st.pyplot(fig)
      st.write("Ces points atypiques  correspondent bien à des points observés dans la réalité. En effet, dans les zones plutôt sèches, le nombre de jours sans pluie peut être important. Mais de manière exceptionnelle, il peut y avoir une pluie forte avec une précipitation supérieure à 200mm. Nous décidons donc de ne pas corriger ces données.")
    st.write("**Par conséquent, il n'y a pas de traitement effectué dans cette partie de l’analyse.**")

  with st.expander("3. Corrélation entre variables explicatives:"):
    st.write("Nous réalisons une analyse de corrélation entre les variables explicatives quantitatives. Nous faisons cette analyse à l’aide d’une heatmap")
    st.image('heatmap.PNG')
    st.write("Nous pouvons constater dans la heatmap qu’il existe une forte corrélation entre certains variables:")
    st.write(" - Temp3pm and MaxTemp  **98%**")
    st.write(" - Pressure3pm and Pressure9am  **96%**")
    st.write(" - Temp9am and MinTemp   **91%**")
    st.write(" - Temp9am and MaxTemp  **89%**")
    st.write(" - Temp3pm and Temp9am  **87%**")
    st.write("Nous pouvons confirmer ces corrélations par les tests de corrélation de Pearson")
    if st.checkbox('Afficher les p_value'):
      st.image('p_value_var_corr.PNG')
    st.write("Au-delà de 50%, il est possible de considérer que la corrélation est forte entre 2 variables.  Pour éviter de supprimer trop d’ informations , nous ne  regardons que les variables avec une corrélation supérieure à 80%")
    st.write("Par conséquent, nous pourrons supprimer les variables suivantes : **MaxTemp, MinTemp, Pressure3pm**")
   
  with st.expander("4. L’impact des variables (quantitatives) sur la cible - visualisation:"):
    st.write("Nous nous intéressons à la corrélation entre les variables explicatives (quantitatives) et la variable cible. Pour cela nous utilisons une approche  basée sur des visualisations.")
    st.write("Lorsque les zones bleue et orange coincident, la variable explicative n'a alors pas ou peu d'impact sur la variable cible.")
    st.image('impact sur la cible.PNG')
    st.write("Nous pouvons ainsi voir que la variable Sunshine, par exemple, a un impact sur la variable cible, comme les zones bleue et orange sont bien disjointes.")
    st.write("Au final, nous pourrons supprimer les variables suivantes qui ne sont pas impactantes pour la variable cible : **MaxTemp, Evaporation, WindSpeed9am, WindSpeed3pm, Temp9am**")

  with st.expander("5. Model Feature selection:"):
    st.write("Avec sklearn, il est possible de sélectionner k features (ou variables) des features de notre dataset avec le sélecteur **SelectKBest**, qui se base sur une métrique prédéfinie (score_func). Tel est l’objet de cette partie de l’analyse")
    st.write("Pour ce faire, nous avons séparé les variables (quantitatives d’une part, et explicatives d’autre part) et appliqué les métriques suivantes :")
    st.write(" - chi2 **>** pour les variables qualitatives")
    st.write(" - mutual_info_classif et f_classif **>** pour les variables quantiatives")
    st.write("**Résultat:**")
    st.image('selectKBest_1.PNG')
    st.write("La liste de variables pour la modélisation : ")
    st.write("**RainToday, Cloud9am_0.0, Cloud3pm_8.0,Cloud9am_8.0,Cloud3pm_1.0, Humidity_9am, Pressure_diff, Sunshine, Humidity_3pm, Rainfall**")


    df_feature = pd.read_csv('feature_selection.csv')

    if st.checkbox('Analyses complémentaires'):
      st.write("#### Traitement des variables Cloud")
      st.caption("Cloud > groupe quantitatif")
      st.write("Le feature score des variables Cloud est faible, et donc conduirait à ne pas les garder en première approche.")
      st.write("Toutefois, nous nous rendons compte qu’il faut peut-être traiter le variable Cloud comme une variable quantitative. Même si les valeurs sont discrètes, comme 0,1, etc. mais La définition de Cloud est le pourcentage de surface du ciel couvert par le nuage. Par exemple Cloud =1, veut dire, 10% de ciel est couvert par le nuage")
      C9_RTm = pd.crosstab(df_feature['Cloud9am'],df_feature['RainTomorrow'], normalize= 0)
      C9_RT = pd.crosstab(df_feature['Cloud9am'],df_feature['RainToday'], normalize= 0)
      C3_RT = pd.crosstab(df_feature['Cloud3pm'],df_feature['RainToday'], normalize= 0)
      C3_RTm = pd.crosstab(df_feature['Cloud3pm'],df_feature['RainTomorrow'], normalize= 0)
      fig = plt.figure(figsize=(4,4))
      plt.plot(C9_RT.index,C9_RTm[1], label = 'C9am_RainTomorrow')
      plt.plot(C9_RT.index,C9_RT[1], label = 'C9am_RainToday')
      plt.plot(C3_RT.index,C3_RTm[1], label = 'C3pm_RainTomorrow')
      plt.plot(C3_RT.index,C3_RT[1], label = 'C3pm_RainToday')
      plt.xlabel('Cloud level')
      plt.ylabel('Probability of rain')
      plt.legend()
      st.pyplot(fig)
      st.write("Ces courbes montrent bien une forte corrélation entre le nuage (à 9h et à 15h) et la pluie (aujourd’hui et demain). **Nous décidons donc de garder ces 2 variables Cloud**.")

      st.write("#### Des résultats différents selon la valeur de K ?")
      st.write("Nous avons constaté des résultats différents en fonction de la valeurs de K:")
      st.image('selectKBest_1_k=5_10_20.PNG')

      st.write("Après, nous avons lancé une 2ème fois le SelectKBest, cette fois sur le dataset initial (seulement corrigé des NA), et nous essayons de vérifier si les résultats sont 'corrects': ")
      st.image('selectKBest_2_qualitative.PNG')

      st.write("##### L'impact de RainToday sur RainTomorrow")
      st.write("RainToday continue à occuper la première place.")
      st.image('impact_raintoday_raintomorrow.PNG')
      st.write("Pour la 2e place, nous avons state_Norfolk")

      st.write("##### L'impact de NorfolkIsland sur RainTomorrow:")
      st.write("Nous avons trouvé les informations suivantes : ")
      st.image('AUS Rainfall 2021.PNG')
      st.image('Norfolk rain.PNG')
      st.write("C’est une île, où la précipitation est assez élevée par rapport aux autres Etats (sauf Tasmania où la pluviométrie est équivalente, autour de 1300 mm comme indiqué dans les figures ci-dessous). **Nous pouvons considérer que Norfolk Island influe beaucoup sur la prédiction de la pluie**, donc cette deuxième position est donc compréhensible.")
      st.image('AUS Rainfall 2021 supposition.PNG')

      st.write("##### L'impact de WindDirction sur RainTomorrow:")
      st.image('Winddir impact.PNG')
      st.write("Les 2 courbes (orange et bleue) ne sont pas confondues, les directions de vent portent un impact sur la pluie, mais il est difficile de conclure parmi les 16 directions, **nous pouvons garder toutes les variables ‘Wind’ avec une importance positive**")

      st.write("##### L'impact de Month sur RainTomorrow:")
      st.write("Nous avons pensé que ‘Month’ pourrait avoir un impact sur la pluie, mais il n'a pas été choisi par SelectKBest. Nous avons vérifié ce point avec la graphique de comparaison")
      st.image('impact month.PNG')
      st.write("L’écart entre les 2 courbes (orange/bleue) est très faible. **Nous pouvons considérer que  ‘Month’ n’est pas une variable importante pour la prédiction de pluie**, confirmant le résultat de KBest")
      st.write("##### Les variables quantitatives:")
      st.image('selectKBest_quantitative.PNG')

  with st.expander("Conclusion:"):
    st.write("La combinaison des analyses précédentes nous conduit à retenir les variables suivantes pour la modélisation : ")
    st.write("**Pressure9am, Sunshine, Temp3pm, Cloud3pm, Humidity3pm, Rainfall, WindGustSpeed, RainToday, State_NorfolkIsland, WindDir3pm_W, WindGustDir_W, WindGustDir_NW, WindDir3pm_NW, WindDir9am_NNE, WindDir9am_E, WindDir9am_ENE**")



#########################
##### MODELISATIONS #####
#########################

def models():
  st.title('Les modélisations')
  st.write("Divers algorithmes de classification supervisée ont été testés. Afin d'en tirer les meilleures performances, un tunning des hyperparamètres par validation croisée (GridSearchCV) a été effectué sur chacun de ces algorithmes.")
  st.write("")

  # partie JL
  if st.checkbox('Rappel de la liste de variables utilisées'):
    st.write("Pressure9am, Sunshine, Temp3pmn Cloud3pm, Humidity3pm, Rainfall, WindGustSpeed, RainToday, State_NorfolkIsland, WindDir3pm_W, WindGustDir_W, WindGustDir_NW, WindDir3pm_NW, WindDir9am_NNE, WindDir9am_E, WindDir9am_ENE")

  st.markdown("<style>.streamlit-expanderHeader {font-size: x-large;}</style>",unsafe_allow_html=True)   # pour configurer la police de st.expander
  with st.expander("Régression logistique"):
    st.image('model_lr.PNG')

  with st.expander("Arbre de décision"):
    st.image('model_dt.PNG')

  with st.expander("Random Forest"):
    st.image('model_rf.PNG')
  
  with st.expander("XG Boost"):
    st.image('model_xgboost.PNG')

  with st.expander("Voting"):
    st.write("L'algorithme d'ensemble learning \"Voting\" a été entraîné avec les modèles essayés ci-dessus. Les meilleurs hyperparamètres ont été retenus. Nous avons ainsi utilisé la régression logistique, les arbres de décision, la random forest et le KNN. Il en ressort des résultats au moins aussi bons que le meilleur des modèles, sans toutefois le surpasser.")   
    st.image('Voting.PNG')

  with st.expander("Stacking"):
    st.write("Pour cet algorithme d'ensemble learning de \"Stacking\", la façon de procéder est la même que le voting ci-dessus. Nous obtenons également des résultats au moins aussi bons que le meilleur des modèles, sans toutefois le surpasser.")   
    st.image('Stacking.PNG')

  with st.expander("Deep learning"):
    st.write("Nous avons pour terminer entraîné un modèle de type \"multi-layer perceptron\". Les meilleurs résultats ont été obtenus avec une architecture à 3 couches. Les 2 premières comportent 32 neurones quand la dernière en compte 2. L'optimizer est Adam et le learning rate défini à 0.001. La fonction de perte utilisée est binary_crossentropy. Nous avons entraîné le modèle sur une taille de batch de 10 et 40 epochs.")   
    st.image('Deep_Learning.PNG')
    st.write("Un callback de type ReduceLROnPlateau a été mis en place et nous constatons bien son action sur la courbe de train_loss")
    st.write("Nous obtenons ici les meilleurs résultats tous modèles confondus avec une accuracy sur le dataset de validation de plus de 0.835!")

  with st.expander("Conclusion comparative"):
    st.write("Le modèle donnant le meilleur score est le XGBoost avec un F1-score de 0.83. Il est suivi de très près par le perceptron multi-couches. Ce dernier offre cependant la meilleur accuracy avec 0.84. Aucun de ces modèles ne sort particulièrement du lot.")   



#############################
##### PAGE DES CONTACTS #####
#############################

def contacts():
  st.title('Contacts des participants au projet')
  st.markdown("### Yassine HAMDAOUI")
  st.write("[link](https://www.linkedin.com/in/yassine-hamdaoui-17a738131/?originalSubdomain=fr)")
  st.markdown("### JingYi LIU")
  st.write("[link](https://www.linkedin.com/in/jingyi-liu-370b5055/)")
  st.markdown("### Julien PROST")
  st.write("[link](https://www.linkedin.com/in/julien-prost-360aab107/)")
  st.markdown("### Mohamed TOUMI")
  st.write("[link](https://www.linkedin.com/in/mohamed-toumi-30035917/)")



#########################################
##### PAGE DU BILAN ET PERSPECTIVES #####
#########################################

def perspectives():
  st.title('Bilan et perspectives')
  st.caption('regard critique sur le déroulement et les résultats du projet')
  st.write("En prenant un peu de recul sur le projet, nous pouvons retenir notamment les deux éléments suivants comme structurants pour la construction d'une modélisation robuste.")
  st.write("Tout d'abord, la **sélection des variables**:")
  st.write("Le preprocessing nous a rapidement conduit à traiter un grand nombre de variables (plus de 140), rendant les modèles de machine learning très lents, voire impossibles à exécuter pour certains, sur nos machines.")
  st.write("La partie sélection des variables a donc été fondamentale dans la définition de notre modèle.")
  st.write("La difficulté a été de faire la passerelle entre les variables métier d'une part, qui selon toute vraisemblance doivent avoir un rôle important à jouer dans la classification de la cible, et d'autre part, les variables issues des analyses de corrélation.")
  st.write("Cette analyse est d'autant plus ardue qu'elle nécessite de prendre en compte certains biais, comme un nombre important de NA pour certaines variables, ou encore la prise en compte de certaines variables comme quantitatives plutot que qualitatives.")
  st.write("Enfin, un axe de travail qui nécessite un effort d'investigation supplémentaire à l'issue de ce projet, consiste à comprendre ce qui peut conduire à l'obtention de listes de variables différentes selon la valeur du K choisi pour un Select KBest")
  st.write("Cette problématique nous a conduit, comme nous l'indiquons dans le rapport du projet, à mener des analyses poussées, en deux temps, de la pertinence des variables, en nous reposant notamment le plus possible sur les connaissances métier.")

  st.write("\n")
  st.write("Ensuite, l'affinage des paramètres du **modèle de deep learning**:")
  st.write("Après avoir mis en oeuvre les algorithmes classiques de machine learning, nous avons essayé de construire un algorithme de deep learning.")
  st.write("Comme indiqué dans la rubrique correspondante, cet algoritme nous a permis d'obtenir les meilleurs résultats de classification.")
  st.write("Toutefois, nous avons manqué de temps pour affiner, autant que faire se peut, le choix des hyperparamètres de ce modèle. Il s'agit donc d'un axe de travail supplémentaire identifé pour l'après-projet")
  st.write("Une ouverture serait de tester un modèle de deep learning avec architecture RNN afin d'étudier la séquentialité des données. Peut-être l'étude sur 10 jours glissants permettrait d'être plus précis dans les prévisions")



#############################
##### AFFICHAGE DU MENU #####
#############################

with st.sidebar:
  selected = option_menu(menu_title = 'Menu principal',
                         options = ['Introduction', 'Dataset', 'Analyse exploratoire', 'Sélection des variables', 'Modélisations', 'Bilan et perspectives', 'Contacts'],
                         default_index = 0)

if selected == 'Introduction':
  introduction()
if selected == 'Dataset':
  dataset(df_raw)
if selected == 'Analyse exploratoire':
  data_explore(df_raw)
if selected == 'Sélection des variables':
  feature_select()
if selected == 'Modélisations':
  models()
if selected == 'Contacts':
  contacts()
if selected == 'Bilan et perspectives':
  perspectives()