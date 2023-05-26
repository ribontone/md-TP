import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import glob

import pprint
import copy

import requests
import json
import time
from datetime import datetime, timedelta
from pandas.plotting import scatter_matrix


from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
#import pymc3 as pm

from sklearn.preprocessing import LabelEncoder



datasets = {}
for filename in glob.glob('./datasets/20**-**.csv'):
    df = pd.read_csv(filename)
    key = filename.replace('.csv', '')
    key = key.replace('./datasets\\', '')
    datasets[key] = df






datasetPosicoes=pd.read_csv('./datasets/EPLStandings.csv')
#calcular o numero de Missing values em cada linha 
missing_values = datasetPosicoes.isnull().sum(axis=1)
#ordenar o dataframe por NaNs 
df_sorted = datasetPosicoes.iloc[missing_values.argsort()]
df_sorted = df_sorted.reset_index(drop=True)

datasetPosicoes=df_sorted





#uniformizar os dados
#para cada um retirar a coluna DIV (que nao serve para nada, e tudo 1a divisao inglesa)
# attendance(so nos primeiros 2) hhw ahw hit woodwork offsides hora do jogo
for key in datasets:
    if isinstance(datasets[key], pd.DataFrame):
        datasets[key] = datasets[key].drop("Div", axis=1)
        
        if "Attendance" in datasets[key].columns:
           datasets[key] = datasets[key].drop("Attendance", axis=1)
        
        if "HHW" in datasets[key].columns:
           datasets[key] = datasets[key].drop("HHW", axis=1)
        
        if "AHW" in datasets[key].columns:
           datasets[key] = datasets[key].drop("AHW", axis=1)  
           
        if "HO" in datasets[key].columns:
           datasets[key] = datasets[key].drop("HO", axis=1)
           
        if "AO" in datasets[key].columns:
           datasets[key] = datasets[key].drop("AO", axis=1) 
        
        if "Time" in datasets[key].columns:
           datasets[key] = datasets[key].drop("Time", axis=1) 
           

#ordenar colunas os primeiros 2 que sao diferentes        
ordem_colunas = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'Referee', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']

datasets['2000-01']=datasets['2000-01'].reindex(columns=ordem_colunas)
datasets['2001-02']=datasets['2001-02'].reindex(columns=ordem_colunas)

#adicionar Season a cada dataframe
for season, df in datasets.items():
    df = df.assign(Season=season)
    datasets[season] = df

#por season em primeiro em todos
for season, df in datasets.items():
    cols = list(df.columns)
    cols.insert(1, cols.pop())
    df = df[cols]
    datasets[season] = df




#fazer uma copia para nao mudar nada nos originais e poder comparar
datasetsPARAmanipular = copy.deepcopy(datasets)




# Define a function to calculate goals scored and conceded for each team
def calculate_goals(df):
    # Create a dictionary to keep track of goals scored and conceded for each team
    team_stats = {}

    # Initialize goals scored and conceded to 0 for all teams
    for team in set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()):
        team_stats[team] = {'GoalsScored': 0, 'GoalsConceded': 0,'DiferencaGolos': 0}

    # Iterate over all rows in the dataframe
    for index, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        home_goals = row['FTHG']
        away_goals = row['FTAG']
        
        
        # Add columns to the dataframe for goals scored and conceded for each team
        df.loc[index, 'gMarCasa'] = team_stats[home_team]['GoalsScored']
        df.loc[index, 'gSofrCasa'] = team_stats[home_team]['GoalsConceded']
        df.loc[index, 'gMarVis'] = team_stats[away_team]['GoalsScored']
        df.loc[index, 'gSofrVis'] = team_stats[away_team]['GoalsConceded']
        
        df.loc[index, 'difGolosCasa'] = team_stats[home_team]['DiferencaGolos']
        df.loc[index, 'difGolosVis'] = team_stats[away_team]['DiferencaGolos']

        # Update goals scored and conceded for home team
        team_stats[home_team]['GoalsScored'] += home_goals
        team_stats[home_team]['GoalsConceded'] += away_goals

        # Update goals scored and conceded for away team
        team_stats[away_team]['GoalsScored'] += away_goals
        team_stats[away_team]['GoalsConceded'] += home_goals
        
        team_stats[away_team]['DiferencaGolos'] = team_stats[away_team]['GoalsScored']-team_stats[away_team]['GoalsConceded']
        team_stats[home_team]['DiferencaGolos'] = team_stats[home_team]['GoalsScored']-team_stats[home_team]['GoalsConceded']



    return df





# Define a function to calculate points for each team
def calculate_points(df):
    # Create a dictionary to keep track of points for each team
    team_points = {}

    # Initialize points to 0 for all teams
    for team in set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()):
        team_points[team] = 0

    # Iterate over all rows in the dataframe
    for index, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        home_goals = row['FTHG']
        away_goals = row['FTAG']

        # Determine the result of the match
        if home_goals > away_goals:
            home_result = 'W'
            away_result = 'L'
            home_points = 3
            away_points = 0
        elif home_goals < away_goals:
            home_result = 'L'
            away_result = 'W'
            home_points = 0
            away_points = 3
        else:
            home_result = 'D'
            away_result = 'D'
            home_points = 1
            away_points = 1


        # Add columns to the dataframe for points for each team
        df.loc[index, 'pontosCasa'] = team_points[home_team]
        df.loc[index, 'pontosVis'] = team_points[away_team]
        
        # Add points to the team's total
        team_points[home_team] += home_points
        team_points[away_team] += away_points


    return df





def calculate_positions(df):
    df['posCasa'] = None
    df['posVis'] = None

    stats = pd.DataFrame(columns=['Team', 'Points', 'GoalDifference', 'GoalsScored'])

    for i in range(len(df)):
        stats.loc[stats['Team'] == df.loc[i, 'HomeTeam'], 'Points'] = df.loc[i, 'pontosCasa']
        stats.loc[stats['Team'] == df.loc[i, 'AwayTeam'], 'Points'] = df.loc[i, 'pontosVis']
        stats.loc[stats['Team'] == df.loc[i, 'HomeTeam'], 'GoalDifference'] = df.loc[i, 'difGolosCasa']
        stats.loc[stats['Team'] == df.loc[i, 'AwayTeam'], 'GoalDifference'] = df.loc[i, 'difGolosVis']
        stats.loc[stats['Team'] == df.loc[i, 'HomeTeam'], 'GoalsScored'] = df.loc[i, 'gMarCasa']
        stats.loc[stats['Team'] == df.loc[i, 'AwayTeam'], 'GoalsScored'] = df.loc[i, 'gMarVis']

        if df.loc[i, 'HomeTeam'] not in stats['Team'].values:
            stats = stats.append({'Team': df.loc[i, 'HomeTeam'], 'Points': df.loc[i, 'pontosCasa'], 
                                  'GoalDifference': df.loc[i, 'difGolosCasa'], 'GoalsScored': df.loc[i, 'gMarCasa']}, 
                                 ignore_index=True)
        if df.loc[i, 'AwayTeam'] not in stats['Team'].values:
            stats = stats.append({'Team': df.loc[i, 'AwayTeam'], 'Points': df.loc[i, 'pontosVis'], 
                                  'GoalDifference': df.loc[i, 'difGolosVis'], 'GoalsScored': df.loc[i, 'gMarVis']}, 
                                 ignore_index=True)

    # Sort the stats DataFrame outside the loop
    stats = stats.sort_values(by=['Points', 'GoalDifference', 'GoalsScored','Team'], ascending=[False, False, False,True]).reset_index(drop=True)
    #print("scatadadababa 1")
    #print("scatadadababa 2")
    # Assign positions to the final DataFrame outside the loop
    for i in range(len(df)):
        df.loc[i, 'posCasa'] = stats.index[stats['Team'] == df.loc[i, 'HomeTeam']][0] + 1
        df.loc[i, 'posVis'] = stats.index[stats['Team'] == df.loc[i, 'AwayTeam']][0] + 1
    
    #print("scatadadababa 3")
    #print("scatadadababa 4")
    
    return df






# Define a function to calculate the number of games played between two teams
def games_played_between(df, team1, team2):
    # Filter the dataframe to only include matches between the two teams
    filtered_df = df[((df['HomeTeam'] == team1) & (df['AwayTeam'] == team2)) | ((df['HomeTeam'] == team2) & (df['AwayTeam'] == team1))]

    # Return the number of matches played between the two teams
    return len(filtered_df)



def data_anterior(data_str):
    formatos = ["%d/%m/%y", "%d/%m/%Y"]
    for formato in formatos:
        try:
            data = datetime.strptime(data_str, formato)
            data_anterior = data - timedelta(days=1)
            return data_anterior.strftime(formato)
        except ValueError:
            pass
    raise ValueError("Formato de data inválido: " + data_str)












def calcula_posicao(df):
    # Dicionário para armazenar as estatísticas de cada equipe
    stats = {}

    # Adiciona as novas colunas ao dataframe original
    df['posCasa'] = 0
    df['posVis'] = 0

    # Itera sobre cada linha do dataframe
    for i in range(len(df)):
        row = df.loc[i]

        # Atualiza as estatísticas da equipe da casa
        if row['HomeTeam'] not in stats:
            stats[row['HomeTeam']] = {'Pontos': 0, 'Diferenca_Golos': 0, 'Golos_Marcados': 0}
        stats[row['HomeTeam']]['Pontos'] += row['pontosCasa']
        stats[row['HomeTeam']]['Diferenca_Golos'] += row['difGolosCasa']
        stats[row['HomeTeam']]['Golos_Marcados'] += row['gMarCasa']

        # Atualiza as estatísticas da equipe visitante
        if row['AwayTeam'] not in stats:
            stats[row['AwayTeam']] = {'Pontos': 0, 'Diferenca_Golos': 0, 'Golos_Marcados': 0}
        stats[row['AwayTeam']]['Pontos'] += row['pontosVis']
        stats[row['AwayTeam']]['Diferenca_Golos'] += row['difGolosVis']
        stats[row['AwayTeam']]['Golos_Marcados'] += row['gMarVis']

        # Se for a última linha da jornada, atualiza as posições de todas as equipes
        if (i+1) % 10 == 0:
            # Cria um dataframe temporário com as estatísticas das equipes
            temp_df = pd.DataFrame.from_dict(stats, orient='index')

            # Ordena as equipes por pontos, diferença de golos e golos marcados
            temp_df.sort_values(by=['Pontos', 'Diferenca_Golos', 'Golos_Marcados'], inplace=True, ascending=[False, False, False])

            # Atribui a classificação manualmente
            temp_df['Posicao'] = range(1, len(temp_df) + 1)

            # Atualiza as posições no dataframe original para todas as linhas da jornada
            for j in range(i, i - 10, -1):
                df.loc[j, 'posCasa'] = temp_df.loc[df.loc[j, 'HomeTeam'], 'Posicao']
                df.loc[j, 'posVis'] = temp_df.loc[df.loc[j, 'AwayTeam'], 'Posicao']


    return df




#funcao que seleciona apenas as colunas mais interessantes para debugg
def selecionar_colunas(df):
    colunas_desejadas = ['Date', 'HomeTeam', 'AwayTeam', 'gMarCasa', 'gSofrCasa', 'gMarVis', 'gSofrVis', 'difGolosCasa', 'difGolosVis', 'pontosCasa', 'pontosVis', 'posCasa', 'posVis']
    novo_df = df.loc[:, colunas_desejadas]
    return novo_df



def jogos_entre_duas_sempre(TeamA,TeamB):
    soma=0
    for season in datasetsPARAmanipular.keys():
       
        # Calculate the number of games played between two teams in a dataframe
        num_games = games_played_between(datasetsPARAmanipular[season], TeamA,TeamB)
        soma=soma+num_games
        
        # Print the number of games played between the two teams
        #print(f'{season}: Number of games played between {TeamA} and {TeamB}: {num_games}')

    
    
    return soma








def contar_jogos_entre_equipes_antes_data(data, equipe1, equipe2):
    count = 0
    data_limite = data_anterior(data)
    
    for temporada, dataframe in datasetsPARAmanipular.items():
        jogos_temporada = dataframe[(dataframe['HomeTeam'] == equipe1) & (dataframe['AwayTeam'] == equipe2) |
                                    (dataframe['HomeTeam'] == equipe2) & (dataframe['AwayTeam'] == equipe1)]
        jogos_temporada['Date'] = pd.to_datetime(jogos_temporada['Date'], format="%d/%m/%y", errors='coerce')  # Converter a coluna Date para o formato datetime
        jogos_antes_data = jogos_temporada[jogos_temporada['Date'] <= data_limite]
        count += len(jogos_antes_data)
    return count





#datasetsPARAmanipular
def _5_ultimos_jogos_entre_equipes(equipe1, equipe2, data_limite):
    ultimos_jogos = pd.DataFrame(columns=['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'Referee', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR'])
    for temporada, dataframe in datasetsPARAmanipular.items():
        jogos_temporada = pd.concat([
            dataframe[(dataframe['HomeTeam'] == equipe1) & (dataframe['AwayTeam'] == equipe2)],
            dataframe[(dataframe['HomeTeam'] == equipe2) & (dataframe['AwayTeam'] == equipe1)]
        ])
        jogos_temporada['Date'] = pd.to_datetime(jogos_temporada['Date'], format="%d/%m/%y", errors='coerce')  # Converter a coluna Date para o formato datetime
        
        data_limite = data_anterior(data_limite)
        
        jogos_temporada = jogos_temporada[jogos_temporada['Date'] < data_limite]
        ultimos_jogos_temporada = jogos_temporada.tail(5)[['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'Referee', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']]
        ultimos_jogos = pd.concat([ultimos_jogos, ultimos_jogos_temporada])
        ultimos_jogos = ultimos_jogos.reset_index(drop=True)
        ultimos_jogos = ultimos_jogos.sort_values(by='Date', ascending=False)
        ultimos_jogos = ultimos_jogos.head(5)[['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'Referee', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']]
        
        
    return ultimos_jogos



def _5_ultimos_jogos_uma_team(equipe1, data_limite):
    ultimos_jogos = pd.DataFrame(columns=['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'Referee', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR'])
    for temporada, dataframe in datasetsPARAmanipular.items():
        jogos_temporada = pd.concat([
            dataframe[(dataframe['HomeTeam'] == equipe1) | (dataframe['AwayTeam'] == equipe1)]        ])
        
        jogos_temporada['Date'] = pd.to_datetime(jogos_temporada['Date'], format="%d/%m/%y", errors='coerce')  # Converter a coluna Date para o formato datetime
        
        data_limite = data_anterior(data_limite)
        
        jogos_temporada = jogos_temporada[jogos_temporada['Date'] < data_limite ]
        ultimos_jogos_temporada = jogos_temporada.tail(5)[['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'Referee', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']]
        ultimos_jogos = pd.concat([ultimos_jogos, ultimos_jogos_temporada])
        ultimos_jogos = ultimos_jogos.reset_index(drop=True)
        ultimos_jogos = ultimos_jogos.sort_values(by='Date', ascending=False)
        ultimos_jogos = ultimos_jogos.head(5)[['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'Referee', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']]
        
        
    return ultimos_jogos




def calcular_resultados(dataframe, equipe):
    resultados = []
    for index, row in dataframe.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        ftr = row['FTR']
        if home_team == equipe:
            if ftr == 'H':
                resultados.append('W')
            elif ftr == 'D':
                resultados.append('D')
            elif ftr == 'A':
                resultados.append('L')
        elif away_team == equipe:
            if ftr == 'A':
                resultados.append('W')
            elif ftr == 'D':
                resultados.append('D')
            elif ftr == 'H':
                resultados.append('L')
        #O mais a direita e o mais recente
    return ''.join(resultados[::-1])






def calculate_h2hString_and_last5GameString_andPoints(df):
    
    for index, row in df.iterrows():
        
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        date = row['Date']
        
        print(f"{index}: {home_team} vs {away_team}")
        
        #dataframes minis
        df_h2h = _5_ultimos_jogos_entre_equipes(home_team,away_team,date)
        df_homelast5 = _5_ultimos_jogos_uma_team(home_team,date)
        df_awaylast5 = _5_ultimos_jogos_uma_team(away_team,date)
        
        
        
        stringH2Hhome = calcular_resultados(df_h2h, home_team)
        stringH2Haway = calcular_resultados(df_h2h, away_team)
        
        
        string_home = calcular_resultados(df_homelast5, home_team)
        string_away = calcular_resultados(df_awaylast5, away_team)
        
        
        pontosCasah2h = calcular_pontos_team(df_h2h,home_team)
        pontosAwayh2h = calcular_pontos_team(df_h2h,away_team)
        
        pontosCasaL5 = calcular_pontos_team(df_homelast5,home_team)
        pontosAwayL5 = calcular_pontos_team(df_awaylast5,away_team)
        
    
        
        
        golosMarcadosH2Hcasa=calcular_golosMarcados_equipe(df_h2h,home_team)
        golosMarcadosH2Haway=calcular_golosMarcados_equipe(df_h2h,away_team)
        
        golosSofridosH2Hcasa=calcular_golosSofridos_equipe(df_h2h,home_team)
        golosSofridosH2Haway=calcular_golosSofridos_equipe(df_h2h,away_team)
        
        golosMarcadosL5casa=calcular_golosMarcados_equipe(df_homelast5,home_team)
        golosMarcadosL5away=calcular_golosMarcados_equipe(df_awaylast5,away_team)
        
        golosSofridosL5casa=calcular_golosSofridos_equipe(df_homelast5,home_team)
        golosSofridosL5away=calcular_golosSofridos_equipe(df_awaylast5,away_team)
        
        
        
        
        diffGolosH2Hcasa=golosMarcadosH2Hcasa-golosSofridosH2Hcasa
        diffGolosH2Haway=golosMarcadosH2Haway-golosSofridosH2Haway
        
        diffGolosL5casa=golosMarcadosL5casa-golosSofridosL5casa
        diffGolosL5away=golosMarcadosL5away-golosSofridosL5away
        
        
        
        
        yHomeL5, vHomeL5 = calcular_Amarelos_Vermelhos_equipe(df_homelast5, home_team)
        yAwayL5, vAwayL5 = calcular_Amarelos_Vermelhos_equipe(df_awaylast5, away_team)
        
        
        

        shotsHome, shots_on_targetHome, foulsHome, cornersHome = contar_estatisticas_equipe(df_homelast5,home_team)
        shotsAway, shots_on_targetAway, foulsAway, cornersAway = contar_estatisticas_equipe(df_awaylast5,away_team)
        
        
        
        
        vitoriash2hHOME = calcular_vitorias_equipe(df_h2h, home_team)
        empatesh2hHOME = calcular_empates_equipe(df_h2h, home_team)
        derrotash2hHOME = calcular_derrotas_equipe(df_h2h, home_team)
        
        vitoriasL5HOME = calcular_vitorias_equipe(df_homelast5, home_team)
        empatesL5HOME = calcular_empates_equipe(df_homelast5, home_team)
        derrotasL5HOME = calcular_derrotas_equipe(df_homelast5, home_team)
        
        vitoriash2hAWAY = calcular_vitorias_equipe(df_h2h, away_team)
        empatesh2hAWAY = calcular_empates_equipe(df_h2h, away_team)
        derrotash2hAWAY = calcular_derrotas_equipe(df_h2h, away_team)
        
        vitoriasL5AWAY = calcular_vitorias_equipe(df_awaylast5, away_team)
        empatesL5AWAY = calcular_empates_equipe(df_awaylast5, away_team)
        derrotasL5AWAY = calcular_derrotas_equipe(df_awaylast5, away_team)
        

        # Add columns to the dataframe for points for each team
        df.loc[index, 'form_h2h_Home'] = stringH2Hhome
        df.loc[index, 'form_h2h_Away'] = stringH2Haway
        
        df.loc[index, 'form_L5_Home'] = string_home
        df.loc[index, 'form_L5_Away'] = string_away
        
        
        df.loc[index, 'pontosCasah2h'] = pontosCasah2h
        df.loc[index, 'pontosAwayh2h'] = pontosAwayh2h
        df.loc[index, 'pontosCasaL5'] = pontosCasaL5
        df.loc[index, 'pontosAwayL5'] = pontosAwayL5
        
        df.loc[index, 'golosMarcadosH2Hcasa'] = golosMarcadosH2Hcasa
        df.loc[index, 'golosMarcadosH2Haway'] = golosMarcadosH2Haway
        df.loc[index, 'golosSofridosH2Hcasa'] = golosSofridosH2Hcasa
        df.loc[index, 'golosSofridosH2Haway'] = golosSofridosH2Haway
        
        df.loc[index, 'golosMarcadosL5casa'] = golosMarcadosL5casa
        df.loc[index, 'golosMarcadosL5away'] = golosMarcadosL5away
        df.loc[index, 'golosSofridosL5casa'] = golosSofridosL5casa
        df.loc[index, 'golosSofridosL5away'] = golosSofridosL5away
        
        df.loc[index, 'diffGolosH2Hcasa'] = diffGolosH2Hcasa
        df.loc[index, 'diffGolosH2Haway'] = diffGolosH2Haway
        df.loc[index, 'diffGolosL5casa'] = diffGolosL5casa
        df.loc[index, 'diffGolosL5away'] = diffGolosL5away
        
        df.loc[index, 'yHomeL5'] = yHomeL5
        df.loc[index, 'vHomeL5'] = vHomeL5
        
        df.loc[index, 'yAwayL5'] = yAwayL5
        df.loc[index, 'vAwayL5'] = vAwayL5
        
        
        
        
        
        df.loc[index, 'shotsHome'] = shotsHome
        df.loc[index, 'shots_on_targetHome'] = shots_on_targetHome
        df.loc[index, 'foulsHome'] = foulsHome
        df.loc[index, 'cornersHome'] = cornersHome
        
        
        df.loc[index, 'shotsAway'] = shotsAway
        df.loc[index, 'shots_on_targetAway'] = shots_on_targetAway
        df.loc[index, 'foulsAway'] = foulsAway
        df.loc[index, 'cornersAway'] = cornersAway
        
            
        df.loc[index, 'vitoriash2hHOME'] = vitoriash2hHOME
        df.loc[index, 'empatesh2hHOME'] = empatesh2hHOME
        df.loc[index, 'derrotash2hHOME'] = derrotash2hHOME
        df.loc[index, 'vitoriasL5HOME'] = vitoriasL5HOME
        df.loc[index, 'empatesL5HOME'] = empatesL5HOME
        df.loc[index, 'derrotasL5HOME'] = derrotasL5HOME
        
        
        df.loc[index, 'vitoriash2hAWAY'] = vitoriash2hAWAY
        df.loc[index, 'empatesh2hAWAY'] = empatesh2hAWAY
        df.loc[index, 'derrotash2hAWAY'] = derrotash2hAWAY
        df.loc[index, 'vitoriasL5AWAY'] = vitoriasL5AWAY
        df.loc[index, 'empatesL5AWAY'] = empatesL5AWAY
        df.loc[index, 'derrotasL5AWAY'] = derrotasL5AWAY
        
        
        
        i=5




    return df



def calcular_vitorias_equipe(dataframe, equipe):
    vitorias = 0
    for index, row in dataframe.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        ftr = row['FTR']
        if home_team == equipe and ftr == 'H':
            vitorias += 1
        elif away_team == equipe and ftr == 'A':
            vitorias += 1
    return vitorias

def calcular_empates_equipe(dataframe, equipe):
    empates = 0
    for index, row in dataframe.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        ftr = row['FTR']
        if home_team == equipe and ftr == 'D':
            empates += 1
        elif away_team == equipe and ftr == 'D':
            empates += 1
    return empates

def calcular_derrotas_equipe(dataframe, equipe):
    derrotas = 0
    for index, row in dataframe.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        ftr = row['FTR']
        if home_team == equipe and ftr == 'A':
            derrotas += 1
        elif away_team == equipe and ftr == 'H':
            derrotas += 1
    return derrotas











def calcular_pontos_team(dataframe, equipe):
    pontos = 0
    for index, row in dataframe.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        ftr = row['FTR']
        if home_team == equipe:
            if ftr == 'H':
                pontos += 3
            elif ftr == 'D':
                pontos += 1
        elif away_team == equipe:
            if ftr == 'A':
                pontos += 3
            elif ftr == 'D':
                pontos += 1
    return pontos



def calcular_golosMarcados_equipe(dataframe, equipe):
    golos = 0
    for index, row in dataframe.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        fthg = row['FTHG']
        ftag = row['FTAG']
        if home_team == equipe:
            golos += fthg
        elif away_team == equipe:
            golos += ftag
    return golos

def calcular_golosSofridos_equipe(dataframe, equipe):
    golos_sofridos = 0
    for index, row in dataframe.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        fthg = row['FTHG']
        ftag = row['FTAG']
        if home_team == equipe:
            golos_sofridos += ftag
        elif away_team == equipe:
            golos_sofridos += fthg
    return golos_sofridos




def calcular_Amarelos_Vermelhos_equipe(dataframe, equipe):
    amarelos = 0
    vermelhos = 0
    for index, row in dataframe.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        hy = row['HY']
        ay = row['AY']
        hr = row['HR']
        ar = row['AR']
        if home_team == equipe:
            amarelos += hy
            vermelhos += hr
        elif away_team == equipe:
            amarelos += ay
            vermelhos += ar
    return amarelos, vermelhos


def contar_estatisticas_equipe(dataframe, equipe):
    shots = 0
    shots_on_target = 0
    fouls = 0
    corners = 0
    for index, row in dataframe.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        hs = row['HS']
        as_ = row['AS']
        hst = row['HST']
        ast = row['AST']
        hf = row['HF']
        af = row['AF']
        hc = row['HC']
        ac = row['AC']
        if home_team == equipe:
            shots += hs
            shots_on_target += hst
            fouls += hf
            corners += hc
        elif away_team == equipe:
            shots += as_
            shots_on_target += ast
            fouls += af
            corners += ac
    return shots, shots_on_target, fouls, corners


def converter_data(data_string):
    formatos = ["%d/%m/%y", "%d/%m/%Y"]
    for formato in formatos:
        try:
            data = datetime.strptime(data_string, formato)
            return data
        except ValueError:
            pass
    raise ValueError("Formato de data inválido: " + data_string)


def drop_colunas(df, colunas):
    colunas_existentes = [coluna for coluna in colunas if coluna in df.columns]
    df.drop(colunas_existentes, axis=1, inplace=True)







####        FUNCOES EM CIMA
###         START YOUR ENGINES




"""
# iterar todas as seasons e aplicar as funcoes a cada dataframe
for season in datasetsPARAmanipular.keys():
    print(f"a comecar {season}")
    datasetsPARAmanipular[season] = calculate_goals(datasetsPARAmanipular[season])
    datasetsPARAmanipular[season] = calculate_points(datasetsPARAmanipular[season])
    #posicao funciona mal
    datasetsPARAmanipular[season] = calcula_posicao(datasetsPARAmanipular[season])
    datasetsPARAmanipular[season] = calculate_h2hString_and_last5GameString_andPoints(datasetsPARAmanipular[season])




# Lista para armazenar os dataframes
dataframes = []

# Concatenar os dataframes
for ano in datasetsPARAmanipular:
    dataframe = datasetsPARAmanipular[ano]
    dataframes.append(dataframe)

combined_df = pd.concat(dataframes)



combined_df.to_csv('GIGAset.csv', index=False)

"""


#                                                                ler o dataset e começar DM

dataset_final = pd.read_csv('GIGAset.csv')
dataset_final_pmanipular = dataset_final.copy()




#remover colunas 
colunas_para_dropar = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'PSH',
                       'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA', 'MaxH', 'MaxD',
                       'MaxA', 'AvgH', 'AvgD', 'AvgA', 'B365>2.5', 'B365<2.5', 'P>2.5', 'P<2.5',
                       'Max>2.5', 'Max<2.5', 'Avg>2.5', 'Avg<2.5', 'AHh', 'B365AHH', 'B365AHA',
                       'PAHH', 'PAHA', 'MaxAHH', 'MaxAHA', 'AvgAHH', 'AvgAHA', 'B365CH', 'B365CD',
                       'B365CA', 'BWCH', 'BWCD', 'BWCA', 'IWCH', 'IWCD', 'IWCA', 'PSCH', 'PSCD',
                       'PSCA', 'WHCH', 'WHCD', 'WHCA', 'VCCH', 'VCCD', 'VCCA', 'MaxCH', 'MaxCD',
                       'MaxCA', 'AvgCH', 'AvgCD', 'AvgCA', 'B365C>2.5', 'B365C<2.5', 'PC>2.5',
                       'PC<2.5', 'MaxC>2.5', 'MaxC<2.5', 'AvgC>2.5', 'AvgC<2.5', 'AHCh', 'B365CAHH',
                       'B365CAHA', 'PCAHH', 'PCAHA', 'MaxCAHH', 'MaxCAHA', 'AvgCAHH', 'AvgCAHA',
                       'LBH', 'LBD', 'LBA', 'Bb1X2', 'BbMxH', 'BbAvH', 'BbMxD', 'BbAvD', 'BbMxA',
                       'BbAvA', 'BbOU', 'BbMx>2.5', 'BbAv>2.5', 'BbMx<2.5', 'BbAv<2.5', 'BbAH',
                       'BbAHh', 'BbMxAHH', 'BbAvAHH', 'BbMxAHA', 'BbAvAHA',
                       'GBH', 'GBD', 'GBA', 'SBH', 'SBD', 'SBA', 'SJH', 'SJD', 'SJA', 'BSH', 'BSD', 'BSA',
                       'HTHG', 'HTAG', 'HTR', 'Referee', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']




drop_colunas(dataset_final_pmanipular, colunas_para_dropar)



# Criar a matriz de correlação
#correlation_matrix = dataset_final_pmanipular.corr()

#plt.figure(figsize=(80, 60))
#sns.heatmap(correlation_matrix, annot=True)





# Iterar pelos elementos da matriz de correlação
"""
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        if correlation_matrix.iloc[i, j] > 0.85 or correlation_matrix.iloc[i, j] < -0.85 :
            print(f" {correlation_matrix.columns[i]} e {correlation_matrix.columns[j]}: {correlation_matrix.iloc[i, j]}")

"""





colunas_para_dropar2 = ['pontosCasa', 'pontosVis', 'pontosCasah2h', 'pontosAwayh2h', 'pontosCasaL5', 'pontosAwayL5', 
            'golosSofridosH2Haway', 'golosSofridosH2Hcasa', 'derrotash2hAWAY', 'empatesh2hAWAY', 'vitoriash2hAWAY',
              'diffGolosH2Haway'      ]


print(4)
drop_colunas(dataset_final_pmanipular, colunas_para_dropar2)



print(4)

# Criar a 2a matriz de correlação
#correlation_matrix = dataset_final_pmanipular.corr()

#plt.figure(figsize=(80, 60))
#sns.heatmap(correlation_matrix, annot=True)


    
colunas_para_dropar3 = ['FTHG', 'FTAG','Date' ]

drop_colunas(dataset_final_pmanipular, colunas_para_dropar3)    


#print("\n\ncolunas no dataframe final:\n")
for coluna in dataset_final_pmanipular.columns:
    i=5
    #print(coluna, end=' ')

    
   
    
    





#scatter matrix
#scatter_matrix(dataset_final_pmanipular[['diffGolosL5casa','diffGolosL5away','shots_on_targetHome','shots_on_targetAway','vitoriasL5HOME','vitoriasL5AWAY']], figsize=(30,30))




# Separar o conjunto de dados em features (X) e variável alvo (y)
X = dataset_final_pmanipular.drop('FTR', axis=1)
y = dataset_final_pmanipular['FTR']

# Pré-processamento para lidar com variáveis categóricas usando codificação one-hot
X_encoded = pd.get_dummies(X)

# Dividir os dados em conjunto de treinamento e conjunto de teste
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)





#logisticRegression
"""
# Criar uma instância do modelo de regressão logística
model = LogisticRegression()

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões com base nos recursos do conjunto de teste
y_pred = model.predict(X_test)

# Calcular a acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy*100)
"""

#RandomFOREST


# Criar uma instância do modelo Random Forest
model = RandomForestClassifier()

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões com base nos recursos do conjunto de teste
y_pred = model.predict(X_test)

# Calcular a acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy * 100)



###RANDOM GRID
























   









del df,filename,key,df_sorted,missing_values,ordem_colunas,cols,season
del colunas_para_dropar


"""





COMBINAR OS DATAFRAMES NUM GIGA DATAFRAME

--------------------------------------
# Lista para armazenar os dataframes
dataframes = []

# Concatenar os dataframes
for ano in datasetsPARAmanipular:
    dataframe = datasetsPARAmanipular[ano]
    dataframes.append(dataframe)

combined_df = pd.concat(dataframes)



combined_df.to_csv('seraqueGIGA.csv', index=False)









CHUNKUS
-----------------------------------------
def calculate_pos_REI(df):
    columns = df.columns
    new_df = pd.DataFrame(columns=columns)
    
    for i in range(0, 370, 10):
        df_aux_jornada_i = df[df.index < i+10]
        df_aux_jornada_i = calculate_positions(df_aux_jornada_i)
        df_aux_jornada_i=get_rows_and_delete(df_aux_jornada_i, i, i+11)
       
        new_df=pd.concat([new_df, df_aux_jornada_i], ignore_index=True)  
        
        print(i)
        
    return new_df
  



def processar_dataframe_em_chunks(dataframe):
    chunk_size = 10  # Tamanho do chunk (quantidade de linhas por chunk)
    total_rows = len(dataframe)
    chunks = total_rows // chunk_size  # Calcula o número de chunks

    chunked_dfs = []  # Lista para armazenar os chunks atualizados

    for chunk_index in range(chunks):
        start = chunk_index * chunk_size
        end = min((chunk_index + 1) * chunk_size, total_rows)
        chunk = dataframe.iloc[start:end].copy()  # Seleciona o chunk atual e faz uma cópia

        # Realize as operações que deseja fazer no chunk atual
        # Exemplo: atualizar colunas, calcular valores, etc.
        #chunk['Nova_Coluna'] = "lalalala" + str(chunk_index)
        chunk=calculate_positions(chunk)

        chunked_dfs.append(chunk)  # Adiciona o chunk atual à lista

        print("lala")

    # Concatena os chunks atualizados para obter o DataFrame final
    final_dataframe = pd.concat(chunked_dfs)

    return final_dataframe













-------------------------------------------

### SEGUNDOOOOWWWWWWWW             16/08/17


datasetsPARAmanipular['2018-19']=calculate_goals(datasetsPARAmanipular['2018-19'])
datasetsPARAmanipular['2018-19']=calculate_points(datasetsPARAmanipular['2018-19'])
#posicao funciona mal
datasetsPARAmanipular['2018-19']=calcula_posicao(datasetsPARAmanipular['2018-19'])

start_time = time.time()
datasetsPARAmanipular['2018-19']=calculate_h2hString_and_last5GameString_andPoints(datasetsPARAmanipular['2018-19'])
end_time = time.time()
execution_time = end_time - start_time

print(f"Execution time: {execution_time} seconds")







combined_df = pd.concat([datasetsPARAmanipular['2017-18'], datasetsPARAmanipular['2018-19']])
















#numero_jogos = contar_jogos_entre_equipes_antes_data('31/07/05', 'Man United', 'Chelsea')
#print(f"Número de jogos entre man utd e chelsea antes de 31 12 04: {numero_jogos}")


    
#kd=_5_ultimos_jogos_entre_equipes('Man United','Chelsea','31/07/17')
#resultado_string = calcular_resultados(kd, 'Man United')
#kdd=_5_ultimos_jogos_uma_team('Man United','31/07/17')
#resultado_string2 = calcular_resultados(kdd, 'Man United')



 #novo_df = selecionar_colunas(datasetsPARAmanipular['2017-18'])




mini_h2h=_5_ultimos_jogos_entre_equipes('Man United','Chelsea','31/07/17')
mini_UTD=_5_ultimos_jogos_uma_team('Man United','31/07/17')


pontiis=calcular_pontos_team(mini_h2h,'Man United')
pontiis2=calcular_pontos_team(mini_h2h,'Chelsea')



---------------------------------------------



pontos working


# Define a function to calculate points for each team
def calculate_points(df):
    # Create a dictionary to keep track of points for each team
    team_points = {}

    # Initialize points to 0 for all teams
    for team in set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()):
        team_points[team] = 0

    # Iterate over all rows in the dataframe
    for index, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        home_goals = row['FTHG']
        away_goals = row['FTAG']

        # Determine the result of the match
        if home_goals > away_goals:
            home_result = 'W'
            away_result = 'L'
            home_points = 3
            away_points = 0
        elif home_goals < away_goals:
            home_result = 'L'
            away_result = 'W'
            home_points = 0
            away_points = 3
        else:
            home_result = 'D'
            away_result = 'D'
            home_points = 1
            away_points = 1


        # Add columns to the dataframe for points for each team
        df.loc[index, 'pontosCasa'] = team_points[home_team]
        df.loc[index, 'pontosVis'] = team_points[away_team]
        
        # Add points to the team's total
        team_points[home_team] += home_points
        team_points[away_team] += away_points


    return df
-------------------------------------











numero de jogos entre duas equipas
-------------------------------------
TeamA='Man City'
TeamB='Portsmouth'

for season in datasetsPARAmanipular.keys():
    # Calculate the number of games played between two teams in a dataframe
    num_games = games_played_between(datasetsPARAmanipular[season], TeamA,TeamB)
    
    # Print the number of games played between the two teams
    print(f'{season}: Number of games played between {TeamA} and {TeamB}: {num_games}')



---------------------------------














CENAS DO KAGGGGGGGGGGGGGGGG


# Gets the goals scored agg arranged by teams and matchweek
def get_goals_scored(dataset):
    # Create a dictionary with team names as keys
    teams = {}
    for i in dataset.groupby('HomeTeam').mean().T.columns:
        teams[i] = []
    
    # the value corresponding to keys is a list containing the match location.
    for i in range(len(dataset)):
        HTGS = dataset.iloc[i]['FTHG']
        ATGS = dataset.iloc[i]['FTAG']
        teams[dataset.iloc[i].HomeTeam].append(HTGS)
        teams[dataset.iloc[i].AwayTeam].append(ATGS)
    
    # Create a dataframe for goals scored where rows are teams and cols are matchweek.
    GoalsScored = pd.DataFrame(data=teams, index = [i for i in range(1,39)]).T
    GoalsScored[0] = 0
    # Aggregate to get uptil that point
    for i in range(2,39):
        GoalsScored[i] = GoalsScored[i] + GoalsScored[i-1]
    return GoalsScored



# Gets the goals conceded agg arranged by teams and matchweek
def get_goals_conceded(dataset):
    # Create a dictionary with team names as keys
    teams = {}
    for i in dataset.groupby('HomeTeam').mean().T.columns:
        teams[i] = []
    
    # the value corresponding to keys is a list containing the match location.
    for i in range(len(dataset)):
        ATGC = dataset.iloc[i]['FTHG']
        HTGC = dataset.iloc[i]['FTAG']
        teams[dataset.iloc[i].HomeTeam].append(HTGC)
        teams[dataset.iloc[i].AwayTeam].append(ATGC)
    
    # Create a dataframe for goals scored where rows are teams and cols are matchweek.
    GoalsConceded = pd.DataFrame(data=teams, index = [i for i in range(1,39)]).T
    GoalsConceded[0] = 0
    # Aggregate to get uptil that point
    for i in range(2,39):
        GoalsConceded[i] = GoalsConceded[i] + GoalsConceded[i-1]
    return GoalsConceded

def get_gss(dataset):
    GC = get_goals_conceded(dataset)
    GS = get_goals_scored(dataset)
   
    j = 0
    HTGS = []
    ATGS = []
    HTGC = []
    ATGC = []

    for i in range(380):
        ht = dataset.iloc[i].HomeTeam
        at = dataset.iloc[i].AwayTeam
        HTGS.append(GS.loc[ht][j])
        ATGS.append(GS.loc[at][j])
        HTGC.append(GC.loc[ht][j])
        ATGC.append(GC.loc[at][j])
        
        if ((i + 1)% 10) == 0:
            j = j + 1
        
    dataset['HTGS'] = HTGS
    dataset['ATGS'] = ATGS
    dataset['HTGC'] = HTGC
    dataset['ATGC'] = ATGC
    
    return dataset



blablabla = datasetsPARAmanipular['2000-01']
# Apply to each dataset
get_gss(blablabla)

























#Unir tudo num dataset.

# Create an empty list to store the dataframes
dfs = []

# Loop through the dictionary and append each dataframe to the list
for key in datasets:
    dfs.append(datasets[key])

# Concatenate the dataframes in the list
ALLin = pd.concat(dfs)






------------------------------------------------
Unique teams SET

# get unique team names
unique_teams = np.sort(ALLin['HomeTeam'].unique())[::1]

# print unique team names
print(unique_teams)







-------------------------------------

print o nome das colunas



print("\n\n   COMEZZZZZZZZZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA  \n\n")
for name, df in datasetsPARAmanipular.items():
    print(f" {name}:")
    print(' '.join(df.columns))
    print(f"Numero de colunas: {len(df.columns)}")
    print()  





------------------------------------------

API

url = 'http://api.football-data.org/v4/competitions/2021/matches?season=2019'
headers = { 'X-Auth-Token': '5adb44abffad447ab6ff07b34bef96bd' }



response = requests.get(url, headers=headers)
output_file = 'aux2.json'


if response.status_code == 200:
    matches = response.json()['matches']
    with open(output_file, 'w') as f:
        f.write(response.text)
    print(f'Successfully wrote response to {output_file}')

    #for match in matches:
     #   pprint.pprint(match)
else:
    print('Request failed with status code', response.status_code)
    
    
    
    
    
    
    
    
-------------------------------------
Dados do FM

datasets2 = {}
for filename in glob.glob('C:/Users/DavidGilmour/Downloads/fm/**.csv'):
    df = pd.read_csv(filename)
    key = filename.replace('.csv', '')
    key = key.replace('C:/Users/DavidGilmour/Downloads/fm\\', '')
    key = key.replace('./datasets\\', '')
    datasets2[key] = df



gammmm=datasets2['games']
gammmm = gammmm.loc[(gammmm['competition_id'] == 'GB1') & (gammmm['season'] == 2020)]
-----------------------------------

"""





