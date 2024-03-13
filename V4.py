# %% [markdown]
# # ***Random Forest***

# %%
import os 
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import mysql.connector
import itertools
import pickle
from sqlalchemy import create_engine

# %% [markdown]
# ##### Se realiza la conexión a la base de datos y asignadola a un cursor para poder mandarla a llamar cada vez que sea requerido

# %%
cnn = mysql.connector.connect(user = 'root', password = 'root', host = 'localhost')
cursor = cnn.cursor()

cursor.execute('USE RanSap_v1')

# %%
engine = create_engine('mysql+mysqlconnector://root:root@localhost/RanSap_V1')

# %% [markdown]
# ### Entropia 
# ##### Aquí se plantea la función para poder medir la impureza. La impureza quiere decir a que cuando se realiza un corte que tan probable es de que una variable sea clasificada de forma incorrecta. Los valores cercanos a cero son menos impuros que aquellos que se acercan al 1.
# $$ E(S) = \sum_{i=1}^c -p_i \log_2 p_i $$
# 

# %%
def entropia(y):
    ''' Dada una Serie Pandas, calcular la Entropia.
        y: variable con la que se calcula la Entropia. '''
    if isinstance(y, pd.Series):
        a = y.value_counts()/y.shape[0]
        entropia = np.sum(-a*np.log2(a+1e-9))

        return(entropia)
    else:
        raise('El Objeto debe ser una serie de Pandas')

# %% [markdown]
# ### Information Gain 
# ##### Esta métrica indica la mejora al hacer diferentes particiones y se suele utilizar con la entropía
# ##### El cálculo del Information Gain dependerá de si se trata de un árbol de decisión de clasificación o de regresión. Habría dos opciones:
# $$ \text{InformationGainClassification} = E(d) - \sum \left( \frac{|s|}{|d|} \right) E(s) $$
# $$ \text{InformationGainRegresion} = \text{Variance}(d) - \sum \left( \frac{|s|}{|d|} \right) \text{Variance}(s) $$
# 

# %%
def varianza(y):
    ''' Función para ayudar a calcular la varianza evitando nan.
        y: variable para calcular la varianza. Debería ser una serie Pandas.'''
    
    if y.dtype != 'object':  # Asegurarse de que los datos son numéricos
        if(len(y) == 1):
            return 0
        else:
            return y.var()
    else:
        return 0  # Devolver 0 si los datos no son numéricos
    

# %%
def information_gain(y, mascara, func=entropia):
    ''' Devuelve la ganancia de información de una variable dada una función de pérdida.
        y: variable objetivo.
        máscara: elección dividida.
        func: función que se utilizará para calcular la ganancia de información en el caso de la clasificación.
    '''

    a = sum(mascara)
    b = mascara.shape[0] - a

    if(a == 0 or b == 0):
        ig = 0
    else: 
        if y.dtypes != 'o':
            ig = varianza(y) - (a/(a+b) * varianza(y[mascara])) - (b/(a+b)*varianza(y[-mascara]))
        else:
            ig = func(y) - a/(a+b)*func(y[mascara]) - b/(a+b)*func(y[-mascara])
    return ig

# %% [markdown]
# ### Mejor Split
# ##### Se calculan todos los posibles valores que se toman de la variable base. Una vez obtenida los resultados se calcula el information gain para cada variable

# %%
def opciones_categoricas(a):
    '''
        Crea todas las combinaciones posibles a partir de una Serie Pandas.
        a: Serie Pandas de donde conseguir todas las combinaciones posibles.
    '''

    a = a.unique()
    opciones = []

    for L in range(0, len(a)+1):
        for subset in itertools.combinations(a, L):
            subset = list(subset)
            opciones.append(subset)
            
    return opciones[1:-1]

# %%
def max_information_gain_split(x, y, func=entropia):
    '''
        Dada una variable predictora y objetivo, devuelve la mejor división, el error y el tipo de variable en función de una función de costo seleccionada.
        x: variable predictora como Serie Pandas.
        y: variable objetivo como Serie Pandas.
        func: función que se utilizará para calcular el mejor split.
    '''

    split_value = []
    ig = []

    variable_numerica = True if x.dtypes != 'o' else False

    # Crear opciones de acuerdo al tipo de variable
    if variable_numerica:
        options = x.sort_values().unique()[1:]
    else: 
        options = opciones_categoricas(x)
    
    # Calcular ig para todos los valores
    for val in options:
        mascara = x < val if variable_numerica else x.isin(val)
        val_ig = information_gain(y, mascara, func)

        #Resultados
        ig.append(val_ig)
        split_value.append(val)
    
    # Checar si estos son mas de 1 resultado y si no, retornar falso
        
    if len(ig) == 0:
        return(None, None, None, False)

    else:
        # Obtener los resultados con mayor IG
        best_ig = max(ig)
        best_ig_index = ig.index(best_ig)
        best_split = split_value[best_ig_index]
        return(best_ig, best_split, variable_numerica, True)
    


# %%
def mejor_split(y, data):
    '''
        Dados unos datos, seleccione la mejor división y devuelva la variable, el valor, el tipo de variable y la ganancia de información.
        y: nombre de la variable de destino
        datos: marco de datos donde encontrar la mejor división.
    '''
    mascaras = data.drop(y, axis=1).apply(max_information_gain_split, y = data[y])
    if sum(mascaras.loc[3, :]) == 0:
        return(None, None, None, None)
    
    else: 
        #Mostrar solo las mascaras que pueden ser divididas
        mascaras = mascaras.loc[:, mascaras.loc[3, :]]

        #Obtener los resultados para dividir con el mayor IG
        split_variable = mascaras.iloc[0].astype(np.float32).idxmax()
        split_value = mascaras[split_variable][1]
        split_ig = mascaras[split_variable][0]
        split_num = mascaras[split_variable][2]

        return(split_variable, split_value, split_ig, split_num)

# %%
def make_split(variable, value, data, is_numeric):
    '''
        Dados unos datos y unas condiciones de división, realice la división.
        variable: variable con la que se realiza el split.
        valor: valor de la variable para realizar la división.
        data: datos que se van a dividir.
        is_numeric: booleano que considera si la variable a dividir es numérica o no.
    '''

    if is_numeric:
        data_1 = data[data[variable] < value]
        data_2 = data[(data[variable] < value) == False]
    else:
        data_1 = data[data[variable].isin(value)]
        data_2 = data[(data[variable].isin(value)) == False]

    return(data_1,data_2)


# %%
def make_prediction(data, target_factor):
    '''
        Dada la variable objetivo, haga una predicción.
        data: serie pandas para la variable objetivo
        target_factor: booleano que considera si la variable es un factor o no
    '''

    # Hacer prediciones
    if target_factor:
        pred = data.value_counts().idxmax()
    else:
        pred = data.mean()

    return pred

# %%
def train_tree(data,y, target_factor, max_depth = None,min_samples_split = None, min_information_gain = 1e-20, counter=0, max_categories = 20):
  if counter==0:
    types = data.dtypes
    check_columns = types[types == "object"].index

    for column in check_columns:
      var_length = len(data[column].value_counts()) 
      if var_length > max_categories:
        raise ValueError('The variable ' + column + ' has '+ str(var_length) + ' unique values, which is more than the accepted ones: ' +  str(max_categories))
        
  if max_depth == None:
    depth_cond = True
  else:
    if counter < max_depth:
      depth_cond = True
    else:
      depth_cond = False

  if min_samples_split == None:
      sample_cond = True
  else:
      if data.shape[0] > min_samples_split:
        sample_cond = True
      else:
        sample_cond = False
        
  if depth_cond & sample_cond:

    var,val,ig,var_type = mejor_split(y, data)

    if ig is not None and ig >= min_information_gain:
      counter += 1
      left,right = make_split(var, val, data,var_type)
      
      if var_type:
        question = "{} <= {}".format(var, val)
      else:
        question = "{} in {}".format(var, val)

      subtree = {question: []}
      yes_answer = train_tree(left,y, target_factor, max_depth,min_samples_split,min_information_gain, counter)
      no_answer = train_tree(right,y, target_factor, max_depth,min_samples_split,min_information_gain, counter)

      if yes_answer == no_answer:
        subtree = yes_answer
      else:
        subtree[question].append(yes_answer)
        subtree[question].append(no_answer)
    else:
      pred = make_prediction(data[y],target_factor)
      return pred
  else:
    pred = make_prediction(data[y],target_factor)
    return pred
  return subtree



# %%
'''
# Definir los hiperparámetros que quieres ajustar
max_depth_values = [None, 10, 20, 30]
min_samples_split_values = [2, 5, 10]
min_information_gain_values = [1e-20, 1e-10, 1e-5]
'''
max_depth = 30
min_samples_split = 10
min_information_gain  = 1e-10

# %%
try:
    with open('modelC.pkl', 'rb') as file:
        loaded_model, loaded_counter = pickle.load(file)
    print("Modelo y contador cargados desde 'modelC.pkl'")
    # Ahora puedes usar 'loaded_model' en lugar de 'forest' y 'loaded_counter' en lugar de 'contador'
    forest = loaded_model
    contador = loaded_counter
except FileNotFoundError:
    print("Archivo 'modelC.pkl' no encontrado. Se creará un nuevo modelo y contador.")
    forest = []  # Inicializa un nuevo bosque
    contador = 0  # Inicializa un nuevo contador

# %%
contador

# %%
# Definir el tamaño del lote
batch_size = 25000  # Ajusta este número según la capacidad de la máquina

# Definir el número total de registros que se desea procesar
total_records = 100000  

# Calcular el número total de lotes
total_batches = int(total_records / (2 * batch_size))

totalMuestras = 0


# %%
from sklearn.preprocessing import LabelEncoder

# Crear el codificador
le = LabelEncoder()

# %%

# Procesar los datos en lotes
for i in range(total_batches):
    # Leer un lote de datos benignos
    query_benigno = "SELECT * FROM escritura WHERE `Tipo de Archivo` = 'benigno' ORDER BY RAND() LIMIT %d;" % (batch_size)
    batch_benigno = pd.read_sql_query(query_benigno, engine)

    # Leer un lote de datos de ransomware
    query_ransomware = "SELECT * FROM escritura WHERE `Tipo de Archivo` = 'ransomware' ORDER BY RAND() LIMIT %d;" % (batch_size)
    batch_ransomware = pd.read_sql_query(query_ransomware, engine)

    # Concatenar los lotes
    batch = pd.concat([batch_benigno, batch_ransomware], ignore_index=True)

    batch['Tipo de Archivo'] = le.fit_transform(batch['Tipo de Archivo'])

    # Incrementar el contador por el tamaño del lote
    contador += (2*batch_size)

    # Imprimir el contador
    print("Número de registros por procesar: ", (2*batch_size))

    totalMuestras += contador

    # Entrenar el árbol de decisión en el lote de datos
    decision_tree = train_tree(batch, 'Tipo de Archivo', True, max_depth, min_samples_split, min_information_gain)

    # Imprimir el contador
    print("Número de registros procesados hasta ahora: ", contador)
    
    # Agregar el árbol de decisión al bosque
    forest.append(decision_tree)


# %%
# Guardar el modelo y el contador después de procesar todos los lotes
with open('modelC.pkl', 'wb') as file:
    pickle.dump((forest, contador), file)

print("Modelo y contador guardados en 'modelC.pkl' después de procesar todos los lotes")

# %%
forest

# %%
def add_edges(graph, parent_node, tree_dict, counter):
    for key in tree_dict:
        counter += 1
        child_node = counter
        graph.add_edge(parent_node, child_node)
        graph.nodes[child_node]["label"] = key
        if isinstance(tree_dict[key], list):
            for i, item in enumerate(tree_dict[key]):
                if isinstance(item, dict):
                    graph, counter = add_edges(graph, child_node, item, counter)
                else:
                    counter += 1
                    graph.add_edge(child_node, counter)
                    graph.nodes[counter]["label"] = str(item)
        elif isinstance(tree_dict[key], dict):
            graph, counter = add_edges(graph, child_node, tree_dict[key], counter)
    return graph, counter

def plot_tree(decision_tree):
    G = nx.DiGraph()
    G.add_node(0, label="root")
    G, _ = add_edges(G, 0, decision_tree, 0)
    pos = nx.kamada_kawai_layout(G)  # Cambiamos a la disposición 'kamada_kawai'
    labels = nx.get_node_attributes(G, 'label')
    plt.figure(figsize=(12, 12))  # Aumentamos el tamaño de la figura
    nx.draw(G, pos, labels=labels, with_labels=True, arrows=False)
    plt.show()

# Para cada árbol en el bosque
for i, tree in enumerate(forest):
    print(f"Árbol {i+1}:")
    plot_tree(tree)
    print("\n")  # Agrega una línea en blanco entre cada árbol



# %%
import re

def clasificar_datos(observacion, arbol):
  question = list(arbol.keys())[0]

  # Utilizar una expresión regular para dividir la cadena 'question'
  match = re.match(r'(.*) (<=|>) (.*)', question)
  column_name, operator, value = match.groups()

  # Verificar si 'observacion[column_name]' es un número
  try:
    obs_value = float(observacion[column_name])
    value = float(value)
  except ValueError:
    obs_value = observacion[column_name]
    value = value

  if operator == '<=':
    if obs_value <= value:
      answer = arbol[question][0]
    else:
      answer = arbol[question][1]
  else:
    if obs_value == value:
      answer = arbol[question][0]
    else:
      answer = arbol[question][1]

  # Si la respuesta no es un diccionario
  if not isinstance(answer, dict):
    return answer
  else:
    residual_tree = answer
    return clasificar_datos(observacion, answer)


# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# %%

# Definir el tamaño del lote para cada tipo de datos
muestra_real = 10  # Ajusta este número según la capacidad de la maquina

q_real = "SELECT * FROM escritura ORDER BY RAND() LIMIT %d;" % muestra_real
df_real = pd.read_sql_query(q_real, engine)

'''
# Leer un lote de datos benignos
q_benigno = "SELECT * FROM escritura WHERE `Tipo de Archivo` = 'benigno' ORDER BY RAND() LIMIT %d;" % muestra_real
real_benigno = pd.read_sql_query(q_benigno, engine)

# Leer un lote de datos de ransomware
q_ransomware = "SELECT * FROM escritura WHERE `Tipo de Archivo` = 'ransomware' ORDER BY RAND() LIMIT %d;" % muestra_real
real_ransomware = pd.read_sql_query(q_ransomware, engine)

# Concatenar los lotes
df_real = pd.concat([real_benigno, real_ransomware], ignore_index=True) '''

# %%
#df_real.to_csv('muestra_real.csv', index=False)

# %%
# df_real = pd.read_csv('muestra_real.csv') 

# %%
df_real

# %%
# Obtener las etiquetas verdaderas de los datos
y_true = df_real['Tipo de Archivo'].tolist()

# Crear un objeto LabelEncoder
le = LabelEncoder()

# Ajustar el LabelEncoder a los datos
le.fit(y_true)

y_true_encoded = le.transform(y_true)

predictions = []
for i in range(len(df_real)):
    # Hacer una predicción con cada árbol
    tree_predictions = [clasificar_datos(df_real.iloc[i,:], tree) for tree in forest]
    
    # Usar la votación para obtener la predicción final
    obs_pred = max(set(tree_predictions), key=tree_predictions.count)
    
    predictions.append(obs_pred)

# Transformar las predicciones numéricas a las etiquetas originales
predictions_labels = le.inverse_transform(predictions)

for pred in predictions_labels:
    if pred == 'benigno':
        print("La predicción es benigna.")
    elif pred == 'ransomware':
        print("La predicción es ransomware.")
    else:
        print("Etiqueta desconocida.")

accuracy = accuracy_score(y_true_encoded, predictions)
precision = precision_score(y_true_encoded, predictions)
recall = recall_score(y_true_encoded, predictions)
f1 = f1_score(y_true_encoded, predictions)

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)


# %%
print("Predicciones: ", predictions)
print("Verdaderas: ", y_true_encoded)
print(" ")


