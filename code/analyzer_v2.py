import os
import pandas as pd
import matplotlib.pyplot as plt
import json

# Definir la ruta base para tus archivos
base_path_PC = "PATH TO FILES HERE"

# Función para limpiar nombres de columnas y valores de celdas
def clean_dataframe(df):
    df.columns = df.columns.str.strip()
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    return df

# Función para cargar un archivo JSON en un DataFrame
def load_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    df = pd.DataFrame([data])
    return df

# Inicializar el diccionario principal para los usuarios
users_data = {}

# Recorrer los directorios de usuarios
for user_dir in os.listdir(base_path_PC):
    user_path = os.path.join(base_path_PC, user_dir)
    if os.path.isdir(user_path):
        # Inicializar diccionarios para las carpetas 'Adapted' y 'Unadapted'
        user_dict = {'Adapted': {}, 'Unadapted': {}}
        
        for session_dir in os.listdir(user_path):
            session_path = os.path.join(user_path, session_dir)
            if os.path.isdir(session_path):
                
                for file in os.listdir(session_path):
                    file_path = os.path.join(session_path, file)
                    
                    if 'BigEnvironment' in file:
                        df = pd.read_csv(file_path)
                        df = clean_dataframe(df)
                        user_dict['Unadapted'][file] = df
                    elif 'VR-Commerce-Accesible' in file:
                        df = pd.read_csv(file_path)
                        df = clean_dataframe(df)
                        user_dict['Adapted'][file] = df
                    elif file == 'WristAnglesData.json':
                        wrist_data = load_json(file_path)
                        user_dict['Adapted']['WristAnglesData'] = wrist_data

        # Asignar el diccionario del usuario al diccionario principal
        users_data[user_dir] = user_dict


# Inicializar lista para almacenar datos de completitud de tareas
task_completion_data = []

def efectiveness(environment_name, csv_file_name, task_duration_limits, data_list):
# Extraer los datos de completitud de tareas de cada usuario
  for user_id, data in users_data.items():
    simplified_user_id = 'U' + user_id.split('_')[-1][1:]  # Extraer y ajustar el identificador de usuario
    if csv_file_name in data[environment_name]:
        task_durations = data[environment_name][csv_file_name]['TaskDuration'].values
        for task_num, task_duration in enumerate(task_durations, start=1):
            completed = 'Yes' if task_duration <= task_duration_limits[task_num - 1] else 'No'
            data_list.append([simplified_user_id, f'#{task_num}', task_duration, completed])

  # Convertir los datos de completitud de tareas a un DataFrame
  task_completion_df = pd.DataFrame(data_list, columns=['User', 'Task', 'TaskDuration', 'Completed'])

  # Ordenar usuarios y tareas numéricamente
  task_completion_df['User'] = pd.Categorical(task_completion_df['User'], 
                                              categories=sorted(task_completion_df['User'].unique(), key=lambda x: int(x[1:])),
                                              ordered=True)
  task_completion_df['Task'] = pd.Categorical(task_completion_df['Task'], 
                                              categories=sorted(task_completion_df['Task'].unique(), key=lambda x: int(x[1:])),
                                              ordered=True)

  # Crear la tabla pivote para mostrar el estado de completitud de cada tarea para cada usuario
  pivot_table = task_completion_df.pivot(index='Task', columns='User', values='Completed')

  # Calcular el porcentaje de tareas completadas por cada usuario, redondeado a 1 decimal
  completion_rate_by_user = (pivot_table == 'Yes').mean().round(1) * 100
  pivot_table.loc['Completion Rate (%)'] = completion_rate_by_user

  # Calcular el porcentaje de veces que cada tarea fue completada, redondeado a 1 decimal
  completion_rate_by_task = (pivot_table == 'Yes').mean(axis=1).round(1) * 100
  pivot_table['Comp. Rate (%)'] = completion_rate_by_task  # Abreviar el nombre de la columna

  # Ajustar el tamaño de la figura y la fuente para mejorar la visibilidad
  fig, ax = plt.subplots(figsize=(15, 8))
  ax.axis('tight')
  ax.axis('off')

  # Crear la tabla con un tamaño de fuente mayor
  table = ax.table(cellText=pivot_table.values, 
                  colLabels=pivot_table.columns, 
                  rowLabels=pivot_table.index, 
                  cellLoc='center', 
                  loc='center',
                  colColours=['#f4f4f4']*len(pivot_table.columns),
                  rowColours=['#f4f4f4']*len(pivot_table.index))

  table.auto_set_font_size(False)
  table.set_fontsize(12)  # Aumenta el tamaño de la fuente

  plt.show()

  return task_completion_df