import pandas as pd
import numpy as np
import zipfile

# Function to clean CSV data and remove extra spaces
def clean_csv_data(file_path):
    try:
        data = pd.read_csv(file_path)
        data.columns = data.columns.str.strip()
        data = data.map(lambda x: x.strip() if isinstance(x, str) else x)
        return data
    except FileNotFoundError:
        return None

# Kalman filter function adjusted for movement counting
def kalman_filter_adjusted(data, R=0.01, Q=0.001):
    n = len(data)
    xhat = np.zeros(n)  # Estimated states
    P = np.zeros(n)  # Estimated error covariance
    xhat[0] = data[0]
    P[0] = 1.0

    for k in range(1, n):
        # Time update (prediction)
        xhat[k] = xhat[k-1]
        P[k] = P[k-1] + Q
        
        # Measurement update (correction)
        K = P[k] / (P[k] + R)  # Kalman gain
        xhat[k] = xhat[k] + K * (data[k] - xhat[k])
        P[k] = (1 - K) * P[k]

    return xhat

# Function to calculate volume based on bounding box
def calculate_bounding_box_correct(data, columns):
    bounding_box = {}
    for col in columns:
        bounding_box[col] = data[col].max() - data[col].min()
    return bounding_box

# Function to calculate volume of bounding box
def calculate_volume(bounding_box):
    return np.prod(list(bounding_box.values()))

# Function to calculate total distance traveled by hand
def calculate_distance_traveled(data, columns):
    return data[columns].diff().abs().sum().sum()

#def calculate_velocity_magnitude(velocity_x, velocity_y, velocity_z):
    return np.sqrt(velocity_x**2 + velocity_y**2 + velocity_z**2)

# # Function to calculate variance of movements
# def calculate_variance(hand_data, hand_columns):
#     variance_x = hand_data[hand_columns[0]].var()
#     variance_y = hand_data[hand_columns[1]].var()
#     variance_z = hand_data[hand_columns[2]].var()

#     # Average variance across x, y, z axes
#     total_variance = (variance_x + variance_y + variance_z) / 3
#     return total_variance

# # Function to calculate the kinetic energy given the velocity components
#     velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2 + velocity_z**2)
#     return 0.5 * mass * velocity_magnitude**2

# # Function to calculate variance for the movement components
# def calculate_variance(velocity_x, velocity_y, velocity_z):
    return (np.var(velocity_x) + np.var(velocity_y) + np.var(velocity_z)) / 3

def calculate_total_distance_with_frame_jump(filtered_hand_x, filtered_hand_y, filtered_hand_z, frame_jump=1):
    total_distance = 0
    num_points = len(filtered_hand_x)
    
    # Iteramos saltando cada 'frame_jump' frames
    for i in range(0, num_points - frame_jump, frame_jump):
        d_segment = np.sqrt((filtered_hand_x[i + frame_jump] - filtered_hand_x[i])**2 +
                            (filtered_hand_y[i + frame_jump] - filtered_hand_y[i])**2 +
                            (filtered_hand_z[i + frame_jump] - filtered_hand_z[i])**2)
        total_distance += d_segment
    
    return total_distance

def calculate_modified_efficiency_with_frame_jump(filtered_hand_x, filtered_hand_y, filtered_hand_z, frame_jump=1, lambda_factor=1.0):
    total_distance = calculate_total_distance_with_frame_jump(filtered_hand_x, filtered_hand_y, filtered_hand_z, frame_jump)
    
    # Distancia recta entre el primer y último punto
    straight_distance = np.sqrt((filtered_hand_x[-1] - filtered_hand_x[0])**2 +
                                (filtered_hand_y[-1] - filtered_hand_y[0])**2 +
                                (filtered_hand_z[-1] - filtered_hand_z[0])**2)

    # Inicializamos penalizaciones
    angle_sum = 0
    redundant_movements = 0
    num_segments = len(filtered_hand_x) - frame_jump
    
    # Penalizamos la complejidad de la trayectoria (cálculo de ángulos y redundancias)
    for i in range(1, num_segments - 1, frame_jump):
        v1 = np.array([filtered_hand_x[i] - filtered_hand_x[i-frame_jump], 
                       filtered_hand_y[i] - filtered_hand_y[i-frame_jump], 
                       filtered_hand_z[i] - filtered_hand_z[i-frame_jump]])
        v2 = np.array([filtered_hand_x[i+frame_jump] - filtered_hand_x[i], 
                       filtered_hand_y[i+frame_jump] - filtered_hand_y[i], 
                       filtered_hand_z[i+frame_jump] - filtered_hand_z[i]])
        
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle = np.arccos(cos_theta)
            angle_sum += angle

        # Penalizamos movimientos redundantes si la distancia es muy pequeña
        if np.linalg.norm(v1) < 1e-2:
            redundant_movements += 1

    # Eficiencia final con penalización de ángulos y redundancias
    modified_efficiency = (straight_distance / total_distance) * (1 / (1 + angle_sum + lambda_factor * redundant_movements))
    
    return modified_efficiency

# Function to calculate the difficulty index with movements normalized by time
def compute_difficulty_index_with_normalized_components(user_number, alpha=1.0, beta=1.0, gamma=0.0):
    environment_labels = {"Unadapted": "BigEnvironment", "Adapted": "VR-Commerce-Accesible"}
    difficulty_results = []
    
    for environment, label in environment_labels.items():
        path_onedrive = r'D:\OneDrive - Universidad de Castilla-La Mancha\PROYECTO_VR_SHOPPIING\IEEE_ACCESS_INTERACCIONES_ADAPTADAS_HNPT\HNPT\extracted_data'
        base_path = f'{path_onedrive}\HNPT_04_07_2024_U{user_number}\{environment}'
        print(base_path)
        hand_data_file = f'{base_path}\HeadHandsData{label}.csv'
        
        hand_data = clean_csv_data(hand_data_file)
        if hand_data is None:
            continue
        
        handR_columns = ['HandR_x', 'HandR_y', 'HandR_z']
        handL_columns = ['HandL_x', 'HandL_y', 'HandL_z']
        handR_columns_vel = ['Velocity_HandR_x', 'Velocity_HandR_y', 'Velocity_HandR_z']
        handL_columns_vel = ['Velocity_HandL_x', 'Velocity_HandL_y', 'Velocity_HandL_z']

        handR_movement_data = hand_data[handR_columns]
        handL_movement_data = hand_data[handL_columns]
        handR_movement_data_vel = hand_data[handR_columns_vel]
        handL_movement_data_vel = hand_data[handL_columns_vel]

        # Apply Kalman filter to each axis of the hands with adjusted parameters
        filtered_handR_x = kalman_filter_adjusted(handR_movement_data['HandR_x'].values)
        filtered_handR_y = kalman_filter_adjusted(handR_movement_data['HandR_y'].values)
        filtered_handR_z = kalman_filter_adjusted(handR_movement_data['HandR_z'].values)

        filtered_handL_x = kalman_filter_adjusted(handL_movement_data['HandL_x'].values)
        filtered_handL_y = kalman_filter_adjusted(handL_movement_data['HandL_y'].values)
        filtered_handL_z = kalman_filter_adjusted(handL_movement_data['HandL_z'].values)

        filtered_handR_vel_x = kalman_filter_adjusted(handR_movement_data_vel['Velocity_HandR_x'].values)
        filtered_handR_vel_y = kalman_filter_adjusted(handR_movement_data_vel['Velocity_HandR_y'].values)
        filtered_handR_vel_z = kalman_filter_adjusted(handR_movement_data_vel['Velocity_HandR_z'].values)

        filtered_handL_vel_x = kalman_filter_adjusted(handL_movement_data_vel['Velocity_HandL_x'].values)
        filtered_handL_vel_y = kalman_filter_adjusted(handL_movement_data_vel['Velocity_HandL_y'].values)
        filtered_handL_vel_z = kalman_filter_adjusted(handL_movement_data_vel['Velocity_HandL_z'].values)

        # Calculate total distance using Kalman-filtered data
        distance_handR = np.sum(np.sqrt(np.diff(filtered_handR_x)**2 + np.diff(filtered_handR_y)**2 + np.diff(filtered_handR_z)**2))
        distance_handL = np.sum(np.sqrt(np.diff(filtered_handL_x)**2 + np.diff(filtered_handL_y)**2 + np.diff(filtered_handL_z)**2))

        total_distance = distance_handR + distance_handL

        # Calculate bounding box volume (still using unfiltered data)
        bounding_box_handR = calculate_bounding_box_correct(handR_movement_data, handR_columns)
        volume_handR = calculate_volume(bounding_box_handR)

        bounding_box_handL = calculate_bounding_box_correct(handL_movement_data, handL_columns)
        volume_handL = calculate_volume(bounding_box_handL)

        total_volume = volume_handR + volume_handL

        # Calculate movements using Kalman filter (for fatigue calculation)
        num_frames = len(handR_movement_data)
        initial_index = int(num_frames * 0.5)  # 60% for initial movements

        handR_movements_initial = np.sum(np.sqrt(np.diff(filtered_handR_x[:initial_index])**2 + np.diff(filtered_handR_y[:initial_index])**2 + np.diff(filtered_handR_z[:initial_index])**2) > 1e-2)
        handL_movements_initial = np.sum(np.sqrt(np.diff(filtered_handL_x[:initial_index])**2 + np.diff(filtered_handL_y[:initial_index])**2 + np.diff(filtered_handL_z[:initial_index])**2) > 1e-2)

        handR_movements_final = np.sum(np.sqrt(np.diff(filtered_handR_x[initial_index:])**2 + np.diff(filtered_handR_y[initial_index:])**2 + np.diff(filtered_handR_z[initial_index:])**2) > 1e-2)
        handL_movements_final = np.sum(np.sqrt(np.diff(filtered_handL_x[initial_index:])**2 + np.diff(filtered_handL_y[initial_index:])**2 + np.diff(filtered_handL_z[initial_index:])**2) > 1e-2)

        total_movements_initial = handR_movements_initial + handL_movements_initial
        total_movements_final = handR_movements_final + handL_movements_final

        total_movements = total_movements_initial + total_movements_final
        total_time = hand_data['Timestamp'].iloc[-1]
        movements_per_second = total_movements / total_time if total_time > 0 else total_movements

        # Calculate fatigue factor based on Kalman-filtered initial and final movements
        #fatigue_factor = 1 + (total_movements_initial - total_movements_final) / total_movements_initial if total_movements_initial > 0 else 1

        # Compute the difficulty index
        # difficulty_index = (1/total_volume) + (alpha * total_distance) + (beta * fatigue_factor) + (gamma * movements_per_second)
        # difficulty_index = (1/total_volume) + (alpha * total_distance) + (gamma * movements_per_second)

        # # Store the result and the components used
        # difficulty_results.append({
        #     'user': user_number,
        #     'environment': environment,
        #     'difficulty_index': difficulty_index,
        #     'total_volume': total_volume,
        #     'total_distance': total_distance,
        #     # 'fatigue_factor': fatigue_factor,
        #     'movements_per_second': movements_per_second
        # })

        # NUEVO: Calcular la velocidad media y la eficiencia
        # velocity_magnitude_handR = calculate_velocity_magnitude(filtered_handR_vel_x, filtered_handR_vel_y, filtered_handR_vel_z)
        # velocity_magnitude_handL = calculate_velocity_magnitude(filtered_handL_vel_x, filtered_handL_vel_y, filtered_handL_vel_z)
        # mean_velocity_handR = np.mean(velocity_magnitude_handR)
        # mean_velocity_handL = np.mean(velocity_magnitude_handL)
        # mean_velocity = (mean_velocity_handR + mean_velocity_handL) / 2 

        # Calcular la eficiencia del movimiento (distancia recta / distancia total)
        straight_distance_handR = np.sqrt((filtered_handR_x[-1] - filtered_handR_x[0])**2 +
                                          (filtered_handR_y[-1] - filtered_handR_y[0])**2 +
                                          (filtered_handR_z[-1] - filtered_handR_z[0])**2)
        straight_distance_handL = np.sqrt((filtered_handL_x[-1] - filtered_handL_x[0])**2 +
                                          (filtered_handL_y[-1] - filtered_handL_y[0])**2 +
                                          (filtered_handL_z[-1] - filtered_handL_z[0])**2)
        # efficiency_handR = straight_distance_handR / distance_handR if distance_handR != 0 else 0
        # efficiency_handL = straight_distance_handL / distance_handL if distance_handL != 0 else 0
        # efficiency = (efficiency_handR + efficiency_handL) / 2  # Eficiencia media
        
        efficiency_R = calculate_modified_efficiency_with_frame_jump(filtered_handR_x, filtered_handR_y, filtered_handR_z, frame_jump=12, lambda_factor=1.0)
        efficiency_L = calculate_modified_efficiency_with_frame_jump(filtered_handL_x, filtered_handL_y, filtered_handL_z, frame_jump=12, lambda_factor=1.0)
        efficiency = efficiency_R + efficiency_L
        print(1/efficiency)

        # NUEVO: Agregar velocidad media y eficiencia al índice de dificultad
        difficulty_index = (1/total_volume) + (alpha * total_distance) + (gamma * movements_per_second) + (1 * (1/efficiency))

        difficulty_results.append({
            'user': user_number,
            'environment': environment,
            'difficulty_index': difficulty_index.round(2),
            'total_volume': total_volume.round(2),
            'total_distance': total_distance.round(2),
            #'mean_velocity': mean_velocity,
            'efficiency': (1/efficiency).round(2),
            'movements_per_second': movements_per_second.round(2)
        })

    return difficulty_results

# Defining a mock hand_mapping variable to simulate user data
hand_mapping = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {}, 8: {}, 9: {}, 10: {}, 11: {}, 12: {}}

# Re-run the calculations after the data extraction
normalized_component_results = []

for user in hand_mapping.keys():
    user_results = compute_difficulty_index_with_normalized_components(user, alpha=1.0, beta=1.0, gamma=1.0)
    normalized_component_results.extend(user_results)

# Step 2: Extract min and max values for each component
max_volume = max(result['total_volume'] for result in normalized_component_results)
max_distance = max(result['total_distance'] for result in normalized_component_results)
#max_fatigue = max(result['fatigue_factor'] for result in normalized_component_results)
max_movements_per_second = max(result['movements_per_second'] for result in normalized_component_results)
#max_velocity = max(result['mean_velocity'] for result in normalized_component_results)
max_efficiency = max(result['efficiency'] for result in normalized_component_results)

# Scale the final difficulty index to be between 0 and 1 by dividing by the maximum possible sum of components
max_sum_components = 1 + 1 + 1 + 1 + 1

for result in normalized_component_results:
    result['normalized_volume'] = (result['total_volume'] / max_volume if result['total_volume'] > 0 else 0).round(2)
    result['normalized_distance'] = (result['total_distance'] / max_distance).round(2)
    # result['normalized_fatigue'] = result['fatigue_factor'] / max_fatigue
    #result['normalized_velocity'] = result['mean_velocity'] / max_velocity if max_velocity > 0 else 0
    result['normalized_efficiency'] = (result['efficiency'] / max_efficiency if max_efficiency > 0 else 0).round(2)
    result['normalized_movements_per_second'] = (result['movements_per_second'] / max_movements_per_second).round(2)
    
    result['normalized_difficulty_index'] = ((
        result['normalized_volume'] + 1 * result['normalized_distance'] + 
        1 * result['normalized_movements_per_second'] + 
        #1 * result['normalized_velocity'] + 
        1 * result['normalized_efficiency']
    ) / max_sum_components).round(2)

normalized_component_df = pd.DataFrame(normalized_component_results)

print(normalized_component_df)