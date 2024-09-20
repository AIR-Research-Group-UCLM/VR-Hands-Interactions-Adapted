import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to generate 3D scatter plots and show them
def create_3d_scatter_plots_v4(df_adapted, df_unadapted, user_id, y_shift=1.0, unadapted_point_size=20):
    fig = plt.figure(figsize=(12, 6))

    df_unadapted[' HandL_y_shifted'] = df_unadapted[' HandL_y'] + y_shift
    df_unadapted[' HandR_y_shifted'] = df_unadapted[' HandR_y'] + y_shift

    # Left hand plot (subgraph 1)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(df_unadapted[' HandL_x'], df_unadapted[' HandL_z'], df_unadapted[' HandL_y_shifted'], 
                c='r', label='Unadapted', marker='o', alpha=0.7, s=unadapted_point_size)  # Circle markers with Y shift and size control
    ax1.scatter(df_adapted[' HandL_x'], df_adapted[' HandL_z'], df_adapted[' HandL_y'], 
                c='b', label='Adapted', marker='^', alpha=0.5, s=unadapted_point_size)  # Triangle markers without Y shift
    ax1.set_title(f'{user_id} - Left Hand')
    ax1.set_xlabel('Position X')
    ax1.set_zlabel('Position Y')  # Now Z is Y
    ax1.set_ylabel('Position Z')  # Now Y is Z
    ax1.legend()

    # Right hand plot (subgraph 2)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(df_unadapted[' HandR_x'], df_unadapted[' HandR_z'], df_unadapted[' HandR_y_shifted'], 
                c='r', label='Unadapted', marker='o', alpha=0.7, s=unadapted_point_size)  # Circle markers with Y shift and size control
    ax2.scatter(df_adapted[' HandR_x'], df_adapted[' HandR_z'], df_adapted[' HandR_y'], 
                c='b', label='Adapted', marker='^', alpha=0.5, s=unadapted_point_size)  # Triangle markers without Y shift
    ax2.set_title(f'{user_id} - Right Hand')
    ax2.set_xlabel('Position X')
    ax2.set_zlabel('Position Y')  # Now Z is Y
    ax2.set_ylabel('Position Z')  # Now Y is Z
    ax2.legend()

    # Show the figure
    plt.show()

# Example of loading the data and using the function
# Load the CSVs
df_adapted = pd.read_csv(r'D:\OneDrive - Universidad de Castilla-La Mancha\PROYECTO_VR_SHOPPIING\IEEE_ACCESS_INTERACCIONES_ADAPTADAS_HNPT\HNPT\HNPT_04_07_2024_U1\Adapted\HeadHandsDataVR-Commerce-Accesible.csv')
df_unadapted = pd.read_csv(r'D:\OneDrive - Universidad de Castilla-La Mancha\PROYECTO_VR_SHOPPIING\IEEE_ACCESS_INTERACCIONES_ADAPTADAS_HNPT\HNPT\HNPT_04_07_2024_U1\Unadapted\HeadHandsDataBigEnvironment.csv')

create_3d_scatter_plots_v4(df_adapted, df_unadapted, "U1", y_shift = -1.5, unadapted_point_size = 2)

df_adapted = pd.read_csv(r'D:\OneDrive - Universidad de Castilla-La Mancha\PROYECTO_VR_SHOPPIING\IEEE_ACCESS_INTERACCIONES_ADAPTADAS_HNPT\HNPT\HNPT_04_07_2024_U5\Adapted\HeadHandsDataVR-Commerce-Accesible.csv')
df_unadapted = pd.read_csv(r'D:\OneDrive - Universidad de Castilla-La Mancha\PROYECTO_VR_SHOPPIING\IEEE_ACCESS_INTERACCIONES_ADAPTADAS_HNPT\HNPT\HNPT_04_07_2024_U5\Unadapted\HeadHandsDataBigEnvironment.csv')

create_3d_scatter_plots_v4(df_adapted, df_unadapted, "U5", y_shift = -3.0, unadapted_point_size = 2)

df_adapted = pd.read_csv(r'D:\OneDrive - Universidad de Castilla-La Mancha\PROYECTO_VR_SHOPPIING\IEEE_ACCESS_INTERACCIONES_ADAPTADAS_HNPT\HNPT\HNPT_04_07_2024_U8\Adapted\HeadHandsDataVR-Commerce-Accesible.csv')
df_unadapted = pd.read_csv(r'D:\OneDrive - Universidad de Castilla-La Mancha\PROYECTO_VR_SHOPPIING\IEEE_ACCESS_INTERACCIONES_ADAPTADAS_HNPT\HNPT\HNPT_04_07_2024_U8\Unadapted\HeadHandsDataBigEnvironment.csv')

create_3d_scatter_plots_v4(df_adapted, df_unadapted, "U8", y_shift = -2.5, unadapted_point_size = 2)

df_adapted = pd.read_csv(r'D:\OneDrive - Universidad de Castilla-La Mancha\PROYECTO_VR_SHOPPIING\IEEE_ACCESS_INTERACCIONES_ADAPTADAS_HNPT\HNPT\HNPT_04_07_2024_U7\Adapted\HeadHandsDataVR-Commerce-Accesible.csv')
df_unadapted = pd.read_csv(r'D:\OneDrive - Universidad de Castilla-La Mancha\PROYECTO_VR_SHOPPIING\IEEE_ACCESS_INTERACCIONES_ADAPTADAS_HNPT\HNPT\HNPT_04_07_2024_U7\Unadapted\HeadHandsDataBigEnvironment.csv')

create_3d_scatter_plots_v4(df_adapted, df_unadapted, "U7", y_shift = -1.5, unadapted_point_size = 2)