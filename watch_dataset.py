import pandas as pd
import matplotlib.pyplot as plt
import os

check_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'processed_datasets')
csv_list = [f for f in os.listdir(check_path) if os.path.splitext(f)[-1] == '.csv']
print(csv_list)
csv_path = csv_list[0]

df = pd.read_csv(os.path.join(check_path, csv_path))
if 'CSI_Mag_15' in df.columns:
    plt.figure(figsize=(16,5))
    plt.plot(df['CSI_Mag_1'].values)
    plt.show()