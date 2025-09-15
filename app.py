# compute_pcm_capacity.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- user inputs ---
csv_file = "readings.csv"   # your 15-row CSV
flow_L_per_min = 1.66       # given constant
rho_water = 1000.0          # kg/m^3
c_water = 4186.0            # J/kg.K
dt_seconds = 120.0          # 2 minutes between readings

# --- load data ---
df = pd.read_csv(csv_file)
# ensure sorted by time
df = df.sort_values('time_min').reset_index(drop=True)

# compute mass flow (kg/s)
m_dot = rho_water * (flow_L_per_min / 1000.0) / 60.0

# compute T_pcm
df['T_pcm'] = 0.5 * (df['Ta2'] + df['Ta3'])

# Compute Q removed from water per interval using trapezoidal for (Ta1 - Ta4)
dQ = []
for i in range(len(df)-1):
    dT1 = df.loc[i, 'Ta1'] - df.loc[i, 'Ta4']
    dT2 = df.loc[i+1, 'Ta1'] - df.loc[i+1, 'Ta4']
    avg_dT = 0.5 * (dT1 + dT2)
    deltaQ = m_dot * c_water * avg_dT * dt_seconds
    dQ.append(deltaQ)

# cumulative Q array aligned with timestamps (we will assign Q cumulative at each reading)
cumQ = np.concatenate(([0.0], np.cumsum(dQ)))

df['cumQ_J'] = cumQ

# summary
Q_total = cumQ[-1]
T_pcm_initial = df.loc[0, 'T_pcm']
T_pcm_final = df.loc[df.index[-1], 'T_pcm']
dT_pcm = T_pcm_final - T_pcm_initial
C_pcm = Q_total / dT_pcm if abs(dT_pcm) > 1e-9 else np.nan


# Linear fit Q vs T_pcm
slope, intercept, r_value, p_value, std_err = stats.linregress(df['T_pcm'], df['cumQ_J'])

# plots
plt.figure()
plt.plot(df['time_min'], df['cumQ_J'], marker='o')
plt.xlabel('Time (min)')
plt.ylabel('Cumulative heat absorbed Q (J)')
plt.title('Cumulative heat vs time')
plt.grid(True)
plt.savefig('Q_vs_time.png', dpi=200)

plt.figure()
plt.plot(df['T_pcm'], df['cumQ_J'], marker='o')
xline = np.linspace(df['T_pcm'].min(), df['T_pcm'].max(), 50)
plt.plot(xline, slope * xline + intercept, '--', label=f'fit slope={slope:.1f} J/K')
plt.xlabel('Average PCM temperature (°C)')
plt.ylabel('Cumulative heat absorbed Q (J)')
plt.title('Q vs T_pcm (slope = C_pcm)')
plt.legend()
plt.grid(True)
plt.savefig('Q_vs_Tpcm.png', dpi=200)
plt.show()


plt.figure()
plt.plot(df['time_min'], df['Ta1'], marker='o', label='Ta1: water inlet')
plt.plot(df['time_min'], df['Ta4'], marker='o', label='Ta4: water outlet')
plt.plot(df['time_min'], df['Ta2'], marker='o', label='Ta2: PCM near inlet')
plt.plot(df['time_min'], df['Ta3'], marker='o', label='Ta3: PCM near outlet')

plt.xlabel('Time (min)')
plt.ylabel('Temperature (°C)')
plt.title('Temperatures vs Time')
plt.legend()
plt.grid(True)
plt.savefig('Temperatures_vs_time.png', dpi=200)
plt.show()

# --- Effective heat capacity vs PCM temperature ---

# compute dQ/dT between intervals
df['dQ'] = df['cumQ_J'].diff()
df['dT'] = df['T_pcm'].diff()
df['C_eff'] = df['dQ'] / df['dT']   # J/K
df['T_mid'] = (df['T_pcm'] + df['T_pcm'].shift(1)) / 2  # midpoint temp for plotting

# drop the first NaN row
df_C = df.dropna(subset=['C_eff'])

plt.figure()
plt.plot(df_C['T_mid'], df_C['C_eff'], marker='o')
plt.xlabel('Average PCM temperature (°C)')
plt.ylabel('Effective heat capacity C (J/K)')
plt.title('Effective thermal capacity vs PCM temperature')
plt.grid(True)
plt.savefig('C_vs_Tpcm.png', dpi=200)
plt.show()
