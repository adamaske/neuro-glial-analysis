import matplotlib.pyplot as plt

# Session numbers
sessions = list(range(1, 8))  # 1 to 7

# Times in minutes
times_3km = [23, 22, 22 + 10/60, None, 22, 21 + 35/60, 20]
times_5km = [40, 38 + 40/60, None, None, 38, 37 + 40/60, 35 + 40/60]
times_7_5km = [None, 57 + 45/60, None, 64, None, None, 53]

# Plot
plt.figure(figsize=(12,7))

# Plotting only available data
plt.plot(sessions, times_3km, marker='o', label='3km')
plt.plot(sessions, times_5km, marker='o', label='5km')
plt.plot(sessions, times_7_5km, marker='o', label='7.5km')

plt.title('Running Progress Over Sessions')
plt.xlabel('Session')
plt.ylabel('Time (minutes)')
plt.xticks(sessions)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
