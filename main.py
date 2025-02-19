import matplotlib.pyplot as plt

# Define the data points
methods = ['EEG', 'fNIRS', 'Hybrid EEG-fNIRS']
spatial_resolution = [1, 9, 9]  # Updated positions
temporal_resolution = [9, 1, 9]  # Updated positions

# Create the scatter plot
plt.figure(figsize=(7, 5))
plt.scatter(spatial_resolution, temporal_resolution, color=['blue', 'red', 'purple'], s=50)

# Annotate points
for i, method in enumerate(methods):
    plt.text(spatial_resolution[i], temporal_resolution[i], method, fontsize=12)

# Labels and title
plt.xlabel("Spatial Resolution", fontsize=14)
plt.ylabel("Temporal Resolution", fontsize=14)
plt.title("EEG & fNIRS: Spatial-Temporal Resolution")
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.grid(True)

# Show the plot
plt.show()
