import matplotlib.pyplot as plt

# Data points
modalities = ['EEG', 'fNIRS', 'Hybrid EEG-fNIRS']
spatial_resolution = [1, 10, 10]  # Arbitrary values for illustration
temporal_resolution = [10, 1, 10] # Arbitrary values for illustration

# Create the plot
plt.figure(figsize=(8, 6))
plt.scatter(spatial_resolution, temporal_resolution, s=100)

# Annotate points
for i, modality in enumerate(modalities):
    plt.annotate(modality, (spatial_resolution[i], temporal_resolution[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=14)

# Set axis labels and title
plt.xlabel('Spatial', fontsize=20)
plt.ylabel('Temporal', fontsize=20)
plt.title('Spatial vs. Temporal Resolution: EEG-fNIRS', fontsize=16)

# Set axis limits for better visualization
plt.xlim(0, 11)
plt.ylim(0, 11)

# Show the plot
plt.grid(True)
plt.show()