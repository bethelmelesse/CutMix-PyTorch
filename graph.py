import torch
import np

def compute_rms_trend(data):
    rms_trend = torch.sqrt((data**2).mean(dim=1)).cpu().numpy()
    return rms_trend

def plot_rms_trend(normal_data, misalign_data, cavitation_data):
    # Computing RMS trends for all datasets
    normal_rms_trend = compute_rms_trend(normal_data)
    misalign_rms_trend = compute_rms_trend(misalign_data)
    cavitation_rms_trend = compute_rms_trend(cavitation_data)
    time_normal = np.arange(len(normal_rms_trend[0]))  # Generating a time array for normal data
    time_misalign = np.arange(len(misalign_rms_trend[0]))  # Generating a time array for misaligned data
    time_cavitation = np.arange(len(cavitation_rms_tr

def compute_rms_trend(data):
    rms_trend = torch.sqrt((data**2).mean(dim=1)).cpu().numpy()
    return rms_trend


def plot_rms_trend(normal_data, misalign_data, cavitation_data):
    # Computing RMS trends for all datasets
    normal_rms_trend = compute_rms_trend(normal_data)
    misalign_rms_trend = compute_rms_trend(misalign_data)
    cavitation_rms_trend = compute_rms_trend(cavitation_data)

    time_normal = np.arange(len(normal_rms_trend[0]))  # Generating a time array for normal data
    time_misalign = np.arange(len(misalign_rms_trend[0]))  # Generating a time array for misaligned data
    time_cavitation = np.arange(len(cavitation_rms_trend[0]))  # Generating a time array for cavitation data

    for i in range(normal_data.shape[0]):
        plt.figure(figsize=(10, 5))
        plt.plot(time_normal, normal_rms_trend[i], label='Normal', color='blue')
        plt.plot(time_misalign, misalign_rms_trend[i], label='Misalign', color='green')
        plt.plot(time_cavitation, cavitation_rms_trend[i], label='Cavitation', color='red')
        plt.title(f'RMS Trend for Sensor {i + 1}')
        plt.xlabel('Data Points')
        plt.ylabel('RMS Value')
        plt.ylim(0, 4)  # Limiting the y-axis to a specific range for visualization clarity
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

# RMS trends for normal, misaligned, and cavitation datasets
plot_entire_data(normal_data, "Normal Data")
plot_entire_data(misalign_data, "Misalign Data")
plot_entire_data(cavitation_data, "Cavitation Data")

# RMS trends for all datasets
plot_rms_trend(normal_data, misalign_data, cavitation_data)