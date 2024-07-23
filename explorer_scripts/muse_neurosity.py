import pandas as pd
import json
import mne
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from scipy import stats

def compute_confidence_interval(data, confidence=0.95):
    """Compute the confidence interval for the data."""
    n = data.shape[0]
    m = np.mean(data, axis=0)
    se = stats.sem(data, axis=0)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return m - h, m + h

def process_dataset(json_data, csv_data, sfreq):
    """Process dataset to create MNE Raw object and extract events."""
    # Ensure 'unixTimestamp' is present in CSV data
    if 'unixTimestamp' not in csv_data.columns:
        raise ValueError("CSV data must contain 'unixTimestamp' column.")
    
    # Extract timestamps from JSON data
    json_timestamps = [trial['unixTimestamp'] for trial in json_data['trials'] if 'unixTimestamp' in trial]

    # Extract timestamps from CSV data
    csv_timestamps = csv_data['unixTimestamp'].tolist()

    # Define CSV timestamps range
    csv_timestamps_range = (min(csv_timestamps), max(csv_timestamps))

    # Filter JSON events to only those within the range of CSV timestamps
    filtered_json_events = [trial for trial in json_data['trials'] if 'unixTimestamp' in trial and 
                            csv_timestamps_range[0] <= trial['unixTimestamp'] <= csv_timestamps_range[1]]

    # Create MNE events array from filtered JSON events
    event_id = {'standard': 1, 'deviant': 2}
    events = []
    start_time = csv_timestamps[0] / 1e3  # Convert to seconds
    for trial in filtered_json_events:
        if 'stimulus' in trial:
            if '1024hz.mp3' in trial['stimulus']:
                event_type = event_id['standard']
            elif '1920hz.mp3' in trial['stimulus']:
                event_type = event_id['deviant']
            else:
                continue
            event_time = trial['unixTimestamp'] / 1e3  # Convert to seconds
            event_sample = int((event_time - start_time) * sfreq)
            events.append([event_sample, 0, event_type])

    events = np.array(events)
    
    if len(events) == 0:
        raise ValueError("No valid events found for creating epochs.")
    
    # Extract channel names based on CSV columns
    if 'TP9' in csv_data.columns and 'AF7' in csv_data.columns:  # Assume it's Muse data based on column names
        possible_channel_names = ['TP9', 'AF7', 'AF8', 'TP10']
        channel_names = [col for col in csv_data.columns if col in possible_channel_names]
    else:  # Assume it's Neurosity data
        possible_channel_names = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']
        channel_names = [col for col in csv_data.columns if col in possible_channel_names]
    
    if len(channel_names) == 0:
        raise ValueError("No channel data found in the CSV file.")
    
    # Create MNE Raw object
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=['eeg'] * len(channel_names))
    raw = mne.io.RawArray(csv_data[channel_names].T / 1e6, info)

    # Add standard electrode locations (montage)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)

    return raw, events, event_id

def plot_erp(epochs, datadir, name_prefix):
    """Plot and save ERP for each channel with confidence intervals."""
    for ch_name in epochs.ch_names:
        # Compute the average ERP for each condition
        evoked_standard = epochs['standard'].average()
        evoked_deviant = epochs['deviant'].average()

        # Extract data for the channel
        data_standard = epochs['standard'].get_data()[:, epochs['standard'].info['ch_names'].index(ch_name), :]
        data_deviant = epochs['deviant'].get_data()[:, epochs['deviant'].info['ch_names'].index(ch_name), :]

        # Compute confidence intervals
        ci_standard = compute_confidence_interval(data_standard)
        ci_deviant = compute_confidence_interval(data_deviant)

        # Save ERP data to CSV files
        for evoked, label in zip([evoked_standard, evoked_deviant], ['standard', 'deviant']):
            erp_file_path = os.path.join(datadir, f'{name_prefix}_erp_{label}_waveform_{ch_name}.csv')
            erp_df = pd.DataFrame(evoked.data[evoked.ch_names.index(ch_name), :], index=evoked.times, columns=[ch_name])
            erp_df.to_csv(erp_file_path)
            print(f'{name_prefix} ERP waveform data for {ch_name} saved to {erp_file_path}')

        # Plot ERP data with confidence intervals
        plt.figure(figsize=(12, 6))
        plt.plot(evoked_standard.times, evoked_standard.data[evoked_standard.ch_names.index(ch_name), :], label='Standard', color='blue')
        plt.plot(evoked_deviant.times, evoked_deviant.data[evoked_deviant.ch_names.index(ch_name), :], label='Deviant', color='red')

        # Plot confidence intervals
        plt.fill_between(evoked_standard.times, ci_standard[0], ci_standard[1], color='blue', alpha=0.2)
        plt.fill_between(evoked_deviant.times, ci_deviant[0], ci_deviant[1], color='red', alpha=0.2)

        plt.title(f'{name_prefix} ERP: Standard vs Deviant for Channel {ch_name}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (μV)')
        plt.legend()
        plt.grid()

        # Save the plot to a file
        plot_file_path = os.path.join(datadir, f'{name_prefix}_erp_standard_vs_deviant_{ch_name}.png')
        plt.savefig(plot_file_path)
        plt.close()
        print(f'{name_prefix} ERP plot for {ch_name} saved to {plot_file_path}')

def simple_aob_ana(datadir, lfreq, ufreq):
    """Main function for the auditory oddball P300 analysis."""
    print("Starting the analysis...")

    # Load JSON and CSV data
    json_path = os.path.join(datadir, "Auditory_Oddball_P300.json")
    neurosity_csv_path = os.path.join(datadir, "fusionDataExport_1721055495_neurosity/neurosity_rawBrainwaves_1721055495.csv")
    muse_csv_path = os.path.join(datadir, "muse_rawBrainwaves_1721055496825.csv")
    
    # Check if the files exist
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    if not os.path.isfile(neurosity_csv_path):
        raise FileNotFoundError(f"Neurosity CSV file not found: {neurosity_csv_path}")
    if not os.path.isfile(muse_csv_path):
        raise FileNotFoundError(f"Muse CSV file not found: {muse_csv_path}")

    # Load JSON data
    with open(json_path, 'r') as file:
        json_data = json.load(file)

    # Load CSV data
    neurosity_csv_data = pd.read_csv(neurosity_csv_path)
    muse_csv_data = pd.read_csv(muse_csv_path)

    # Define sampling frequency
    sfreq = 256  # Sampling frequency for both Neurosity and Muse

    # Process Neurosity data
    raw_neurosity, events_neurosity, event_id_neurosity = process_dataset(json_data, neurosity_csv_data, sfreq)
    # Filter the data
    raw_neurosity.filter(lfreq, ufreq, fir_design='firwin')
    # Create epochs
    epochs_neurosity = mne.Epochs(raw_neurosity, events_neurosity, event_id_neurosity, tmin=-0.2, tmax=0.8, baseline=(None, 0), preload=True)
    # Plot and save ERP for Neurosity data
    plot_erp(epochs_neurosity, datadir, 'neurosity')

    # Process Muse data
    raw_muse, events_muse, event_id_muse = process_dataset(json_data, muse_csv_data, sfreq)
    # Filter the data
    raw_muse.filter(lfreq, ufreq, fir_design='firwin')
    # Create epochs
    epochs_muse = mne.Epochs(raw_muse, events_muse, event_id_muse, tmin=-0.2, tmax=0.8, baseline=(None, 0), preload=True)
    # Plot and save ERP for Muse data
    plot_erp(epochs_muse, datadir, 'muse')

    print("Analysis completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform ERP analysis for Neurosity and Muse data.")
    parser.add_argument('datadir', type=str, help="Directory containing the data files.")
    parser.add_argument('lfreq', type=float, help="Lower frequency bound for band-pass filter.")
    parser.add_argument('ufreq', type=float, help="Upper frequency bound for band-pass filter.")

    args = parser.parse_args()

    simple_aob_ana(args.datadir, args.lfreq, args.ufreq)
