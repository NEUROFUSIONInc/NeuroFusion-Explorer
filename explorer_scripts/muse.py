import pandas as pd
import json
import mne
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from scipy import stats

def compute_confidence_interval(data, confidence=0.95):
    """
    Compute the confidence interval for the provided data.

    Parameters:
    - data (np.array): Array of data points.
    - confidence (float): Confidence level (default is 0.95).

    Returns:
    - ci_lower (np.array): Lower bound of the confidence interval.
    - ci_upper (np.array): Upper bound of the confidence interval.
    """
    n = data.shape[0]
    m = np.mean(data, axis=0)
    se = stats.sem(data, axis=0)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return m - h, m + h

def simple_aob_ana(datadir, thr, lfreq, ufreq, do_plot2):
    """
    Analyzes EEG data from an auditory oddball experiment to generate and save ERP (Event-Related Potential) waveforms.
    
    Parameters:
    - datadir (str): Directory containing the data files.
    - thr (float): Threshold value (not used in this script).
    - lfreq (float): Lower frequency bound for band-pass filtering.
    - ufreq (float): Upper frequency bound for band-pass filtering.
    - do_plot2 (bool): Whether to plot ERP waveforms comparison.

    Returns:
    - raw (mne.io.Raw): Raw EEG data.
    - epochs (mne.Epochs): Epochs object for the data.
    - evoked_standard (mne.Evoked): Evoked response object for standard stimuli.
    - evoked_deviant (mne.Evoked): Evoked response object for deviant stimuli.
    """
    print("Starting the analysis...")

    # Load JSON and CSV data
    json_path = os.path.join(datadir, "Auditory_Oddball_-_P300_Event_Related_Potential_1718356068.json")
    csv_path = os.path.join(datadir, "rawBrainwaves_1718355839553.csv")
    
    # Check if the files exist
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Load JSON data
    with open(json_path, 'r') as file:
        json_data = json.load(file)

    # Load CSV data
    csv_data = pd.read_csv(csv_path)

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

    # Verify the overlap of timestamps
    if len(filtered_json_events) == 0:
        raise ValueError("No valid events found after filtering with the CSV timestamps range.")

    # Create MNE events array from filtered JSON events
    event_id = {'standard': 1, 'deviant': 2}
    events = []
    sfreq = 256  # Sampling frequency
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
    
    # Create MNE Raw object
    info = mne.create_info(ch_names=['TP9', 'AF7', 'AF8', 'TP10'], sfreq=sfreq, ch_types=['eeg'] * 4)
    raw = mne.io.RawArray(csv_data[['TP9', 'AF7', 'AF8', 'TP10']].T / 1e6, info)

    # Add standard electrode locations (montage)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)

    # Filter the data
    raw.filter(lfreq, ufreq, fir_design='firwin')

    # Create epochs
    epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.8, baseline=(None, 0), preload=True)
    
    # Compute the average ERP for each condition
    evoked_standard = epochs['standard'].average()
    evoked_deviant = epochs['deviant'].average()

    print("ERP data computed.")

    # Save the ERP data to files
    evoked_standard_file_path = os.path.join(datadir, 'evoked_standard-ave.fif')
    evoked_deviant_file_path = os.path.join(datadir, 'evoked_deviant-ave.fif')
    
    if os.path.exists(evoked_standard_file_path):
        response = input(f"File {evoked_standard_file_path} already exists. Do you want to overwrite it? (yes/no): ")
        if response.lower() != 'yes':
            print("Operation aborted by the user.")
            return
    evoked_standard.save(evoked_standard_file_path, overwrite=True)  # Overwrite the file if it exists
    print(f"Standard ERP response saved to {evoked_standard_file_path}")

    if os.path.exists(evoked_deviant_file_path):
        response = input(f"File {evoked_deviant_file_path} already exists. Do you want to overwrite it? (yes/no): ")
        if response.lower() != 'yes':
            print("Operation aborted by the user.")
            return
    evoked_deviant.save(evoked_deviant_file_path, overwrite=True)  # Overwrite the file if it exists
    print(f"Deviant ERP response saved to {evoked_deviant_file_path}")

    # Save the ERP waveform data to CSV files
    for ch_name in evoked_standard.ch_names:
        erp_standard_file_path = os.path.join(datadir, f'erp_standard_waveform_{ch_name}.csv')
        erp_deviant_file_path = os.path.join(datadir, f'erp_deviant_waveform_{ch_name}.csv')
        
        erp_standard_df = pd.DataFrame(evoked_standard.data.T, columns=evoked_standard.ch_names, index=evoked_standard.times)  # Transpose to have time as rows
        erp_standard_df[[ch_name]].to_csv(erp_standard_file_path)
        print(f"Standard ERP waveform data saved to {erp_standard_file_path}")

        erp_deviant_df = pd.DataFrame(evoked_deviant.data.T, columns=evoked_deviant.ch_names, index=evoked_deviant.times)  # Transpose to have time as rows
        erp_deviant_df[[ch_name]].to_csv(erp_deviant_file_path)
        print(f"Deviant ERP waveform data saved to {erp_deviant_file_path}")

    # Plotting ERP comparisons with confidence intervals
    if do_plot2:
        print("Plotting ERP comparisons with confidence intervals...")
        evokeds = dict(
            standard=evoked_standard,
            deviant=evoked_deviant
        )
        for ch_name in evoked_standard.ch_names:
            picks = [evoked_standard.ch_names.index(ch_name)]  # Pick specific channel
            fig, ax = plt.subplots()
            for condition, evoked in evokeds.items():
                ci_lower, ci_upper = compute_confidence_interval(epochs[condition].get_data()[:, picks[0], :])
                ax.plot(evoked.times, evoked.data[picks[0]], label=f'{condition} Stimulus', color='blue' if condition == 'standard' else 'red')
                ax.fill_between(evoked.times, ci_lower, ci_upper, alpha=0.3, color='blue' if condition == 'standard' else 'red')
            plt.axvline(0, color='black', linestyle='--', label='Stimulus Onset')
            plt.title(f'ERP Comparison for {ch_name}')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude (µV)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(datadir, f'erp_standard_vs_deviant_plot_{ch_name}.png'))
            print(f"ERP comparison plot saved to {os.path.join(datadir, f'erp_standard_vs_deviant_plot_{ch_name}.png')}")
            plt.show()

    print("Analysis completed.")

    # Return raw, epochs, evoked_standard, and evoked_deviant for further processing if needed
    return raw, epochs, evoked_standard, evoked_deviant

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", help="Directory containing the data files")
    parser.add_argument("--thr", type=float, default=0.5, help="Threshold value")
    parser.add_argument("--lfreq", type=float, default=1, help="Lower frequency bound")
    parser.add_argument("--ufreq", type=float, default=40, help="Upper frequency bound")
    parser.add_argument("--doplot2", action="store_true", help="Enable ERP comparison plot")
    
    args = parser.parse_args()

    simple_aob_ana(args.datadir, args.thr, args.lfreq, args.ufreq, args.doplot2)

if __name__ == "__main__":
    main()
