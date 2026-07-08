import pytest
from src.channel_selector import ChannelSelector

def test_select_channels():
    # Sample EEG data with 10 channels
    eeg_data = {
        'Ch_1': [1, 2, 3],
        'Ch_2': [4, 5, 6],
        'Ch_3': [7, 8, 9],
        'Ch_4': [10, 11, 12],
        'Ch_5': [13, 14, 15],
        'Ch_6': [16, 17, 18],
        'Ch_7': [19, 20, 21],
        'Ch_8': [22, 23, 24],
        'Ch_9': [25, 26, 27],
        'Ch_10': [28, 29, 30]
    }
    
    # Channels to select
    selected_channels = ['Ch_1', 'Ch_3', 'Ch_5', 'Ch_7', 'Ch_9', 'Ch_2', 'Ch_4']
    
    # Create an instance of ChannelSelector
    channel_selector = ChannelSelector()
    
    # Select channels
    filtered_data = channel_selector.select_channels(eeg_data, selected_channels)
    
    # Check if the filtered data contains only the selected channels
    assert set(filtered_data.keys()) == set(selected_channels)
    
    # Check if the data for each selected channel is correct
    for channel in selected_channels:
        assert filtered_data[channel] == eeg_data[channel]