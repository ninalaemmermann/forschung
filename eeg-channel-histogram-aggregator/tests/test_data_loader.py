from src.data_loader import load_eeg_data
import pytest

def test_load_eeg_data_valid():
    # Test loading valid EEG data
    data = load_eeg_data('data/input/sample_eeg_data.csv')
    assert data is not None
    assert len(data) > 0

def test_load_eeg_data_invalid_file():
    # Test loading data from an invalid file path
    with pytest.raises(FileNotFoundError):
        load_eeg_data('data/input/non_existent_file.csv')

def test_load_eeg_data_empty_file():
    # Test loading data from an empty file
    data = load_eeg_data('data/input/empty_file.csv')
    assert data is not None
    assert len(data) == 0

def test_load_eeg_data_format():
    # Test loading data to ensure it has the correct format
    data = load_eeg_data('data/input/sample_eeg_data.csv')
    assert isinstance(data, list)  # Assuming the data is returned as a list
    assert all(isinstance(channel, dict) for channel in data)  # Each channel should be a dictionary

def test_load_eeg_data_performance():
    # Test the performance of loading large EEG data
    import time
    start_time = time.time()
    data = load_eeg_data('data/input/large_eeg_data.csv')
    duration = time.time() - start_time
    assert duration < 2  # Ensure it loads within 2 seconds for large files
    assert data is not None
    assert len(data) > 0