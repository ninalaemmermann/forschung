from src.histogram_aggregator import HistogramAggregator
from src.channel_selector import ChannelSelector
import numpy as np
import pytest

def test_histogram_aggregation():
    # Sample data for 7 EEG channels
    channel_data = {
        'Channel_1': np.random.normal(loc=0, scale=1, size=1000),
        'Channel_2': np.random.normal(loc=1, scale=1, size=1000),
        'Channel_3': np.random.normal(loc=-1, scale=1, size=1000),
        'Channel_4': np.random.normal(loc=0.5, scale=1, size=1000),
        'Channel_5': np.random.normal(loc=-0.5, scale=1, size=1000),
        'Channel_6': np.random.normal(loc=2, scale=1, size=1000),
        'Channel_7': np.random.normal(loc=-2, scale=1, size=1000),
    }

    # Initialize ChannelSelector and HistogramAggregator
    channel_selector = ChannelSelector()
    selected_channels = channel_selector.select_channels(channel_data, ['Channel_1', 'Channel_2', 'Channel_3', 'Channel_4', 'Channel_5', 'Channel_6', 'Channel_7'])

    histogram_aggregator = HistogramAggregator()
    combined_histogram = histogram_aggregator.aggregate_histograms(selected_channels)

    # Check if the combined histogram is not empty
    assert combined_histogram is not None
    assert len(combined_histogram) > 0

    # Check if the histogram is normalized
    total_count = sum(combined_histogram.values())
    for count in combined_histogram.values():
        assert count >= 0
        assert count <= total_count

    # Check if the number of bins is as expected
    expected_bins = histogram_aggregator.bins
    assert len(combined_histogram) == expected_bins

def test_channel_selection():
    # Sample data for channel selection
    channel_data = {
        'Channel_1': np.random.normal(size=1000),
        'Channel_2': np.random.normal(size=1000),
        'Channel_3': np.random.normal(size=1000),
        'Channel_4': np.random.normal(size=1000),
        'Channel_5': np.random.normal(size=1000),
        'Channel_6': np.random.normal(size=1000),
        'Channel_7': np.random.normal(size=1000),
    }

    channel_selector = ChannelSelector()
    selected_channels = channel_selector.select_channels(channel_data, ['Channel_1', 'Channel_3', 'Channel_5'])

    # Check if the selected channels are correct
    assert 'Channel_1' in selected_channels
    assert 'Channel_3' in selected_channels
    assert 'Channel_5' in selected_channels
    assert 'Channel_2' not in selected_channels
    assert 'Channel_4' not in selected_channels
    assert 'Channel_6' not in selected_channels
    assert 'Channel_7' not in selected_channels

if __name__ == "__main__":
    pytest.main()