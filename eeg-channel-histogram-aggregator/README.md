# EEG Channel Histogram Aggregator

This project is designed to efficiently aggregate and visualize histograms from multiple EEG channels. It provides a streamlined workflow for loading EEG data, selecting specific channels, combining their histograms, and plotting the results.

## Project Structure

```
eeg-channel-histogram-aggregator
├── src
│   ├── __init__.py
│   ├── main.py
│   ├── data_loader.py
│   ├── histogram_aggregator.py
│   ├── channel_selector.py
│   └── plotter.py
├── config
│   ├── __init__.py
│   └── settings.py
├── utils
│   ├── __init__.py
│   └── performance_utils.py
├── tests
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_histogram_aggregator.py
│   └── test_channel_selector.py
├── data
│   ├── input
│   │   └── .gitkeep
│   └── output
│       └── .gitkeep
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

1. **Load EEG Data**: Use the `data_loader.py` module to load EEG data from specified input sources.
2. **Select Channels**: Utilize the `channel_selector.py` to filter and select the desired EEG channels.
3. **Aggregate Histograms**: The `histogram_aggregator.py` module combines histograms from the selected channels, ensuring efficient performance.
4. **Plot Results**: Finally, use the `plotter.py` to visualize the combined histogram of the selected channels.

## Example

To run the application, execute the `main.py` file:

```
python src/main.py
```

This will orchestrate the entire process from loading data to displaying the histogram.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.