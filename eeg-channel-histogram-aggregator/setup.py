from setuptools import setup, find_packages

setup(
    name='eeg-channel-histogram-aggregator',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A project to aggregate histograms from multiple EEG channels.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'pandas',
        'mne'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)