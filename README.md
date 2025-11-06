# Crime and Incarceration Analysis

## Project Overview

This project investigates the relationship between crime rates and incarceration levels across U.S. states from 2000 onward, with special focus on New Mexico. The analysis includes per-capita metrics, trend analysis, state comparisons, and an interactive Streamlit dashboard for data exploration.

## Dataset

**Source:** [Kaggle - Crime and Incarceration in the United States](https://www.kaggle.com/datasets/christophercorrea/prisoners-and-crime-in-united-states)

The dataset contains information about crime rates and incarceration levels across U.S. states. The project processes this data to calculate per-capita metrics and analyze trends over time.

## Project Structure

```
project/
├── data/                   # Dataset files (downloaded and cleaned)
├── src/                    # Source code
│   ├── download_data.py    # Download dataset from Kaggle
│   ├── clean_data.py       # Data cleaning and preprocessing
│   ├── analyze.py          # Statistical analysis
│   ├── visualizations.py   # Visualization functions
│   └── app.py              # Streamlit application
├── figures/                # Saved visualizations
├── requirements.txt        # Python dependencies
├── kaggle.json             # Kaggle API credentials (not in git)
├── streamlit_app.py        # Entry point for Streamlit Cloud
└── README.md               # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Kaggle account and API credentials

### Setup Steps

1. **Clone the repository** (or download the project files)

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Kaggle API credentials:**
   
   The `kaggle.json` file should already be in the project root with your credentials:
   ```json
   {
     "username": "your_username",
     "key": "your_api_key"
   }
   ```
   
   Alternatively, you can place your Kaggle credentials at `~/.kaggle/kaggle.json` on Linux/Mac or `C:\Users\<username>\.kaggle\kaggle.json` on Windows.

4. **Download the dataset:**
   ```bash
   python src/download_data.py
   ```
   
   This will download the dataset to the `data/` directory.

5. **Clean and process the data:**
   ```bash
   python src/clean_data.py
   ```
   
   This will:
   - Handle missing values
   - Standardize column names
   - Calculate per-capita metrics
   - Filter data from 2000 onward
   - Save cleaned data to `data/cleaned_data.csv`

6. **Run analysis (optional):**
   ```bash
   python src/analyze.py
   ```
   
   This will generate summary statistics, correlations, and trend analysis.

## Running the Streamlit App

### Local Development

To run the Streamlit app locally:

```bash
streamlit run src/app.py
```

Or use the root entry point:

```bash
streamlit run streamlit_app.py
```

The app will open in your default web browser at `http://localhost:8501`.

### App Features

The Streamlit app includes:

- **Sidebar Filters:**
  - Year range slider (2000 to latest available)
  - Multi-select state selector
  - Metric selector (crime rate, incarceration rate, etc.)

- **Tabs:**
  1. **Overview:** Summary statistics and data table
  2. **Trends Over Time:** Line graphs showing trends by state
  3. **State Comparisons:** Bar charts for state rankings and distributions
  4. **Relationships:** Scatter plots and correlation heatmaps
  5. **New Mexico Focus:** Detailed analysis comparing New Mexico to other states

- **Interactive Visualizations:**
  - All charts are interactive using Plotly
  - Hover for detailed information
  - Zoom and pan capabilities
  - New Mexico is highlighted in red across all visualizations

## Deployment to Streamlit Cloud

### Steps for Deployment

1. **Push your code to GitHub:**
   - Make sure all files are committed
   - Push to your GitHub repository

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository
   - Set the main file path to: `streamlit_app.py`
   - Click "Deploy"

3. **Configure Secrets (if needed):**
   - If you need to use Kaggle API in the cloud, add secrets in Streamlit Cloud settings
   - Go to Settings → Secrets and add:
     ```
     KAGGLE_USERNAME=your_username
     KAGGLE_KEY=your_api_key
     ```

### Important Notes for Deployment

- The `kaggle.json` file is in `.gitignore` and won't be pushed to GitHub
- For Streamlit Cloud, you may need to download the dataset manually and include it in the repository, or use Streamlit Secrets for API credentials
- Ensure `requirements.txt` includes all necessary dependencies
- The app will automatically load cleaned data if available, or attempt to process raw data

## Usage Workflow

1. **Initial Setup (one time):**
   ```bash
   # Download dataset
   python src/download_data.py
   
   # Clean data
   python src/clean_data.py
   ```

2. **Run Analysis (optional):**
   ```bash
   python src/analyze.py
   ```

3. **Launch App:**
   ```bash
   streamlit run streamlit_app.py
   ```

## Key Metrics Calculated

- **Crime Rate per 100,000:** Total crimes per 100,000 population
- **Incarceration Rate per 100,000:** Total incarcerated individuals per 100,000 population
- **Incarceration to Crime Ratio:** Ratio of incarceration rate to crime rate

## Analysis Features

- Descriptive statistics by state and year
- Correlation analysis between crime and incarceration
- Trend analysis over time (2000+)
- New Mexico-specific comparisons
- State rankings and distributions

## Team Members

- Team size: 5 members
- Roles: (To be assigned)

## Technologies Used

- **Python 3.8+**
- **Pandas:** Data manipulation and analysis
- **NumPy:** Numerical computations
- **Streamlit:** Interactive web application
- **Plotly:** Interactive visualizations
- **Matplotlib/Seaborn:** Additional plotting capabilities
- **Scipy:** Statistical analysis
- **Kaggle API:** Dataset download

## Troubleshooting

### Common Issues

1. **Kaggle API Authentication Error:**
   - Ensure `kaggle.json` is in the project root or in `~/.kaggle/`
   - Verify your API credentials are correct

2. **Data Not Found:**
   - Run `download_data.py` first to download the dataset
   - Run `clean_data.py` to process the raw data

3. **Module Not Found:**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check that you're running scripts from the correct directory

4. **Streamlit App Not Loading Data:**
   - Verify `data/cleaned_data.csv` exists
   - Check that the data files are not in `.gitignore` if deploying

## License

This project is for educational purposes as part of ENG 220 - Engineering in Society course.

## Contact

For questions or issues, please contact the project team.

