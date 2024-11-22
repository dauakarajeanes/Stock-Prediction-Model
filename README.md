# Google Stock Price Prediction Using Machine Learning and Neural Networks

This project is a **Google stock price prediction tool** designed to forecast future stock prices using a neural network model. The model is trained on historical data from **January 1, 2012**, to **January 1, 2024**, utilizing a **100-day historical period** to predict the stock price for the next day (Day 101). Below is an overview of its features, functionality, and how it works.

---

## Core Features

### **Neural Network-Based Prediction**:
- The tool uses a **pre-trained machine learning model** (`Stock Prediction Model.keras`) built with **TensorFlow** and **Keras**.
- The model predicts the future **Google stock prices** based on 100 days of historical closing prices.

### **Google Stock Data Retrieval**:
- Stock price data is retrieved from **Yahoo Finance** (`yfinance`) for the specified period: **January 1, 2012** to **January 1, 2024**.

### **Data Visualization**:
- **Stock Data Display**: Displays the historical Google stock price data in an interactive format using Streamlit.
- **Moving Averages Analysis**:
  - Plots moving averages (50-day, 100-day, and 200-day) alongside actual stock prices.
  - Provides insights into trends and smoothens price fluctuations over time.
- **Prediction Comparison**:
  - Compares **predicted** Google stock prices with actual historical prices for evaluating the performance of the model.

### **Data Preprocessing**:
- **MinMaxScaler** is used to scale the data to ensure better performance of the neural network.
- Handles **historical data aggregation** and **test data preparation** for precise prediction generation.

### **Streamlit Integration**:
- The project is built as an **interactive web application** using **Streamlit**, providing an intuitive way for users to visualize predictions and trends in real time.
- Outputs **interactive plots** and **tables** for easy analysis and exploration.

---

## How It Works

1. **Historical Data Input**: The model takes 100 consecutive days of **Google stock closing prices** as input.
2. **Neural Network Prediction**: Using the pre-trained neural network, the tool predicts the stock price for the **101st day**.
3. **Visualization**: The results are displayed using interactive graphs, which compare the **predicted prices** with the **actual historical prices**, alongside moving averages to visualize stock price trends.

---

## Ideal Use Cases
This tool is perfect for:
- **Financial enthusiasts** and **analysts** who want to explore stock price predictions for **Google**.
- **Machine learning learners** who want to see the application of neural networks in stock market analysis.
- Anyone looking to **visualize trends** and analyze the performance of a **predictive model** for stock prices over a long period of time.

---

## Requirements

To run this project locally, you will need:
- **Python 3.x**
- **TensorFlow**
- **Keras**
- **Streamlit**
- **yfinance**
- **pandas**
- **numpy**
- **matplotlib**
- **scikit-learn**

---

### **How to Run the Project**
1. Clone this repository or download the files.
2. Install the required dependencies using pip or conda.
```
pip install -r requirements.txt
```
4. Run the app using Streamlit.
```
streamlit run app.py
```
5. Enter the stock symbol (Google, use "GOOG") and view the predicted stock prices and visualizations.

