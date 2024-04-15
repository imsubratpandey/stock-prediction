import streamlit as st
import yfinance as yahoo_finance

from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as graph_objects

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")
stocks = ("AAPL", "GOOG", "MSFT", "GME")
selected_stocks = st.selectbox("Select dataset for prediction", stocks)
n_years = st.slider("Years of prediction:", 1, 4);
period = n_years * 365

# @st.cache
def load_data(ticker):
    data = yahoo_finance.download(ticker, START, TODAY);
    data.reset_index(inplace = True)
    return data

data_load_state = st.text("Loading data...")
data = load_data(selected_stocks)
data_load_state.text("Loading data...done!")

st.subheader("Raw Data")
st.write(data.tail())

def plot_raw_data():
    figure = graph_objects.Figure()
    figure.add_trace(graph_objects.Scatter(x = data["Date"], y = data["Open"], name = "stock_open"))
    figure.add_trace(graph_objects.Scatter(x = data["Date"], y = data["Close"], name = "stock_close"))
    figure.layout.update(title_text = "Time Series Data", xaxis_rangeslider_visible = True)
    st.plotly_chart(figure)

plot_raw_data()

dataframe_train = data[["Date", "Close"]]
dataframe_train = dataframe_train.rename(columns = {"Date": "ds", "Close": "y"})

model = Prophet()
model.fit(dataframe_train)
future_dataframe = model.make_future_dataframe(periods = period)
forecast = model.predict(future_dataframe)

st.subheader("Forecast Data")
st.write(forecast.tail())

st.subheader("Forecast Plot")
figure_data = plot_plotly(model, forecast)
st.plotly_chart(figure_data)

st.subheader("Forecast Components")
figure_components = model.plot_components(forecast)
st.write(figure_components)