import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy import stats
import pickle

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class MarketTrendAnalyzer:
    def __init__(self, model_path, scaler_path, historical_data_path, seq_length=30, forecast_periods=10, confidence_level=0.95):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load your existing LSTM model without any additional parameters
        self.model = self.load_model(model_path)

        # Load other components (scaler and historical data)
        self.scaler = self.load_scaler(scaler_path)
        self.historical_data = self.load_historical_data(historical_data_path)

        # Set sequence length and forecast periods
        self.seq_length = seq_length
        self.forecast_periods = forecast_periods
        self.confidence_level = confidence_level

    def load_model(self, model_path):
        """Load and return the pre-trained PyTorch model."""
        try:
            # Initialize the model using your LSTMModel architecture
            model = LSTMModel(input_size=1, hidden_size=50, output_size=1)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()  # Set the model to evaluation mode
            return model
        except RuntimeError as e:
            print(f"Error loading model: {e}")
            raise

    
    def load_scaler(self, scaler_path):
        """Load and return the scaler."""
        with open(scaler_path, 'rb') as file:
            return pickle.load(file)

    def load_historical_data(self, historical_data_path):
        """Load and return historical data as a DataFrame."""
        return pd.read_csv(historical_data_path)

    from datetime import timedelta

    def generate_forecast(self, save_path=None):
        # Convert 'Close' column to numeric values, coercing errors to NaN
        self.historical_data['Close'] = pd.to_numeric(self.historical_data['Close'], errors='coerce')

        # Drop rows with NaN values in 'Close' column (could happen due to coercion)
        self.historical_data = self.historical_data.dropna(subset=['Close'])

        # Convert 'Date' column to datetime if it's not already
        self.historical_data['Date'] = pd.to_datetime(self.historical_data['Date'], errors='coerce')

        # Ensure raw data is a float32 numpy array
        raw_data = self.historical_data[['Close']].values.flatten().astype(np.float32)

        # Prepare initial sequence (last 'seq_length' closing prices)
        last_sequence = raw_data[-self.seq_length:]
        scaled_forecast = []

        # Generate forecasted values using the LSTM model
        current_sequence = torch.tensor(last_sequence, dtype=torch.float32).to(self.device)
        current_sequence = current_sequence.view(1, self.seq_length, 1)  # Reshape to (batch_size, seq_length, features)

        for t in range(self.forecast_periods):
            with torch.no_grad():  # No need to track gradients during inference
                pred = self.model(current_sequence).item()  # Predict the next value
                scaled_forecast.append(pred)

            # Update the sequence for the next prediction
            current_sequence = torch.cat((current_sequence[:, 1:, :], torch.tensor([[pred]], dtype=torch.float32).to(self.device).view(1, 1, 1)), dim=1)

        # Forecasted values (no scaling)
        forecasted_prices = np.array(scaled_forecast).flatten()

        # Calculate confidence intervals with time-decay factor
        last_known_price = self.historical_data['Close'].iloc[-1]
        forecast_std = np.std(self.historical_data['Close'].pct_change().dropna())  # Standard deviation of returns
        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)  # Z-score for confidence level

        # Time decay factor for increasing uncertainty (variance grows over time)
        time_decay = np.linspace(1, 1.5, num=self.forecast_periods)
        
        # Calculate lower and upper bounds for the confidence intervals
        confidence_intervals = {
            'lower': forecasted_prices - (z_score * forecast_std * forecasted_prices * time_decay),
            'upper': forecasted_prices + (z_score * forecast_std * forecasted_prices * time_decay)
        }

        # Ensure 'Date' is in datetime format before doing arithmetic
        last_date = self.historical_data['Date'].max()

        # Generate forecast dates (next 'forecast_periods' days from the last known date)
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=self.forecast_periods, freq='D')

        # Create DataFrame to store forecast results
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': forecasted_prices,
            'Lower_CI': confidence_intervals['lower'],
            'Upper_CI': confidence_intervals['upper']
        })

        # Save the forecast to a CSV file if save_path is provided
        if save_path:
            forecast_df.to_csv(save_path, index=False)
            print(f"Forecast saved to {save_path}")

        return forecast_df


    def plot_forecast(self):
        """Plot historical data, forecast, and confidence intervals."""
        plt.figure(figsize=(15, 8))

        # Plot historical data (original Close prices)
        plt.plot(self.historical_data['Date'], self.historical_data['Close'], label='Historical Data', color='blue')

        # Plot forecast
        forecast_df = self.generate_forecast()
        plt.plot(forecast_df['Date'], forecast_df['Forecast'], label='Forecast', color='red', linestyle='--')

        # Plot confidence intervals
        plt.fill_between(forecast_df['Date'], forecast_df['Lower_CI'], forecast_df['Upper_CI'],
                        alpha=0.2, color='red', label=f'{self.confidence_level*100}% Confidence Interval')

        # Set X-axis limits to focus between 2024 and 2025
        plt.xlim(pd.to_datetime('2024-01-01'), pd.to_datetime('2025-12-31'))

        plt.title('Stock Price Forecast with Confidence Intervals')
        plt.xlabel('Date')
        plt.ylabel('Stock Price ($)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


    def analyze_trends(self):
        """Analyze market trends with trend magnitude capped at 15%."""
        forecast_df = self.generate_forecast()

        # Calculate percentage-based trend
        initial_price = forecast_df['Forecast'].iloc[0]
        final_price = forecast_df['Forecast'].iloc[-1]
        trend_pct = ((final_price - initial_price) / initial_price) * 100

        # Cap trend magnitude and calculate volatility
        forecast_volatility = min(forecast_df['Forecast'].pct_change().std(), 0.02)  # Cap at 2%

        return {
            'Trend': 'Upward' if trend_pct > 0 else 'Downward',
            'Trend_Magnitude': min(abs(trend_pct), 15),
            'Volatility': forecast_volatility,
            'Max_Price': forecast_df['Forecast'].max(),
            'Min_Price': forecast_df['Forecast'].min(),
            'Price_Range': forecast_df['Forecast'].max() - forecast_df['Forecast'].min()
        }

    def generate_report(self):
        """Generate a report based on trend analysis."""
        risk_metrics = self.analyze_trends()
        report = f"""
        Market Trend Analysis Report
        ============================

        1. Overall Trend Analysis:
        -------------------------
        Direction: {risk_metrics['Trend']}
        Trend Magnitude: {risk_metrics['Trend_Magnitude']:.2f}

        2. Price Projections:
        --------------------
        Forecasted Maximum Price: ${risk_metrics['Max_Price']:.2f}
        Forecasted Minimum Price: ${risk_metrics['Min_Price']:.2f}
        Expected Price Range: ${risk_metrics['Price_Range']:.2f}

        3. Volatility Analysis:
        ----------------------
        Forecasted Volatility: {risk_metrics['Volatility']*100:.2f}%

        4. Market Opportunities and Risks:
        -------------------------------
        Main Opportunities:
        * {'Price appreciation potential' if risk_metrics['Trend'] == 'Upward' else 'Potential buying opportunities during dips'}
        * {'Momentum trading opportunities' if risk_metrics['Volatility'] > 0.02 else 'Stable price movement expected'}

        Main Risks:
        * {'High volatility risk' if risk_metrics['Volatility'] > 0.02 else 'Limited price movement'}
        * {'Potential for significant drawdowns' if risk_metrics['Trend'] == 'Downward' else 'Overvaluation risk'}

        5. Investment Implications:
        ------------------------
        {'Consider position sizing and stop-loss orders due to high volatility' if risk_metrics['Volatility'] > 0.02 else 'Suitable for longer-term position holding'}
        """
        return report
