import numpy as np
import matplotlib.pyplot as plt # Keep import just in case, though not used in GUI output
from datetime import date, datetime
import yfinance as yf
import pandas as pd
import math
import tkinter as tk
from tkinter import ttk, scrolledtext # scrolledtext is good for output
import sys # Needed to redirect stdout
import io # Needed for string buffer

# --- Configuration Parameters (Can be modified in the GUI later if desired) ---
# Initial values when the GUI starts
DEFAULT_TICKER = "QQQ"
DEFAULT_EXPIRY_DATE = "2025-07-18" # YYYY-MM-DD
DEFAULT_FIDELITY = 256
DEFAULT_RISK_FREE_RATE = 0.04

# --- Utility Functions (from your existing code, adapted slightly for GUI) ---

def get_options_chain_filtered(ticker_symbol, expiration_date_str):
    """
    Fetches the options chain for a given ticker and expiration date using yfinance,
    returning 'ask', 'strike', and 'expirationDate'. Includes bid for potential future use.

    Args:
        ticker_symbol (str): The stock ticker symbol (e.g., 'SPY', 'AAPL').
        expiration_date_str (str): The expiration date in 'YYYY-MM-DD' format.

    Returns:
        tuple: A tuple containing two pandas DataFrames: (filtered_calls_df, filtered_puts_df).
               Each DataFrame contains 'strike', 'ask', 'bid', and 'expirationDate' columns.
               Returns (None, None) if the ticker or date is invalid, or
               if an error occurs fetching the data.
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        available_dates = stock.options

        if expiration_date_str not in available_dates:
            # Print error message will now go to the GUI text area
            print(f"Error: Expiration date '{expiration_date_str}' not found for {ticker_symbol}.")
            print(f"Available dates are: {available_dates}")
            return None, None

        options_chain = stock.option_chain(expiration_date_str)
        calls_df = options_chain.calls
        puts_df = options_chain.puts

        if calls_df.empty and puts_df.empty:
            print(f"Warning: No options data found for {ticker_symbol} on {expiration_date_str}.")
            return None, None

        # Add the expiration date as a new column
        if not calls_df.empty:
             calls_df['expirationDate'] = expiration_date_str
        if not puts_df.empty:
             puts_df['expirationDate'] = expiration_date_str

        # Select only the desired columns (including 'bid' and 'ask')
        # It's good practice to fetch bid/ask together if available
        required_columns = ['strike', 'bid', 'ask', 'expirationDate'] 

        # Robustly select columns that are actually present
        # Ensure 'strike' and at least 'ask' or 'bid' are present
        calls_available_cols = [col for col in required_columns if col in calls_df.columns]
        puts_available_cols = [col for col in required_columns if col in puts_df.columns]
        
        # Add a check to ensure essential columns are present
        if 'strike' not in calls_available_cols or ('ask' not in calls_available_cols and 'bid' not in calls_available_cols):
             print(f"Error: Essential columns ('strike' and 'ask'/'bid') not found in calls data.")
             return None, None
        if 'strike' not in puts_available_cols or ('ask' not in puts_available_cols and 'bid' not in puts_available_cols):
             print(f"Error: Essential columns ('strike' and 'ask'/'bid') not found in puts data.")
             return None, None


        filtered_calls_df = calls_df[calls_available_cols]
        filtered_puts_df = puts_df[puts_available_cols]
        
        # Rename 'ask' to 'realized_value' as before for consistency with model comparison
        # Ensure 'ask' exists before renaming
        if 'ask' in filtered_calls_df.columns:
            filtered_calls_df = filtered_calls_df.rename(columns={'ask': 'realized_value'})
        else: # If 'ask' is not available, use 'bid' as 'realized_value' (less ideal but prevents failure)
             if 'bid' in filtered_calls_df.columns:
                  filtered_calls_df = filtered_calls_df.rename(columns={'bid': 'realized_value'})
                  print("Warning: 'ask' price not available for calls, using 'bid' price as 'realized_value'.")
             else:
                  print("Error: Neither 'ask' nor 'bid' price found for calls.")
                  return None, None # Cannot proceed without a market price

        if 'ask' in filtered_puts_df.columns:
            filtered_puts_df = filtered_puts_df.rename(columns={'ask': 'realized_value'})
        else: # If 'ask' is not available, use 'bid' as 'realized_value' (less ideal but prevents failure)
             if 'bid' in filtered_puts_df.columns:
                  filtered_puts_df = filtered_puts_df.rename(columns={'bid': 'realized_value'})
                  print("Warning: 'ask' price not available for puts, using 'bid' price as 'realized_value'.")
             else:
                  print("Error: Neither 'ask' nor 'bid' price found for puts.")
                  return None, None # Cannot proceed without a market price


        # Optional: Filter out rows where realized_value (ask/bid) is 0 or NaN
        filtered_calls_df = filtered_calls_df[filtered_calls_df['realized_value'] > 0].dropna(subset=['realized_value'])
        filtered_puts_df = filtered_puts_df[filtered_puts_df['realized_value'] > 0].dropna(subset=['realized_value'])


        return filtered_calls_df, filtered_puts_df

    except Exception as e:
        print(f"An error occurred while fetching or processing options data for {ticker_symbol} on {expiration_date_str}: {e}")
        return None, None

def calculate_dte(expiration_date_str, date_format='%Y-%m-%d'):
    """
    Calculates the number of whole days from today until a specified date.

    Args:
        expiration_date_str (str): The date string for the expiration date.
        date_format (str): The format string corresponding to expiration_date_str
                           (e.g., '%Y-%m-%d' for '2025-07-18', '%m/%d/%Y' for '07/18/2025').
                           Defaults to 'YYYY-MM-DD'.

    Returns:
        int or None: The number of whole days until the expiration date (non-negative).
                     Returns 0 if the date is today or in the past.
                     Returns None if the date string cannot be parsed
                     using the provided format.
    """
    today = date.today()
    try:
        expiration_date_obj = datetime.strptime(expiration_date_str, date_format).date()
        time_difference = expiration_date_obj - today
        dte = time_difference.days
        return max(0, dte) # Return 0 for today or past dates
    except ValueError:
        # Error message already handled by print, which is redirected to GUI
        # print(f"Error: Could not parse date string '{expiration_date_str}' with format '{date_format}'.")
        return None

# Pascal's Triangle function (unchanged)
def get_pascal_row(row_index):
    """
    Generates a list representing the row_index-th row of Pascal's Triangle.
    Row index starts at 0.
    """
    if row_index < 0:
        return "Row index cannot be negative."
    if row_index == 0:
        return [1]
    current_row = [1]
    for i in range(1, row_index + 1):
        next_row = [1]
        for j in range(len(current_row) - 1):
            next_value = current_row[j] + current_row[j+1]
            next_row.append(next_value)
        next_row.append(1)
        current_row = next_row
    return current_row

# --- Binomial Option Pricing Function (mostly unchanged) ---
def calculate_binomial_price(S, K, T, sigma, r, N):
    """
    Calculates theoretical Call and Put prices using a simple binomial model
    based on the provided parameters and the user's formulas for u, d, p, Discount_Factor.

    Args:
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to expiry in years (DTE / ~365).
        sigma (float): Annual volatility (as a decimal).
        r (float): Annual risk-free interest rate (as a decimal).
        N (int): Number of steps (Fidelity).

    Returns:
        tuple: (calculated_call_price, calculated_put_price) or (0.0, 0.0) if invalid input or calculation fails.
    """
    if N <= 0 or T <= 0 or sigma <= 0:
         return 0.0, 0.0

    dt = T / N # Time step

    try:
        u = (1 + sigma)**(1/N)
        d = 1 / u
        if u == d:
             print("Warning: u equals d, calculation not possible for K={}. Returning 0.".format(K)) # Add strike context to warning
             return 0.0, 0.0

        p = ((1 + r * dt) - d) / (u - d)
        p_inv = 1 - p

        Discount_Factor = (1 + r * dt)**N

        if not (0 <= p <= 1):
            print(f"Warning: Calculated probability p ({p:.4f}) outside [0, 1] for K={K}. Returning 0.".format(K)) # Add strike context to warning
            return 0.0, 0.0

    except Exception as e:
         print(f"Error in binomial parameter calculation for K={K}: {e}") # Add strike context to warning
         return 0.0, 0.0


    possible_terminal_values = [S * (u ** i) * (d ** (N - i)) for i in range(N + 1)]
    terminal_call_values = [max(0, val - K) for val in possible_terminal_values]
    terminal_put_values = [max(0, K - val) for val in possible_terminal_values]

    pascal_row_values = get_pascal_row(N)
    if not isinstance(pascal_row_values, list) or len(pascal_row_values) != N + 1:
         print(f"Error getting Pascal's triangle row {N} for K={K}. Cannot calculate probabilities.".format(K)) # Add strike context to warning
         return 0.0, 0.0

    probabilities = [(p ** i) * (p_inv ** (N - i)) * pascal_row_values[i] for i in range(N + 1)]

    # Sum probabilities check is computationally expensive for high N, maybe skip for speed in GUI
    # print(f"Sum of probabilities for K={K}: {sum(probabilities):.4f}")

    weighted_average_Call = np.dot(terminal_call_values, probabilities)
    weighted_average_Put = np.dot(terminal_put_values, probabilities)

    if Discount_Factor == 0:
        print("Warning: Discount factor is zero for K={}. Division by zero prevented.".format(K)) # Add strike context to warning
        return 0.0, 0.0

    calculated_call_price = weighted_average_Call / Discount_Factor
    calculated_put_price = weighted_average_Put / Discount_Factor

    return calculated_call_price, calculated_put_price

# --- GUI Implementation ---

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Options Screening Tool")
        self.pack(padx=10, pady=10) # Add padding around the main frame
        self.create_widgets()

    def create_widgets(self):
        # Input Frame
        input_frame = ttk.Frame(self)
        input_frame.pack(pady=5, fill='x')

        # Ticker Input
        ttk.Label(input_frame, text="Ticker Symbol:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.ticker_var = tk.StringVar(value=DEFAULT_TICKER)
        self.ticker_entry = ttk.Entry(input_frame, textvariable=self.ticker_var, width=20)
        self.ticker_entry.grid(row=0, column=1, sticky='ew', padx=5, pady=2)

        # Expiration Date Input
        ttk.Label(input_frame, text="Expiration Date (YYYY-MM-DD):").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.expiry_date_var = tk.StringVar(value=DEFAULT_EXPIRY_DATE)
        self.expiry_date_entry = ttk.Entry(input_frame, textvariable=self.expiry_date_var, width=20)
        self.expiry_date_entry.grid(row=1, column=1, sticky='ew', padx=5, pady=2)

        # Make the input frame expand with the window
        input_frame.columnconfigure(1, weight=1)


        # Run Button
        self.run_button = ttk.Button(self, text="Run Screening", command=self.run_screening_gui)
        self.run_button.pack(pady=10)

        # Output Text Area
        # Using scrolledtext for automatic scrollbars
        self.output_text = scrolledtext.ScrolledText(self, wrap='word', height=20, width=80)
        self.output_text.pack(pady=5, fill='both', expand=True) # Fill and expand with window

        # Redirect stdout to the text widget
        self.original_stdout = sys.stdout
        self.stdout_redirector = StdoutRedirector(self.output_text)


    def run_screening_gui(self):
        """
        This function orchestrates the screening process when the button is clicked.
        It redirects console output to the GUI text area.
        """
        ticker_symbol = self.ticker_var.get().strip().upper()
        expiration_date_str = self.expiry_date_var.get().strip()

        # Clear previous output and enable the text widget
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)

        # Redirect stdout
        sys.stdout = self.stdout_redirector

        # Disable the button while processing
        self.run_button.config(state=tk.DISABLED)
        self.output_text.insert(tk.END, "Starting screening...\n")
        self.output_text.update_idletasks() # Update GUI immediately

        try:
            # --- Main Processing Logic (adapted from your script) ---

            print(f"--- Options Screening for {ticker_symbol} on {expiration_date_str} ---")

            # 1. Get DTE
            dte = calculate_dte(expiration_date_str)
            if dte is None or dte <= 0:
                 # Error message is already printed by calculate_dte or handled here
                 print("\nInvalid expiration date or date is in the past. Please check the date format (YYYY-MM-DD) and try again.")
                 return # Exit the function

            time_to_expiry_years = dte / 365.0 # Use 365 for time in years


            # 2. Get current Stock Price and Volatility
            try:
                stock = yf.Ticker(ticker_symbol)
                stock_info = stock.info # Get comprehensive info
                current_stock_price = stock_info.get('regularMarketPrice')

                if current_stock_price is None:
                     print(f"\nError: Could not get current market price for {ticker_symbol}. Is the ticker valid?")
                     return # Exit

                # Get historical volatility as a proxy for future volatility
                hist_data = stock.history(period="1y")
                if hist_data.empty or len(hist_data) < 2:
                     print(f"Error: Not enough historical data to calculate volatility for {ticker_symbol} using 1 year. Trying 6 months...")
                     hist_data = stock.history(period="6mo")
                     if hist_data.empty or len(hist_data) < 2:
                         print(f"Error: Not enough historical data to calculate volatility for {ticker_symbol}. Using default volatility ({DEFAULT_VOLATILITY:.2f}).")
                         # Use default volatility if historical data is insufficient
                         volatility = DEFAULT_VOLATILITY = 0.2 # Define a default volatility
                     else:
                          hist_data['returns'] = hist_data['Close'].pct_change().dropna() # Drop NaN from pct_change
                          # Annualize based on number of trading days in the period if less than 1 year?
                          # Simplest approach: assume 252 trading days/year regardless of period length
                          volatility = hist_data['returns'].std() * math.sqrt(252)

                else:
                    hist_data['returns'] = hist_data['Close'].pct_change().dropna() # Drop NaN from pct_change
                    volatility = hist_data['returns'].std() * math.sqrt(252) # Annualize daily volatility

                # Define DEFAULT_VOLATILITY if it wasn't set above
                try:
                     DEFAULT_VOLATILITY
                except NameError:
                     DEFAULT_VOLATILITY = 0.2 # Ensure it's defined

                print(f"Current Stock Price ({ticker_symbol}): {current_stock_price:.2f}")
                print(f"Calculated Annual Volatility (proxy): {volatility:.4f}")
                print(f"Time to Expiry (DTE): {dte} days ({time_to_expiry_years:.4f} years)")
                print(f"Model Fidelity (Steps): {DEFAULT_FIDELITY}")
                print(f"Risk-Free Rate: {DEFAULT_RISK_FREE_RATE:.4f}")


            except Exception as e:
                print(f"\nAn error occurred while fetching stock data or calculating volatility: {e}")
                return # Exit


            # 3. Fetch Options Chain Data
            print(f"\nFetching options chain data for {ticker_symbol} expiring on {expiration_date_str}...")
            calls_data_filtered, puts_data_filtered = get_options_chain_filtered(ticker_symbol, expiration_date_str)

            if calls_data_filtered is None and puts_data_filtered is None:
                print("\nCould not retrieve any options data. Please check ticker and date.")
                return # Exit

            # Initialize list to store profitable contracts
            profitable_contracts = []

            # 4. Process Call Options
            print("\nProcessing Call Options...")
            if calls_data_filtered is not None and not calls_data_filtered.empty:
                for index, row in calls_data_filtered.iterrows():
                    strike = row['strike']
                    realized_value = row['realized_value'] # This is the 'ask' price from yfinance

                    # Calculate theoretical price using the binomial model
                    calculated_call_value, _ = calculate_binomial_price(
                        S=current_stock_price,
                        K=strike,
                        T=time_to_expiry_years,
                        sigma=volatility,
                        r=DEFAULT_RISK_FREE_RATE,
                        N=DEFAULT_FIDELITY
                    )

                    # Check for profitability (Realized < Calculated)
                    # Only add if calculated value is positive and greater than the real value
                    if calculated_call_value > 0 and calculated_call_value > realized_value:
                        profit_margin = calculated_call_value - realized_value
                        profitable_contracts.append({
                            'Strike': strike,
                            'Expiration': expiration_date_str,
                            'Type': 'Call',
                            'Calculated Value': calculated_call_value,
                            'Realized Value (Ask)': realized_value,
                            'Profit Margin': profit_margin
                        })

            else:
                print("No calls data to process.")


            # 5. Process Put Options
            print("\nProcessing Put Options...")
            if puts_data_filtered is not None and not puts_data_filtered.empty:
                 for index, row in puts_data_filtered.iterrows():
                    strike = row['strike']
                    realized_value = row['realized_value'] # This is the 'ask' price from yfinance

                    # Calculate theoretical price using the binomial model
                    _, calculated_put_value = calculate_binomial_price(
                        S=current_stock_price,
                        K=strike,
                        T=time_to_expiry_years,
                        sigma=volatility,
                        r=DEFAULT_RISK_FREE_RATE,
                        N=DEFAULT_FIDELITY
                    )

                    # Check for profitability (Realized < Calculated)
                    # Only add if calculated value is positive and greater than the real value
                    if calculated_put_value > 0 and calculated_put_value > realized_value:
                        profit_margin = calculated_put_value - realized_value
                        profitable_contracts.append({
                            'Strike': strike,
                            'Expiration': expiration_date_str,
                            'Type': 'Put',
                            'Calculated Value': calculated_put_value,
                            'Realized Value (Ask)': realized_value,
                            'Profit Margin': profit_margin
                        })

            else:
                print("No puts data to process.")

            # 6. Create and Sort the Database (DataFrame)
            profitable_df = pd.DataFrame(profitable_contracts)

            if profitable_df.empty:
                print("\n--- No Profitable Contracts Found ---")
            else:
                # Sort by Profit Margin descending
                profitable_df_sorted = profitable_df.sort_values(by='Profit Margin', ascending=False).reset_index(drop=True)

                # Format currency columns - do this AFTER sorting
                # Create a copy to avoid SettingWithCopyWarning if needed, though .map is usually safe
                profitable_df_sorted_display = profitable_df_sorted.copy()
                profitable_df_sorted_display['Calculated Value'] = profitable_df_sorted_display['Calculated Value'].map('${:,.2f}'.format)
                profitable_df_sorted_display['Realized Value (Ask)'] = profitable_df_sorted_display['Realized Value (Ask)'].map('${:,.2f}'.format)
                profitable_df_sorted_display['Profit Margin'] = profitable_df_sorted_display['Profit Margin'].map('${:,.2f}'.format)


                print("\n--- Profitable Contracts (Sorted by Profit Margin) ---")
                # Display the DataFrame using to_string()
                print(profitable_df_sorted_display.to_string())

            print("\n--- Screening Complete ---")


        except Exception as e:
            # Catch any unexpected errors during the process
            print(f"\nAn unexpected error occurred during screening: {e}")

        finally:
            # Always restore stdout and re-enable button
            sys.stdout = self.original_stdout
            self.run_button.config(state=tk.NORMAL)
            self.output_text.config(state=tk.DISABLED) # Make text read-only after displaying results


# Helper class to redirect stdout to the Text widget
class StdoutRedirector(object):
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = io.StringIO() # Use an in-memory buffer

    def write(self, str):
        self.buffer.write(str) # Write to buffer first
        # Update the text widget more frequently (e.g., on newline or flush)
        if str.endswith('\n') or str.endswith('\r') or self.buffer.tell() > 100: # Update if newline, return, or buffer is full
             self.flush()

    def flush(self):
        # Insert buffer content into text widget
        content = self.buffer.getvalue()
        if content:
            self.text_widget.insert(tk.END, content)
            self.text_widget.see(tk.END) # Scroll to the end
            self.text_widget.update_idletasks() # Update GUI immediately
            self.buffer.seek(0) # Reset buffer position
            self.buffer.truncate(0) # Clear buffer content


# --- Run the GUI ---
if __name__ == "__main__":
    root = tk.Tk()
    app = Application(master=root)
    root.mainloop()