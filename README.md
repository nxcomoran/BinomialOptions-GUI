# BinomialOptions-GUI
A dashboard GUI made using TKinter--Calculates the binomial fair price of option contracts at a specified expiration for a specified ticker. Dashboard displays margin between realized and calculated prices pulled from yfinance.

# 🧮 Options Screening Tool (Binomial Pricing Model)

An interactive **Python application** that identifies **potentially mispriced options** using a **Binomial Options Pricing Model**.  
The tool integrates live market data via `yfinance`, computes theoretical option values, and compares them against market ask prices — all within a **Tkinter GUI**.

---

## 🚀 Features
- 🔹 **Real-Time Data Fetching** – retrieves live option chains and stock data via Yahoo Finance  
- 🔹 **Dynamic Binomial Pricing Model** – calculates theoretical call and put prices  
- 🔹 **Option Screening** – flags contracts where theoretical price > market ask  
- 🔹 **Adjustable Parameters** – change ticker, expiration date, volatility, and risk-free rate  
- 🔹 **Interactive GUI** – clean Tkinter interface with a scrollable output window  

---

## 🧠 Problem Statement
Options markets are complex and frequently inefficient.  
This project provides a **quantitative screening tool** that helps traders and analysts:
- Systematically find **undervalued contracts**
- Understand how **volatility, interest rates,** and **underlying price** affect option value
- Replace guesswork with a **data-driven approach** to pricing

---

## ⚙️ How It Works
1. Fetch option chain and historical stock data using `yfinance`  
2. Estimate **volatility** from historical returns  
3. Compute **theoretical call and put values** using the Binomial Options Pricing Model  
4. Compare theoretical vs. market ask prices  
5. Display **ranked results** with potential profit margins in the GUI  

---

## 🧩 Tech Stack
| Component | Description |
|------------|-------------|
| **Language** | Python 3.10+ |
| **Libraries** | `yfinance`, `numpy`, `pandas`, `math`, `tkinter`, `matplotlib` |
| **Framework** | Tkinter (GUI) |
| **Data Source** | Yahoo Finance |

---

## 📊 Example Output
```

--- Options Screening for QQQ on 2025-07-18 ---
Current Stock Price (QQQ): 412.55
Calculated Annual Volatility: 0.2164
Risk-Free Rate: 0.0400
Model Steps: 256

--- Profitable Contracts ---

| Type | Strike | Expiration | Theoretical | Ask    | Margin |
| ---- | ------ | ---------- | ----------- | ------ | ------ |
| Call | 385.00 | 2025-07-18 | $32.15      | $27.50 | +$4.65 |
| Put  | 440.00 | 2025-07-18 | $31.82      | $27.90 | +$3.92 |

--- Screening Complete ---

````

---

## 🧰 Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/options-screening-tool.git
cd options-screening-tool

# Install dependencies
pip install numpy pandas yfinance matplotlib
````

---

## ▶️ Run

```bash
python options_screening_tool.py
```

---

## 💡 Future Improvements

* Add **Black-Scholes comparison** for validation
* Visualize **strike–price relationships** using Matplotlib
* Integrate **Greeks (Δ, Γ, Θ, Vega)** for risk analysis
* Enable **batch scanning** for multiple tickers

---

## 📜 License

Released under the **MIT License** — free for educational and analytical use.

---

### 🧩 Author

**Nico Moran**
📈 Quantitative Finance & Data Analysis
📧 [nxcomoran@gmail.com](mailto:nxcomoran@gmail.com)
