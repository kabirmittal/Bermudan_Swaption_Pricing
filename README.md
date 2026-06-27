# Bermudan Swaption Pricing using the Hull–White One-Factor Model

## Overview

This project implements an end-to-end pricing framework for Bermudan payer swaptions using the Hull–White one-factor interest rate model. It reconstructs the complete quantitative workflow used in fixed-income derivatives pricing, including yield curve construction, model calibration, Bermudan swaption valuation, and risk sensitivity analysis.

The implementation is fully parameterized and processes market data from JSON input without relying on external quantitative finance libraries such as QuantLib.

---

## Features

* Construct discount factors and instantaneous forward rates from a continuously compounded zero curve
* Calibrate the Hull–White one-factor model to market swaption volatility quotes
* Price Bermudan payer swaptions using a recombining interest-rate tree with optimal early exercise
* Compute bucketed Delta and Vega using bump-and-reprice methodology
* Fully parameterized implementation supporting varying market inputs and exercise schedules
* JSON-based input/output for automated evaluation

---

## Model Components

### Yield Curve Construction

* Discount factor computation
* Instantaneous forward rate estimation
* Linear interpolation of continuously compounded zero rates

### Hull–White Calibration

* Mean reversion parameter calibration
* Piecewise-constant volatility calibration
* Least-squares optimization against market normal implied volatilities

### Bermudan Swaption Pricing

* Recombining short-rate lattice
* Early exercise optimization
* Co-terminal swap valuation

### Risk Analysis

* Bucketed Vega computation
* Bucketed Delta computation
* Full recalibration after each market bump

---

## Technologies

* Python
* NumPy
* SciPy
* Pandas
* JSON

---

## Repository Structure

```text
.
├── main.py
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/kabirmittal/Bermudan_Swaption_Pricing.git
cd Bermudan_Swaption_Pricing
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Run the solution using

```bash
python main.py < input.json
```

The program reads a JSON object from standard input and outputs a JSON object containing:

* Yield curve
* Calibrated Hull–White parameters
* Bermudan swaption price
* Bucketed Vega sensitivities
* Bucketed Delta sensitivities

---

## Mathematical Techniques

* Hull–White one-factor short-rate model
* Interest rate curve bootstrapping
* Numerical optimization
* Trinomial interest-rate tree
* Early exercise dynamic programming
* Finite-difference risk sensitivities

---

## Notes

This implementation was developed as a quantitative finance programming challenge under the following constraints:

* No QuantLib or external quantitative finance libraries
* Fully parameterized solution
* Runtime-optimized numerical implementation
* Supports arbitrary market inputs and exercise schedules
