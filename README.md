# 📦 Inventory Management

## 🧩 Project Description

This project represents the data behind a **multi-merchant food and retail platform**, similar to systems used by restaurants, cafés, or shops to manage **sales, inventory, and marketing** in one place.

The goal of the project is to make it easy to **analyze revenue**, **track inventory**, and **understand how merchants and customers interact with the platform**.

---

## 💡 Features

The platform earns money in **two main ways**:

### 💼 Platform Revenue
Money earned from merchants through **invoices and transaction fees**.

### 🛒 Merchant Revenue
Money earned when customers **buy food or products** from merchants.

> ⏱️ All dates are stored as **UNIX timestamps**  
> 💰 All monetary values are in **Danish Kroner (DKK)**

---

## 🛠️ Technologies Used



---

## 🍽️ Inventory & Menu Management

The project includes a complete inventory system that connects **raw ingredients** to **menu items**, allowing for accurate stock and cost tracking.

This makes it possible to:

- 📊 Track stock levels of ingredients and finished products  
- 📋 See how menu items are built using recipes (bill of materials)  
- 🏷️ Manage SKUs, stock categories, and add-ons  
- 🔍 Monitor inventory reports and identify stock differences over time  

This setup helps merchants better understand **costs, usage, and potential waste**.

---

## 🧾 Orders, Payments & Operations

Customer purchases and merchant operations are tracked in detail, including:

- 🛍️ Items bought, quantities, and prices  
- 🍴 Order types (eat-in, takeaway, delivery)  
- 📱 Sales channels (App, Kiosk, Counter, delivery platforms)  
- 💵 Daily cash balances and reconciliation  

The platform also records **merchant invoices and invoice items**, making it possible to analyze platform earnings alongside merchant sales.

---

## 📣 Marketing & Campaigns

The project supports marketing and promotional features such as:

- 🎯 Campaign definitions and scheduling  
- 🎟️ Bonus codes and discounts  
- 📈 Campaign performance tracking  

This allows analysis of how **promotions and campaigns impact sales and customer behavior**.

---

## 🗂️ Data Structure Overview

- **📘 Dimension tables** store descriptive information  
  (merchants, users, menu items, inventory items, campaigns, categories, etc.)

- **📕 Fact tables** store transactional data  
  (orders, order items, inventory reports, invoices, payments, campaigns)

The **`dim_places`** table acts as the central hub of the model, linking most business activity to a specific merchant or location.

---
# ⚙️ Installation
1. **Clone and setup environment**:
```bash
git clone https://github.com/yourusername/fresh-flow-dashboard.git
cd fresh-flow-dashboard
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Prepare your data**:
- Place CSV files in `data/Inventory_Management/`
- Required files: `fct_orders.csv`, `fct_order_items.csv`, `fct_inventory_reports.csv`, `dim_items.csv`

4. **Run the dashboard**:
```bash
streamlit run app.py
```
---
## 🚀 Usage:

This project can be used to:

- 📊 Compare platform revenue vs merchant revenue  
- ⭐ Track best-selling menu items  
- 📦 Monitor inventory levels and stock variance  
- 🏪 Analyze merchant performance and churn  
- 📢 Measure the effectiveness of marketing campaigns
---

## 🏗️ Architecture:
The project follows a star schema architecture:

* Dimension tables store descriptive data
(merchants, users, menu items, inventory items, campaigns, categories)

* Fact tables store transactional data
(orders, order items, inventory reports, invoices, payments)

The dim_places table acts as the central hub, linking most business activity to a specific merchant or location.

---

## Team Members:
* Amonios Beshara: 
* Dalia Hassan: 
* Rawan Khalid: 
* Salma El-Hawary: 




