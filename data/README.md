# Data Documentation

This directory contains datasets organized by their primary use cases. The data is structured to support three main business intelligence areas: Inventory Management, Menu Engineering, and Shift Planning.

## Important Technical Notes

- **Timestamps**: All dates (e.g., created, contract_start, end_time) are UNIX Integers. Use `FROM_UNIXTIME(created)` in MySQL.
- **Money**: All monetary values are FLOAT and represent DKK.
- **Two Revenue Streams**:
  1. Platform Revenue: Found in fct_invoices (B2B bills) and fct_payments (Transaction fees)
  2. Merchant Revenue: Found in fct_orders (People buying food/goods)

## Inventory Management

The following datasets support inventory management operations:

### Dimension Tables

- **dim_items** - Raw inventory items and ingredients catalog with stock categories, quantities, units, and threshold levels
- **dim_bill_of_materials** - Recipe ingredient breakdown linking menu items to raw materials with quantities needed for production
- **dim_skus** - SKUs
- **dim_stock_categories** - Categories for organizing and classifying inventory/stock items
- **dim_add_ons** - Individual add-on options catalog with pricing, category assignment, and availability settings
- **dim_menu_item_add_ons** - Links specific add-ons to menu items with pricing, default selection status, and index ordering
- **dim_menu_items** - The merchant's product setup
- **dim_campaigns** - Marketing campaign definitions with placement, status, type, and scheduling information
- **dim_places** - The central hub. Every other table links here via place_id. Contains merchant/shop information including title (Merchant Name), contract_start (Signup Date), termination_date (Date they churned, Active if NULL), onboarded_by, and flags for bankrupt, duplicate, transferred_contract, seasonal, dormant (0/null or 1)
- **dim_taxonomy_terms** - The central lookup table for all standardized lists, tags, and categories used across the platform. Key fields: id (join from other tables), vocabulary (defines the list type: 'area', 'cuisine', 'sales_outcome', 'age_group'), name (human-readable text)
- **dim_users** - Contains BOTH our internal staff, merchant staff, and end-consumers. Key fields: id (Primary Key), type (Critical field to distinguish between 'admin', 'merchant_user', or 'consumer')

### Fact Tables

- **fct_order_items** - What specifically did they buy? Contains title (Item name), quantity (How many), price (Unit price), cost (Cost of Goods Sold if entered)
- **fct_orders** - Buying food/products from the merchant. Contains total_amount (The value of the basket), type ('eat_in', 'takeaway', 'delivery'), channel (App, Kiosk, or Counter), platform (via integrations: Wolt, JustEat, etc.)
- **fct_inventory_reports** - Periodic inventory snapshot reports showing stock levels, values, and variance analysis (contains report_date, item_id, quantity_on_hand, unit_cost, total_value, variance, location_id)
- **fct_cash_balances** - Daily cash drawer balance records with opening/closing amounts and reconciliation data (contains balance_date, opening_balance, closing_balance, expected_cash, actual_cash, variance)
- **fct_invoice_items** - Individual line items on merchant invoices with product/service details, quantities, and pricing (includes invoice_id, item_description, quantity, unit_price, total_amount, tax_amount)
- **fct_bonus_codes** - Promotional bonus codes with discount rules, validity periods, and usage limits (includes code, discount_type, discount_value, valid_from, valid_to, max_uses)
- **fct_campaigns** - Marketing campaign execution records with targeting, scheduling, and performance metrics (includes campaign_id, name, type, start_date, end_date, target_audience, status)

### Aggregated Views

- **most_ordered** - Pre-aggregated view of top-selling menu items by location and time period with order counts and revenue (includes menu_item_id, place_id, period, order_count, total_revenue, rank)

## Menu Engineering

The following datasets support menu engineering and optimization:

### Dimension Tables

- **dim_products** - Our Menu. Contains id (referred as product_id in dim_subscriptions and fct_hardware_requests), title (Product title), type (Software, Hardware or Service), Subtype (For Software type only, might be 'Subscription', 'Profit Share' or 'Addon')
- **dim_menu_items** - The merchant's product setup. Contains title (Item Name), price (Listing price), status (Is it available?)
- **dim_add_ons** - Individual add-on options catalog with pricing, category assignment, and availability settings
- **dim_add_on_categories** - Categories for grouping menu item add-ons and customizations (e.g., "Toppings", "Sauces", "Sides") with selection rules
- **dim_menu_item_add_on_categories** - Junction table linking menu items to their available add-on categories with selection constraints (min/max selections)
- **dim_menu_item_add_ons** - Links specific add-ons to menu items with pricing, default selection status, and index ordering
- **dim_menu_sections** - Menu sections/categories (e.g., "Appetizers", "Entrees") with type classification for organizing menu items
- **dim_sections** - Generic section definitions for organizing menu items with demo/training mode indicators and external IDs
- **dim_bill_of_materials** - Recipe ingredient breakdown linking menu items to raw materials with quantities needed for production
- **dim_items** - Raw inventory items and ingredients catalog with stock categories, quantities, units, and threshold levels
- **dim_campaigns** - Marketing campaign definitions with placement, status, type, and scheduling information
- **dim_taxonomy_terms** - The central lookup table for all standardized lists, tags, and categories used across the platform. Key fields: id (join from other tables), vocabulary (defines the list type: 'area', 'cuisine', 'sales_outcome', 'age_group'), name (human-readable text)
- **dim_users** - Contains BOTH our internal staff, merchant staff, and end-consumers. Key fields: id (Primary Key), type (Critical field to distinguish between 'admin', 'merchant_user', or 'consumer')

### Fact Tables

- **fct_payments** - The raw stream of card swipes, MobilePay, and online payments. Contains amount (Transaction value), type ('mobilepay', 'payment_terminal', 'cash', 'online' etc.), status (FILTER: WHERE status = 'Settled' to get the realized transactions), place_id (Who processed this payment), created (When it happened)
- **fct_order_items** - What specifically did they buy? Contains title (Item name), quantity (How many), price (Unit price), cost (Cost of Goods Sold if entered)
- **fct_app_event** - User interaction events captured in the app with timestamps, user profiles, and custom attribute tracking (10 attribute fields)
- **fct_app_events** - Streamlined app event tracking with event type, product details, and category for analytics
- **fct_app_links** - Tracks app deep links and referral sources with campaign attribution (contains fields like link_id, source, medium, campaign, content, term)
- **fct_notifications** - Push notifications and in-app messages sent to users with delivery status and engagement tracking (contains notification_id, user_id, type, title, message, sent_at, delivered_at, read_at, action_taken)
- **fct_bonus_codes** - Promotional bonus codes with discount rules, validity periods, and usage limits (includes code, discount_type, discount_value, valid_from, valid_to, max_uses)
- **fct_campaigns** - Marketing campaign execution records with targeting, scheduling, and performance metrics (includes campaign_id, name, type, start_date, end_date, target_audience, status)
- **fct_invoice_items** - Individual line items on merchant invoices with product/service details, quantities, and pricing (includes invoice_id, item_description, quantity, unit_price, total_amount, tax_amount)

### Aggregated Views

- **most_ordered** - Pre-aggregated view of top-selling menu items by location and time period with order counts and revenue (includes menu_item_id, place_id, period, order_count, total_revenue, rank)
- **user_events_export** - Exported stream of user interaction events for external analytics platforms (contains event_id, user_id, event_type, event_timestamp, properties, session_id, device_type)

## Shift Planning

The following datasets support shift planning and workforce management:

### Dimension Tables

- **dim_places** - The central hub. Every other table links here via place_id. Contains merchant/shop information including title (Merchant Name), contract_start (Signup Date), termination_date (Date they churned, Active if NULL), onboarded_by, and flags for bankrupt, duplicate, transferred_contract, seasonal, dormant (0/null or 1)
- **dim_tables** - Physical restaurant tables with capacity, location section assignment, booking status, QR codes, and shape attributes
- **dim_delivery_locations** - Delivery zones and addresses served by merchants with geographic coordinates and service tags
- **dim_campaigns** - Marketing campaign definitions with placement, status, type, and scheduling information
- **dim_taxonomy_terms** - The central lookup table for all standardized lists, tags, and categories used across the platform. Key fields: id (join from other tables), vocabulary (defines the list type: 'area', 'cuisine', 'sales_outcome', 'age_group'), name (human-readable text)
- **dim_users** - Contains BOTH our internal staff, merchant staff, and end-consumers. Key fields: id (Primary Key), type (Critical field to distinguish between 'admin', 'merchant_user', or 'consumer')

### Fact Tables

- **fct_order_items** - What specifically did they buy? Contains title (Item name), quantity (How many), price (Unit price), cost (Cost of Goods Sold if entered)
- **fct_orders** - Buying food/products from the merchant. Contains total_amount (The value of the basket), type ('eat_in', 'takeaway', 'delivery'), channel (App, Kiosk, or Counter), platform (via integrations: Wolt, JustEat, etc.)
- **fct_bonus_codes** - Promotional bonus codes with discount rules, validity periods, and usage limits (includes code, discount_type, discount_value, valid_from, valid_to, max_uses)
- **fct_campaigns** - Marketing campaign execution records with targeting, scheduling, and performance metrics (includes campaign_id, name, type, start_date, end_date, target_audience, status)

### Aggregated Views

- **most_ordered** - Pre-aggregated view of top-selling menu items by location and time period with order counts and revenue (includes menu_item_id, place_id, period, order_count, total_revenue, rank)

## Notes

- Tables may appear in multiple use cases as they serve cross-functional purposes
- The **dim_places** table serves as the central hub, with most other tables linking via place_id
- The **dim_taxonomy_terms** table provides standardized lookups across the platform
- The **dim_users** table contains multiple user types: internal staff, merchant staff, and end-consumers

## Additional Tables Available in Full Dataset

The following tables are available in the complete dataset but may be used less frequently or for specialized analyses:

### Financial & Billing
- **dim_accounts** - Account information and financial account structures
- **dim_subscriptions** - What we bill merchants for monthly (SaaS). Contains title (Product Name), amount (Monthly fee), start_date/end_date (Billing period, Active if end_date is NULL)
- **fct_invoices** - The actual bills sent to merchants. Contains total_amount (Invoice total), status ('paid', 'unpaid', 'cancelled'), due_date (When they were supposed to pay)
- **fct_earnings** - Revenue stream tracking (Note: data might not be accurate, different platform used currently)
- **fct_expenses** - Expense tracking (Note: data might not be accurate, different platform used currently)

### Merchant & Location Management
- **dim_chains** - Restaurant/retail chain information for multi-location merchants
- **dim_contacts** - Contact information for merchants and stakeholders
- **dim_table_sections** - Table groupings and floor plan sections within restaurants

### Hardware & Devices
- **dim_devices** - Hardware devices deployed with merchants
- **dim_payment_terminals** - Physical card machines with serial_number and model information
- **dim_receipt_printers** - Receipt printer configurations and hardware
- **fct_hardware_requests** - Service tickets where merchants requested new hardware

### Payment Integrations
- **dim_wolt_integrations** - Integration configurations with Wolt delivery platform
- **dim_vivawallet_integrations** - Integration configurations with VivaWallet payment processor
- **dim_elavon_integrations** - Integration configurations with Elavon payment processor
- **dim_mobilepay_integrations** - Integration configurations with MobilePay mobile payment system
- **dim_verifone_integrations** - Integration configurations with Verifone payment terminals

### Customer & Support
- **fct_interactions** - Communication records between staff and merchants (support tickets, calls, emails)
- **fct_dead_tasks** - Archive of incomplete or abandoned tasks for tracking and analysis
- **dim_tiers** - Customer loyalty/subscription tier levels defining service packages with cashback percentages, delivery charges, and minimum spend requirements

## Data Availability Notes

- Most tables are available in the complete Datasets folder
- Use case-specific folders (Inventory Management, Menu Engineering, Shift Planning) contain subsets relevant to each challenge
- Contact organizers if you need access to additional tables not included in your use case folder
