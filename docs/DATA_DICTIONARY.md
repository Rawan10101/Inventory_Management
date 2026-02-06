# Data Dictionary

This project expects CSV files in a single data directory. The columns below are the minimum required for a successful run.

## fct_orders.csv

| Column | Description |
| --- | --- |
| id | Unique order ID. |
| place_id | Merchant or store ID. |
| created | UNIX timestamp for order creation. |
| status | Order status, use `Closed` for completed orders. |

## fct_order_items.csv

| Column | Description |
| --- | --- |
| order_id | Order ID that joins to `fct_orders.id`. |
| item_id | Item ID that joins to `dim_items.id`. |
| quantity | Quantity purchased. |
| price | Unit price for the item. |

## dim_items.csv

| Column | Description |
| --- | --- |
| id | Unique item ID. |
| title | Item name. |
| type | Item type or category. |
| manage_inventory | 1 if item is inventory-managed, 0 otherwise. |
| price | Optional unit price. |
| unit_cost | Optional unit cost for inventory valuation. |

## dim_places.csv

| Column | Description |
| --- | --- |
| id | Unique place or merchant ID. |
| title | Merchant name. |
| contract_start | Contract start date. |
| termination_date | Termination date, leave empty for active merchants. |

## fct_inventory_reports.csv

| Column | Description |
| --- | --- |
| report_date | Inventory snapshot date. |
| item_id | Item ID that joins to `dim_items.id`. |
| quantity_on_hand | Units currently in stock. |
| unit_cost | Cost per unit. |
| total_value | Quantity times unit cost. |

## dim_bill_of_materials.csv

| Column | Description |
| --- | --- |
| menu_item_id | Menu item ID that joins to `dim_items.id`. |
| ingredient_id | Ingredient item ID. |
| ingredient_name | Ingredient name. |
| quantity_per_serving | Quantity used per serving. |
| stock_unit | Unit of measure. |
| unit_cost | Ingredient unit cost. |
| shelf_life_days | Shelf life in days. |
