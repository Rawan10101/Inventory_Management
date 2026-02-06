"""
Generate synthetic data for Fresh Flow demos.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def _make_places(num_places: int, start_date: datetime) -> pd.DataFrame:
    places = []
    for i in range(1, num_places + 1):
        places.append(
            {
                "id": i,
                "title": f"Merchant {i}",
                "contract_start": (start_date - timedelta(days=365)).date().isoformat(),
                "termination_date": "",
            }
        )
    return pd.DataFrame(places)


def _make_items(num_items: int, rng: np.random.Generator) -> Tuple[pd.DataFrame, np.ndarray]:
    item_ids = np.arange(1, num_items + 1)
    prices = rng.uniform(20, 120, size=num_items).round(2)
    items = []
    for item_id, price in zip(item_ids, prices):
        items.append(
            {
                "id": int(item_id),
                "title": f"Menu Item {item_id}",
                "type": "menu",
                "manage_inventory": 1,
                "price": float(price),
                "unit_cost": float(price * 0.6),
            }
        )
    return pd.DataFrame(items), prices


def _make_ingredients(start_id: int, count: int, rng: np.random.Generator) -> pd.DataFrame:
    ingredient_ids = np.arange(start_id, start_id + count)
    costs = rng.uniform(2, 15, size=count).round(2)
    items = []
    for ingredient_id, cost in zip(ingredient_ids, costs):
        items.append(
            {
                "id": int(ingredient_id),
                "title": f"Ingredient {ingredient_id}",
                "type": "ingredient",
                "manage_inventory": 1,
                "price": float(cost * 1.8),
                "unit_cost": float(cost),
            }
        )
    return pd.DataFrame(items)


def generate_synthetic_data(
    output_dir: str,
    days: int = 90,
    num_places: int = 3,
    num_items: int = 30,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days - 1)

    places_df = _make_places(num_places, datetime.now())
    items_df, item_prices = _make_items(num_items, rng)

    ingredient_count = max(10, num_items // 3)
    ingredients_df = _make_ingredients(1001, ingredient_count, rng)

    dim_items = pd.concat([items_df, ingredients_df], ignore_index=True)

    orders = []
    order_items = []
    order_id = 1

    popularity = rng.uniform(0.5, 1.5, size=num_items)
    weights = popularity / popularity.sum()

    for day_offset in range(days):
        current_date = start_date + timedelta(days=day_offset)
        for place_id in range(1, num_places + 1):
            num_orders = int(max(5, rng.poisson(20)))
            for _ in range(num_orders):
                created_time = datetime.combine(current_date, datetime.min.time()) + timedelta(
                    minutes=int(rng.integers(0, 1440))
                )
                orders.append(
                    {
                        "id": order_id,
                        "place_id": place_id,
                        "created": int(created_time.timestamp()),
                        "status": "Closed",
                    }
                )

                items_in_order = int(rng.integers(1, 4))
                item_choices = rng.choice(np.arange(1, num_items + 1), size=items_in_order, replace=False, p=weights)

                for item_id in item_choices:
                    quantity = int(rng.integers(1, 4))
                    price = float(item_prices[item_id - 1])
                    order_items.append(
                        {
                            "order_id": order_id,
                            "item_id": int(item_id),
                            "quantity": quantity,
                            "price": price,
                        }
                    )

                order_id += 1

    fct_orders = pd.DataFrame(orders)
    fct_order_items = pd.DataFrame(order_items)

    report_date = end_date.isoformat()
    inventory_levels = rng.integers(20, 200, size=num_items)
    inventory_report = pd.DataFrame(
        {
            "report_date": report_date,
            "item_id": np.arange(1, num_items + 1),
            "quantity_on_hand": inventory_levels,
            "unit_cost": (item_prices * 0.6).round(2),
        }
    )
    inventory_report["total_value"] = inventory_report["quantity_on_hand"] * inventory_report["unit_cost"]

    # Bill of materials
    bom_rows = []
    ingredient_ids = ingredients_df["id"].tolist()
    for item_id in range(1, num_items + 1):
        ingredient_choices = rng.choice(ingredient_ids, size=2, replace=False)
        for ingredient_id in ingredient_choices:
            bom_rows.append(
                {
                    "menu_item_id": item_id,
                    "ingredient_id": int(ingredient_id),
                    "ingredient_name": f"Ingredient {ingredient_id}",
                    "quantity_per_serving": float(rng.uniform(0.1, 2.0)),
                    "stock_unit": "unit",
                    "unit_cost": float(rng.uniform(2, 15)),
                    "shelf_life_days": int(rng.integers(2, 7)),
                }
            )

    dim_bom = pd.DataFrame(bom_rows)

    # Write files
    fct_orders.to_csv(output_path / "fct_orders.csv", index=False)
    fct_order_items.to_csv(output_path / "fct_order_items.csv", index=False)
    dim_items.to_csv(output_path / "dim_items.csv", index=False)
    places_df.to_csv(output_path / "dim_places.csv", index=False)
    inventory_report.to_csv(output_path / "fct_inventory_reports.csv", index=False)
    dim_bom.to_csv(output_path / "dim_bill_of_materials.csv", index=False)

    print(f"Synthetic data written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic data for Fresh Flow demos.")
    parser.add_argument("--output", default="data/Inventory_Management", help="Output directory")
    parser.add_argument("--days", type=int, default=90, help="Number of days to generate")
    parser.add_argument("--places", type=int, default=3, help="Number of places")
    parser.add_argument("--items", type=int, default=30, help="Number of menu items")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    generate_synthetic_data(args.output, args.days, args.places, args.items, args.seed)


if __name__ == "__main__":
    main()
