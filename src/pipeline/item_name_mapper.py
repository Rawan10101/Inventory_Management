"""
Item Name Mapper - Maps simple IDs to real product names from dim_menu_items.csv
"""

import pandas as pd
from pathlib import Path

class ItemNameMapper:
    """
    Maps simple item IDs (1-40) to real product names from dim_menu_items.csv
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.mapping = {}
        self._load_mapping()
    
    def _load_mapping(self):
        """Load and create item name mapping"""
        try:
            # Load simple items (IDs 1-40)
            simple_path = self.data_dir / "dim_items.csv"
            if not simple_path.exists():
                print(f"Warning: dim_items.csv not found at {simple_path}")
                return
            
            simple_df = pd.read_csv(simple_path)
            
            # Load real menu items (real product names)
            real_path = self.data_dir / "dim_menu_items.csv"
            if not real_path.exists():
                print(f"Warning: dim_menu_items.csv not found at {real_path}")
                # Fallback: use simple names
                for _, row in simple_df.iterrows():
                    self.mapping[row['id']] = {
                        'simple_name': row['title'],
                        'real_name': row['title'],
                        'price': row.get('price', 50)
                    }
                return
            
            real_df = pd.read_csv(real_path)
            
            # Create sequential mapping (since there's no direct link)
            print(f"Creating item name mapping...")
            print(f"   Simple items: {len(simple_df)}")
            print(f"   Real items: {len(real_df)}")
            
            for i, (_, simple_row) in enumerate(simple_df.iterrows()):
                simple_id = simple_row['id']
                simple_name = simple_row.get('title', f'Menu Item {simple_id}')
                
                if i < len(real_df):
                    real_name = real_df.iloc[i]['title']
                    # Try to get additional info
                    real_price = real_df.iloc[i].get('price', simple_row.get('price', 50))
                    real_rating = real_df.iloc[i].get('rating', 0)
                    real_purchases = real_df.iloc[i].get('purchases', 0)
                    
                    self.mapping[simple_id] = {
                        'simple_name': simple_name,
                        'real_name': real_name,
                        'price': real_price,
                        'rating': real_rating,
                        'purchases': real_purchases,
                        'category': real_df.iloc[i].get('type', 'Normal')
                    }
                    
                    if i < 10:  # Show first 10 mappings
                        print(f"     {simple_name} (ID: {simple_id}) -> {real_name}")
                else:
                    # Fallback
                    self.mapping[simple_id] = {
                        'simple_name': simple_name,
                        'real_name': simple_name,
                        'price': simple_row.get('price', 50),
                        'rating': 0,
                        'purchases': 0,
                        'category': simple_row.get('type', 'menu')
                    }
            
            print(f"   Created mapping for {len(self.mapping)} items")
            
        except Exception as e:
            print(f"Error creating item mapping: {e}")
            import traceback
            traceback.print_exc()
    
    def get_real_name(self, item_id):
        """Get real product name for item ID"""
        if item_id in self.mapping:
            return self.mapping[item_id]['real_name']
        return f"Item {item_id}"
    
    def get_display_name(self, item_id):
        """Get display name: Real Name"""
        if item_id in self.mapping:
            mapping = self.mapping[item_id]
            return f"{mapping['real_name']}"
        return f"Menu Item {item_id}"
    
    def get_item_info(self, item_id):
        """Get complete item information"""
        if item_id in self.mapping:
            return self.mapping[item_id]
        return {
            'simple_name': f"Menu Item {item_id}",
            'real_name': f"Menu Item {item_id}",
            'price': 50,
            'rating': 0,
            'purchases': 0,
            'category': 'Unknown'
        }
    
    def apply_to_dataframe(self, df: pd.DataFrame, item_id_column: str = 'item_id') -> pd.DataFrame:
        """Apply real names to any dataframe containing item IDs."""
        if df.empty or item_id_column not in df.columns:
            return df
        
        df_enhanced = df.copy()
        
        # Add real name column
        df_enhanced['item_real_name'] = df_enhanced[item_id_column].apply(
            lambda x: self.get_real_name(x) if pd.notnull(x) else 'Unknown'
        )
        
        # If there's already an item_name column, preserve it as original
        if 'item_name' in df_enhanced.columns:
            df_enhanced['item_original_name'] = df_enhanced['item_name']
        
        # Replace or add item_name with real name
        df_enhanced['item_name'] = df_enhanced[item_id_column].apply(
            lambda x: self.get_display_name(x) if pd.notnull(x) else 'Unknown'
        )
        
        # Add price if not present
        if 'price' not in df_enhanced.columns:
            df_enhanced['item_price'] = df_enhanced[item_id_column].apply(
                lambda x: self.get_item_info(x).get('price', 50) if pd.notnull(x) else 50
            )
        
        return df_enhanced
    
    def apply_to_forecast_df(self, forecast_df, item_id_column='item_id'):
        """Apply real names to forecast DataFrame"""
        if forecast_df.empty or item_id_column not in forecast_df.columns:
            return forecast_df
        
        df = forecast_df.copy()
        
        # Add real name column
        df['item_real_name'] = df[item_id_column].apply(
            lambda x: self.get_real_name(x) if pd.notnull(x) else 'Unknown'
        )
        
        # Replace item_name with real name for display
        if 'item_name' in df.columns:
            df['item_original_name'] = df['item_name']
        df['item_name'] = df[item_id_column].apply(
            lambda x: self.get_display_name(x) if pd.notnull(x) else 'Unknown'
        )
        
        # Add additional info if needed
        df['item_price'] = df[item_id_column].apply(
            lambda x: self.get_item_info(x).get('price', 50) if pd.notnull(x) else 50
        )
        
        return df
    
    def apply_to_inventory_df(self, inventory_df, item_id_column='item_id'):
        """Apply real names to inventory DataFrame"""
        if inventory_df.empty or item_id_column not in inventory_df.columns:
            return inventory_df
        
        df = inventory_df.copy()
        
        # Add real name
        df['item_real_name'] = df[item_id_column].apply(
            lambda x: self.get_real_name(x) if pd.notnull(x) else 'Unknown'
        )
        
        # Replace item_name with real name
        if 'item_name' in df.columns:
            df['item_original_name'] = df['item_name']
        df['item_name'] = df[item_id_column].apply(
            lambda x: self.get_display_name(x) if pd.notnull(x) else 'Unknown'
        )
        
        return df
