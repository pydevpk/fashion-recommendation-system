# import sqlalchemy
from sqlalchemy import text
# import time
# import pandas as pd

# conn = sqlalchemy.create_engine('mysql+pymysql://root:password@localhost:3306/ashi_final_data')
# cur = conn


# LJ Product Inclusion Rule
# def apply_lj_product_rule(attribute_based, data, base_item_id):
#     base_item_prefix = data.loc[data["ITEM_ID"] == base_item_id, "ITEM_CD"].values[0][:2]
#     if base_item_prefix == "LJ":
#         # Include both LJ and non-LJ styles
#         return attribute_based
#     else:
#         # Exclude LJ styles
#         return [item for item in attribute_based if not data.loc[data["ITEM_ID"] == item, "ITEM_CD"].values[0].startswith("LJ")]
    

async def apply_lj_product_rule(attribute_based, conn, base_item_id):
    with conn.connect() as connection:
        # Query to get the base_item_prefix
        query_base_item_prefix = text("""
        SELECT LEFT(ITEM_CD, 2) AS prefix 
        FROM main 
        WHERE ITEM_ID = :base_item_id
        """)
        result = connection.execute(query_base_item_prefix, {"base_item_id": base_item_id}).fetchone()
        base_item_prefix = result[0] if result else None

        if base_item_prefix == "LJ":
            # Include both LJ and non-LJ styles
            return attribute_based
        else:
            # Query to filter out items with LJ prefix
            attribute_based_placeholder = ', '.join([f"'{item}'" for item in attribute_based])
            query_exclude_lj = text(f"""
            SELECT ITEM_ID 
            FROM main 
            WHERE ITEM_ID IN ({attribute_based_placeholder})
              AND NOT LEFT(ITEM_CD, 2) = 'LJ'
            """)
            result = connection.execute(query_exclude_lj).fetchall()
            return [row[0] for row in result]
    

# def apply_lj_product_rule_df(data, base_item_id):
#     base_item_prefix = data.loc[data["ITEM_ID"] == base_item_id, "ITEM_CD"].values[0][:2]
#     if base_item_prefix == "LJ":
#         return data['ITEM_ID'].tolist()
#     else:
#         return [item for item in data['ITEM_ID'].tolist() if not data.loc[data["ITEM_ID"] == item, "ITEM_CD"].values[0].startswith("LJ")]
    

async def apply_lj_product_rule_df(data, base_item_id):
    with data.connect() as connection:
        # Query to get the base_item_prefix
        query_base_item_prefix = text("""
        SELECT LEFT(ITEM_CD, 2) AS prefix 
        FROM main 
        WHERE ITEM_ID = :base_item_id
        """)
        result = connection.execute(query_base_item_prefix, {"base_item_id": base_item_id}).fetchone()
        base_item_prefix = result[0] if result else None

        if base_item_prefix == "LJ":
            # Return all ITEM_IDs
            query_all_items = text("""
            SELECT ITEM_ID 
            FROM main
            """)
            result = connection.execute(query_all_items).fetchall()
            return [row[0] for row in result]
        else:
            # Return ITEM_IDs excluding those with LJ prefix
            query_exclude_lj = text("""
            SELECT ITEM_ID 
            FROM main
            WHERE NOT LEFT(ITEM_CD, 2) = 'LJ'
            """)
            result = connection.execute(query_exclude_lj).fetchall()
            return [row[0] for row in result]



# # Silver/Platinum Product Rule
# def apply_silver_platinum_rule(attribute_based, data, base_item_id):
#     base_metal_karat = data.loc[data["ITEM_ID"] == base_item_id, "METAL_KARAT_DISPLAY"].values[0]
#     if base_metal_karat in ["Silver", "Platinum"]:
#         # Include only Silver/Platinum styles
#         return [item for item in attribute_based if data.loc[data["ITEM_ID"] == item, "METAL_KARAT_DISPLAY"].values[0] in ["Silver", "Platinum"]]
#     else:
#         # Exclude Silver/Platinum styles
#         return [item for item in attribute_based if data.loc[data["ITEM_ID"] == item, "METAL_KARAT_DISPLAY"].values[0] not in ["Silver", "Platinum"]]

async def apply_silver_platinum_rule(attribute_based, data, base_item_id):
    with data.connect() as connection:
        # Query to get the base metal karat
        query_base_metal_karat = text("""
        SELECT METAL_KARAT_DISPLAY 
        FROM main 
        WHERE ITEM_ID = :base_item_id
        """)
        result = connection.execute(query_base_metal_karat, {"base_item_id": base_item_id}).fetchone()
        base_metal_karat = result[0] if result else None

        if base_metal_karat in ["Silver", "Platinum"]:
            # Include only Silver/Platinum styles
            attribute_based_placeholder = ', '.join([f"'{item}'" for item in attribute_based])
            query_silver_platinum = text(f"""
            SELECT ITEM_ID 
            FROM main 
            WHERE ITEM_ID IN ({attribute_based_placeholder})
              AND METAL_KARAT_DISPLAY IN ('Silver', 'Platinum')
            """)
            result = connection.execute(query_silver_platinum).fetchall()
            return [row[0] for row in result]
        else:
            # Exclude Silver/Platinum styles
            attribute_based_placeholder = ', '.join([f"'{item}'" for item in attribute_based])
            query_exclude_silver_platinum = text(f"""
            SELECT ITEM_ID 
            FROM main 
            WHERE ITEM_ID IN ({attribute_based_placeholder})
              AND METAL_KARAT_DISPLAY NOT IN ('Silver', 'Platinum')
            """)
            result = connection.execute(query_exclude_silver_platinum).fetchall()
            return [row[0] for row in result]
    

# def apply_silver_platinum_rule_df(data, base_item_id):
#     base_metal_karat = data.loc[data["ITEM_ID"] == base_item_id, "METAL_KARAT_DISPLAY"].values[0]
#     if base_metal_karat in ["Silver", "Platinum"]:
#         # Include only Silver/Platinum styles
#         return [item for item in data['ITEM_ID'].tolist() if data.loc[data["ITEM_ID"] == item, "METAL_KARAT_DISPLAY"].values[0] in ["Silver", "Platinum"]]
#     else:
#         # Exclude Silver/Platinum styles
#         return [item for item in data['ITEM_ID'].tolist() if data.loc[data["ITEM_ID"] == item, "METAL_KARAT_DISPLAY"].values[0] not in ["Silver", "Platinum"]]

async def apply_silver_platinum_rule_df(data, base_item_id):
    with data.connect() as connection:
        # Query to get the base metal karat
        query_base_metal_karat = text("""
        SELECT METAL_KARAT_DISPLAY 
        FROM main 
        WHERE ITEM_ID = :base_item_id
        """)
        result = connection.execute(query_base_metal_karat, {"base_item_id": base_item_id}).fetchone()
        base_metal_karat = result[0] if result else None

        if base_metal_karat in ["Silver", "Platinum"]:
            # Include only Silver/Platinum styles
            query_silver_platinum = text("""
            SELECT ITEM_ID 
            FROM main 
            WHERE METAL_KARAT_DISPLAY IN ('Silver', 'Platinum')
            """)
            result = connection.execute(query_silver_platinum).fetchall()
            return [row[0] for row in result]
        else:
            # Exclude Silver/Platinum styles
            query_exclude_silver_platinum = text("""
            SELECT ITEM_ID 
            FROM main 
            WHERE METAL_KARAT_DISPLAY NOT IN ('Silver', 'Platinum')
            """)
            result = connection.execute(query_exclude_silver_platinum).fetchall()
            return [row[0] for row in result]
    

# def get_similar_category_style(arr, data, base_item_id):
#     base_category_type = data.loc[data["ITEM_ID"] == base_item_id, "CATEGORY_TYPE"].values[0]
#     base_category_type_list = base_category_type.split(',')
#     attributes = [item for item in data['ITEM_ID'].tolist() if item in arr]
#     return [item for item in attributes if list(set(sorted(data.loc[data["ITEM_ID"] == item, "CATEGORY_TYPE"].values[0].split(',')))) == list(set(sorted(base_category_type_list)))][:20]

async def get_similar_category_style(arr, data, base_item_id):
    with data.connect() as connection:
        # Query to get the base category type
        query_base_category_type = text("""
        SELECT CATEGORY_TYPE
        FROM main
        WHERE ITEM_ID = :base_item_id
        """)
        result = connection.execute(query_base_category_type, {"base_item_id": base_item_id}).fetchone()
        base_category_type = result[0] if result else None

        if not base_category_type:
            return []

        # Process the base category type into a sorted list
        base_category_type_list = sorted(set(base_category_type.split(',')))

        # Query to get ITEM_IDs and CATEGORY_TYPE for items in arr
        query_attributes = text(f"""
        SELECT ITEM_ID, CATEGORY_TYPE
        FROM main
        WHERE ITEM_ID IN :arr
        """)
        result = connection.execute(query_attributes, {"arr": tuple(arr)}).fetchall()

        # Filter attributes based on matching category types
        filtered_items = []
        for item_id, category_type in result:
            item_category_list = sorted(set(category_type.split(',')))
            if item_category_list == base_category_type_list:
                filtered_items.append(item_id)
                if len(filtered_items) == 20:  # Limit to 20 items
                    break

        return filtered_items


# Exact Matching Rule (±20% Price, Metal Color, Metal KT, etc.)
# def apply_exact_matching_rule(attribute_based, data, base_item_id, price_tolerance=0.2, base_properties=["METAL_COLOR", "METAL_KARAT_DISPLAY", "COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"], action="both"):
#     base_price = data.loc[data["ITEM_ID"] == base_item_id, "C_LEVEL_PRICE"].values[0]
#     base_attributes = data.loc[data["ITEM_ID"] == base_item_id, base_properties].values[0]
    
#     f_base_attributes = []
#     for i in base_attributes:
#         i = i.split(',')
#         i = [item.strip() for item in i]
#         f_base_attributes += list(set(sorted(i)))
#     base_attributes = f_base_attributes
    
#     def matches(item):
#         item_price = data.loc[data["ITEM_ID"] == item, "C_LEVEL_PRICE"].values[0]
#         item_attributes = data.loc[data["ITEM_ID"] == item, base_properties].values[0]
#         f_item_attributes = []
#         for i in item_attributes:
#             i = i.split(',')
#             i = [item.strip() for item in i]
#             f_item_attributes += list(set(sorted(i)))
#         item_attributes = f_item_attributes

#         if action == "positive":
#             price_match = base_price <= item_price <= base_price * (1 + price_tolerance)
#         else:
#             price_match = base_price * (1 - price_tolerance) <= item_price <= base_price * (1 + price_tolerance)
#         if price_tolerance > 0:
#             return price_match and all(base_attr == item_attr for base_attr, item_attr in zip(base_attributes, item_attributes))
#         return all(base_attr == item_attr for base_attr, item_attr in zip(base_attributes, item_attributes))
    
#     return [item for item in attribute_based if matches(item)]

async def apply_exact_matching_rule(attribute_based, data, base_item_id, price_tolerance=0.2, base_properties=["METAL_COLOR", "METAL_KARAT_DISPLAY", "COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"], action="both"):
    with data.connect() as connection:
        # Query to get base item price and attributes
        base_query = text(f"""
        SELECT C_LEVEL_PRICE, {", ".join(base_properties)}
        FROM main
        WHERE ITEM_ID = :base_item_id
        """)
        base_result = connection.execute(base_query, {"base_item_id": base_item_id}).fetchone()
        
        if not base_result:
            return []
        
        base_price = base_result[0]
        base_attributes = [
            sorted(set(attr.strip() for attr in (base_result[i] or "").split(',')))
            for i in range(1, len(base_properties) + 1)
        ]

        # Query to get prices and attributes for items in attribute_based
        attribute_query = text(f"""
        SELECT ITEM_ID, C_LEVEL_PRICE, {", ".join(base_properties)}
        FROM main
        WHERE ITEM_ID IN :attribute_based
        """)
        items_result = connection.execute(attribute_query, {"attribute_based": tuple(attribute_based)}).fetchall()
        
        filtered_items = []
        for item_data in items_result:
            item_id = item_data[0]
            item_price = item_data[1]
            item_attributes = [
                sorted(set(attr.strip() for attr in (item_data[i] or "").split(',')))
                for i in range(2, len(base_properties) + 2)
            ]
            
            # Match logic for price
            if action == "positive":
                price_match = base_price <= item_price <= base_price * (1 + price_tolerance)
            else:
                price_match = base_price * (1 - price_tolerance) <= item_price <= base_price * (1 + price_tolerance)
            
            # Match logic for attributes
            attributes_match = all(base_attr == item_attr for base_attr, item_attr in zip(base_attributes, item_attributes))
            
            if (price_tolerance > 0 and price_match and attributes_match) or (price_tolerance == 0 and attributes_match):
                filtered_items.append(item_id)
        
        return filtered_items



# Distinct Styles and Sorting by Best Seller
# def distinct_and_sort_by_best_seller(attribute_based, data, item_id):
#     unique_styles = {}
#     bese = data.loc[data["ITEM_ID"] == item_id, "UNIQUE_ITEM_CD"].values[0]
#     for item in attribute_based:
#         style = data.loc[data["ITEM_ID"] == item, "UNIQUE_ITEM_CD"].values[0]
#         if style not in unique_styles or data.loc[data["ITEM_ID"] == item, "BestSeller_DisplayOrder"].values[0] < data.loc[data["ITEM_ID"] == unique_styles[style], "BestSeller_DisplayOrder"].values[0]:
#             unique_styles[style] = item
#     if bese in unique_styles:
#         unique_styles.pop(bese)
#     # Sort by BestSeller_DisplayOrder
#     sorted_items = sorted(unique_styles.values(), key=lambda x: data.loc[data["ITEM_ID"] == x, "BestSeller_DisplayOrder"].values[0])
#     return sorted_items


async def distinct_and_sort_by_best_seller(attribute_based, data, item_id):
    with data.connect() as connection:
        # Query to get the UNIQUE_ITEM_CD of the base item
        base_query = text("SELECT UNIQUE_ITEM_CD FROM main WHERE ITEM_ID = :item_id")
        base_result = connection.execute(base_query, {"item_id": item_id}).fetchone()
        if not base_result:
            return []
        bese = base_result[0]
        
        # Query to get the required details for items in attribute_based
        attributes_query = text("""
        SELECT ITEM_ID, UNIQUE_ITEM_CD, BestSeller_DisplayOrder
        FROM main
        WHERE ITEM_ID IN :attribute_based
        """)
        items_result = connection.execute(attributes_query, {"attribute_based": tuple(attribute_based)}).fetchall()

        unique_styles = {}
        
        # Process results
        for item_id, unique_item_cd, best_seller_order in items_result:
            if unique_item_cd not in unique_styles or best_seller_order < unique_styles[unique_item_cd][1]:
                unique_styles[unique_item_cd] = (item_id, best_seller_order)
        
        # Remove base item's style
        if bese in unique_styles:
            unique_styles.pop(bese)
        
        # Sort by BestSeller_DisplayOrder
        # sorted_items = sorted([item[0] for item in unique_styles.values()], key=lambda x: unique_styles[connection.execute(text("SELECT UNIQUE_ITEM_CD FROM main WHERE ITEM_ID = :item_id"), x)])
        sorted_items = sorted(
            [item[0] for item in unique_styles.values()],
            key=lambda x: connection.execute(
                text("SELECT BestSeller_DisplayOrder FROM main WHERE ITEM_ID = :item_id"),
                {"item_id": x}
            ).fetchone()[0]
        )
        
        return sorted_items


# Inject Related Style Shapes
# def inject_related_style_shapes(attribute_based, data, base_item_id):
#     base_style = data.loc[data["ITEM_ID"] == base_item_id, "RELATED_STYLE_SHAPES"].values[0]
#     if base_style == "NO INFO":
#         return []
    
#     final_result = []
#     base_style_l = base_style.split(',')
#     base_style_l = [base_style.strip() for base_style in base_style_l]
#     for base_style_i in base_style_l:
#         related_shapes = data[data["ITEM_CD"] == base_style_i].sort_values("BestSeller_DisplayOrder")["ITEM_ID"].tolist()
#         final_result += related_shapes
#     return final_result

async def inject_related_style_shapes(attribute_based, data, base_item_id):
    with data.connect() as connection:
        # Query to get the RELATED_STYLE_SHAPES of the base item
        base_style_query = text("SELECT RELATED_STYLE_SHAPES FROM main WHERE ITEM_ID = :base_item_id")
        base_style_result = connection.execute(base_style_query, {"base_item_id": base_item_id}).fetchone()
        
        if not base_style_result or base_style_result[0] == "NO INFO":
            return []
        
        base_style = base_style_result[0]
        base_style_list = [style.strip() for style in base_style.split(',')]
        
        final_result = []
        
        # For each base style, get related shapes sorted by BestSeller_DisplayOrder
        for base_style_item in base_style_list:
            related_shapes_query = text("""
            SELECT ITEM_ID
            FROM main
            WHERE ITEM_CD = :base_style_item
            ORDER BY BestSeller_DisplayOrder
            """)
            related_shapes_result = connection.execute(related_shapes_query, {"base_style_item": base_style_item}).fetchall()
            
            final_result.extend([item[0] for item in related_shapes_result])
        
        return final_result


# def get_similar_name_styles(attribute_based, data, base_item_id):
#     base_style_name = data.loc[data["ITEM_ID"] == base_item_id, "ITEM_NAME"].values[0]
#     related_shapes = data[(data["ITEM_NAME"] == base_style_name) & (data["ITEM_ID"].isin(attribute_based))]["ITEM_ID"].tolist()
#     return related_shapes[:20]

async def get_similar_name_styles(attribute_based, data, base_item_id):
    with data.connect() as connection:
        # Query to get the ITEM_NAME of the base item
        base_style_query = text("SELECT ITEM_NAME FROM main WHERE ITEM_ID = :base_item_id")
        base_style_result = connection.execute(base_style_query, {"base_item_id": base_item_id}).fetchone()
        
        if not base_style_result:
            return []
        
        base_style_name = base_style_result[0]
        
        # Query to get ITEM_IDs with the same ITEM_NAME as the base style name and in the attribute_based list
        related_shapes_query = text("""
        SELECT ITEM_ID
        FROM main
        WHERE ITEM_NAME = :base_style_name
        AND ITEM_ID IN :attribute_based
        """)
        related_shapes_result = connection.execute(related_shapes_query, {"base_style_name": base_style_name, "attribute_based": tuple(attribute_based)}).fetchall()
        
        related_shapes = [item[0] for item in related_shapes_result]
        
        return related_shapes[:20]


# def remove_base_uid(array, base_item_id, data):
#     print()
#     unique = set()
#     result = []
#     for item in array:
#         style = data.loc[data["ITEM_ID"] == item, "UNIQUE_ITEM_CD"].values[0]
#         if style not in unique:
#             result.append(item)
#             unique.add(style)
#     return result

async def remove_base_uid(array, base_item_id, data):
    with data.connect() as connection:
        # Query to get the UNIQUE_ITEM_CD for the base_item_id
        base_style_query = text("SELECT UNIQUE_ITEM_CD FROM main WHERE ITEM_ID = :base_item_id")
        base_style_result = connection.execute(base_style_query, {"base_item_id": base_item_id}).fetchone()
        
        if not base_style_result:
            return []
        
        base_style = base_style_result[0]
        unique = set()
        result = []
        
        # For each item in the array, get the UNIQUE_ITEM_CD and check for uniqueness
        for item in array:
            style_query = text("SELECT UNIQUE_ITEM_CD FROM main WHERE ITEM_ID = :item_id")
            style_result = connection.execute(style_query, {"item_id": item}).fetchone()
            
            if style_result:
                style = style_result[0]
                if style not in unique:
                    result.append(item)
                    unique.add(style)
        
        return result


# Final Aggregation Combine arrays as per the specified steps:
async def aggregate_arrays(item_id, data, *arrays):
    aggregated = []
    seen = set()
    for array in arrays:
        for item in array:
            if item not in seen:
                aggregated.append(item)
                seen.add(item)
    try:
        aggregated.remove(item_id)
    except:pass
    return await remove_base_uid(aggregated, item_id, data)


async def aggregate_arrays_single(array):
    aggregated = []
    seen = set()
    for item in array:
        if item not in seen:
            aggregated.append(item)
            seen.add(item)
    return aggregated

