# LJ Product Inclusion Rule
def apply_lj_product_rule(attribute_based, data, base_item_id):
    base_item_prefix = data.loc[data["ITEM_ID"] == base_item_id, "ITEM_CD"].values[0][:2]
    if base_item_prefix == "LJ":
        # Include both LJ and non-LJ styles
        return attribute_based
    else:
        # Exclude LJ styles
        return [item for item in attribute_based if not data.loc[data["ITEM_ID"] == item, "ITEM_CD"].values[0].startswith("LJ")]
    

def apply_lj_product_rule_df(data, base_item_id):
    base_item_prefix = data.loc[data["ITEM_ID"] == base_item_id, "ITEM_CD"].values[0][:2]
    if base_item_prefix == "LJ":
        return data['ITEM_ID'].tolist()
    else:
        return [item for item in data['ITEM_ID'].tolist() if not data.loc[data["ITEM_ID"] == item, "ITEM_CD"].values[0].startswith("LJ")]


# Silver/Platinum Product Rule
def apply_silver_platinum_rule(attribute_based, data, base_item_id):
    base_metal_karat = data.loc[data["ITEM_ID"] == base_item_id, "METAL_KARAT_DISPLAY"].values[0]
    if base_metal_karat in ["Silver", "Platinum"]:
        # Include only Silver/Platinum styles
        return [item for item in attribute_based if data.loc[data["ITEM_ID"] == item, "METAL_KARAT_DISPLAY"].values[0] in ["Silver", "Platinum"]]
    else:
        # Exclude Silver/Platinum styles
        return [item for item in attribute_based if data.loc[data["ITEM_ID"] == item, "METAL_KARAT_DISPLAY"].values[0] not in ["Silver", "Platinum"]]
    

def apply_silver_platinum_rule_df(data, base_item_id):
    base_metal_karat = data.loc[data["ITEM_ID"] == base_item_id, "METAL_KARAT_DISPLAY"].values[0]
    if base_metal_karat in ["Silver", "Platinum"]:
        # Include only Silver/Platinum styles
        return [item for item in data['ITEM_ID'].tolist() if data.loc[data["ITEM_ID"] == item, "METAL_KARAT_DISPLAY"].values[0] in ["Silver", "Platinum"]]
    else:
        # Exclude Silver/Platinum styles
        return [item for item in data['ITEM_ID'].tolist() if data.loc[data["ITEM_ID"] == item, "METAL_KARAT_DISPLAY"].values[0] not in ["Silver", "Platinum"]]
    

def get_similar_category_style(arr, data, base_item_id):
    base_category_type = data.loc[data["ITEM_ID"] == base_item_id, "CATEGORY_TYPE"].values[0]
    base_category_type_list = base_category_type.split(',')
    attributes = [item for item in data['ITEM_ID'].tolist() if item in arr]
    return [item for item in attributes if list(set(sorted(data.loc[data["ITEM_ID"] == item, "CATEGORY_TYPE"].values[0].split(',')))) == list(set(sorted(base_category_type_list)))][:20]




# Exact Matching Rule (±20% Price, Metal Color, Metal KT, etc.)
def apply_exact_matching_rule(attribute_based, data, base_item_id, price_tolerance=0.2, base_properties=["METAL_COLOR", "METAL_KARAT_DISPLAY", "COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"], action="both"):
    base_price = data.loc[data["ITEM_ID"] == base_item_id, "C_LEVEL_PRICE"].values[0]
    base_attributes = data.loc[data["ITEM_ID"] == base_item_id, base_properties].values[0]
    
    f_base_attributes = []
    for i in base_attributes:
        i = i.split(',')
        i = [item.strip() for item in i]
        f_base_attributes += list(set(sorted(i)))
    base_attributes = f_base_attributes
    
    def matches(item):
        item_price = data.loc[data["ITEM_ID"] == item, "C_LEVEL_PRICE"].values[0]
        item_attributes = data.loc[data["ITEM_ID"] == item, base_properties].values[0]
        f_item_attributes = []
        for i in item_attributes:
            i = i.split(',')
            i = [item.strip() for item in i]
            f_item_attributes += list(set(sorted(i)))
        item_attributes = f_item_attributes

        if action == "positive":
            price_match = base_price <= item_price <= base_price * (1 + price_tolerance)
        else:
            price_match = base_price * (1 - price_tolerance) <= item_price <= base_price * (1 + price_tolerance)
        if price_tolerance > 0:
            return price_match and all(base_attr == item_attr for base_attr, item_attr in zip(base_attributes, item_attributes))
        return all(base_attr == item_attr for base_attr, item_attr in zip(base_attributes, item_attributes))
    
    return [item for item in attribute_based if matches(item)]


# Distinct Styles and Sorting by Best Seller
def distinct_and_sort_by_best_seller(attribute_based, data):
    unique_styles = {}
    for item in attribute_based:
        style = data.loc[data["ITEM_ID"] == item, "UNIQUE_ITEM_CD"].values[0]
        best = data.loc[data["ITEM_ID"] == item, "BestSeller_DisplayOrder"].values[0]
        if style not in unique_styles or data.loc[data["ITEM_ID"] == item, "BestSeller_DisplayOrder"].values[0] < data.loc[data["ITEM_ID"] == unique_styles[style], "BestSeller_DisplayOrder"].values[0]:
            unique_styles[style] = item
    
    # Sort by BestSeller_DisplayOrder
    sorted_items = sorted(unique_styles.values(), key=lambda x: data.loc[data["ITEM_ID"] == x, "BestSeller_DisplayOrder"].values[0])
    return sorted_items


# Inject Related Style Shapes
def inject_related_style_shapes(attribute_based, data, base_item_id):
    base_style = data.loc[data["ITEM_ID"] == base_item_id, "RELATED_STYLE_SHAPES"].values[0]
    if base_style == "NO INFO":
        return []
    related_shapes = data[data["RELATED_STYLE_SHAPES"] == base_style].sort_values("BestSeller_DisplayOrder")["ITEM_ID"].tolist()
    return related_shapes


def get_similar_name_styles(attribute_based, data, base_item_id):
    base_style_name = data.loc[data["ITEM_ID"] == base_item_id, "ITEM_NAME"].values[0]
    related_shapes = data[(data["ITEM_NAME"] == base_style_name) & (data["ITEM_ID"].isin(attribute_based))]["ITEM_ID"].tolist()
    return related_shapes[:20]

# Final Aggregation Combine arrays as per the specified steps:
def aggregate_arrays(item_id, *arrays):
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
    return aggregated

