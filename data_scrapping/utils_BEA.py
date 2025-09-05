import requests
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output
import math
import polars as pl ## need to harmonize between pandas et polars
from polars import col as d
import textwrap

def show_bea_datasets(api_key):
    """
    Print the list of all DataSetNames (with a short description) available via the BEA API.

    Parameters:
        api_key (str): BEA API key
    """
    base_url = "https://apps.bea.gov/api/data/"
    params = {
        "UserID": api_key,
        "method": "GetDataSetList",
        "ResultFormat": "JSON"
    }

    r = requests.get(base_url, params=params)
    r.raise_for_status()
    meta = r.json()

    if "Results" not in meta["BEAAPI"] or "Dataset" not in meta["BEAAPI"]["Results"]:
        raise ValueError("Unable to retrieve BEA DataSetName")

    datasets_info = meta['BEAAPI']['Results']['Dataset']

    print("Available BEA Datasets:\n")
    for d in datasets_info:
        key = d["DatasetName"]
        desc = d["DatasetDescription"]
        print(f"{key}: {desc}")


def get_bea_regional_tables(api_key):
    """
    Get all available tables in the BEA Regional dataset with a short description for each.

    Parameters:
        api_key (str): BEA API key

    Returns:
        df_tables (pd.DataFrame): df containing the DataSetName (Regional), the TableName and the TableDescription
    """
    base_url = "https://apps.bea.gov/api/data/"

    params = {
        "UserID": api_key,
        "method": "GetParameterValues",
        "DataSetName": 'Regional',
        "ParameterName": 'TableName',
        "ResultFormat": "JSON"
    }

    r = requests.get(base_url, params=params)
    r.raise_for_status()
    meta = r.json()

    if "Results" not in meta["BEAAPI"] or "ParamValue" not in meta["BEAAPI"]["Results"]:
        raise ValueError("Unable to retrieve BEA TableName")

    tables_info = meta["BEAAPI"]["Results"]["ParamValue"]

    rows = []
    for t in tables_info:
        key = t["Key"]
        desc = t["Desc"]

        rows.append({'DataSetName': 'Regional', "TableName": key, "TableDescription": desc})

    df_tables = pd.DataFrame(rows)

    return df_tables


def get_bea_table_linecodes(table_name, api_key):
    """
    Get all available line codes of a table in the BEA Regional dataset with a short description for each line code.

    Parameters:
        table_name (str): the TableName of a table from the BEA Regional dataset
        api_key (str): BEA API key

    Returns:
        df_tables (pd.DataFrame): df containing the TableName, the LineCode and the LineDescription

    """
    base_url = "https://apps.bea.gov/api/data/"
    params = {
        "UserID": api_key,
        "method": "GetParameterValuesFiltered",
        "DataSetName": "Regional",
        "TargetParameter": "LineCode",   
        "TableName": table_name,         
        "ResultFormat": "JSON"
    }

    r = requests.get(base_url, params=params, timeout=60)
    r.raise_for_status()
    meta = r.json()

    results = meta.get("BEAAPI", {}).get("Results", {})
    if "Error" in results:
        raise ValueError(f"BEA API error for {table_name} table")

    values = results.get("ParamValue", [])
    if not values:
        raise ValueError(f"No LineCode found for {table_name} table")

    df_linecodes = pd.DataFrame(
        [{"TableName": table_name,
          "LineCode": str(v["Key"]),
          "LineDescription": v["Desc"]} for v in values]
    )

    return df_linecodes


def explore_dataframe(df: pd.DataFrame, page_size=20):
    """
    Explore a DataFrame interactively with multi-column filters and pagination.
    
    Parameters:
        df (pd.DataFrame): the DataFrame to explore
        page_size (int): number of rows to display per page (default: 20)
    """
    
    filter_columns = df.columns.tolist()

    ## create a text filter widget for each column
    filters = {col: widgets.Text(placeholder=f"Filter {col}…") for col in filter_columns}
    
    ## slider to move between pages
    page_slider = widgets.IntSlider(value=1, min=1, max=1, step=1, description="Page")
    output = widgets.Output()

    def update(_=None):
        """Update the displayed DataFrame based on filters and current page."""
        with output:
            clear_output()
            df_filtered = df.copy()

            ## apply filters on each selected column
            for col, widget in filters.items():
                if widget.value.strip():
                    df_filtered = df_filtered[
                        df_filtered[col].astype(str).str.contains(widget.value, case=False, na=False)
                    ]
            
            ## update pagination values
            n_pages = max(1, math.ceil(len(df_filtered) / page_size))
            page_slider.max = n_pages
            current_page = min(page_slider.value, n_pages)

            start = (current_page - 1) * page_size
            end = start + page_size
            df_page = df_filtered.iloc[start:end]

            ## display the current page of the DataFrame
            display(df_page.style.hide(axis="index"))
            print(f"Page {current_page}/{n_pages} | {len(df_filtered)} rows found")

    ## observe changes in filters and pagination
    for widget in filters.values():
        widget.observe(update, names="value")
    page_slider.observe(update, names="value")

    ## layout: filters on top, slider below, then the output
    filter_box = widgets.HBox(list(filters.values()))
    display(filter_box, page_slider, output)

    ## initial display
    update()


def get_bea_state_data(api_key, table_name, line_codes, metric_note=False):

    """
    Retrieve state-level economic data from the BEA (U.S. Bureau of Economic Analysis) 
    Regional dataset and return it as a Polars DataFrame.

    This function queries the BEA API for the specified table and line codes, 
    processes the results, and consolidates them into a single DataFrame 
    with numeric values scaled according to their unit multipliers.

    Parameters:
        api_key (str): BEA API key.
        table_name (str): BEA Regional table name.
        line_codes (list[int]): List of BEA LineCodes specifying which metrics to extract.
        metric_note (bool): If True, prints the short descriptive note for each metric retrieved. By default False.

    Returns:
        df (pl.DataFrame): A Polars DataFrame with the following columns:
                - STATE (str): State name.  
                - YEAR (str): Year of the observation.  
                - LINE_CODE (int): LineCode identifying the metric.  
                - UNIT_MULT (int): Power-of-10 multiplier provided by BEA.  
                - VALUE (float): Raw reported value.  
                - VALUE_MULT (float): Value times 10**UNIT_MULT

    Notes:
        - This function uses the BEA Regional dataset (DataSetName="Regional").  
        - By default, it retrieves data for all years (Year="ALL") at the 
          state level (GeoFIPS="STATE").  
        - The function prints a confirmation line for each retrieved metric, 
          including the unit of measure.  
    """

    url = "https://apps.bea.gov/api/data/"
    dfs = []

    for line_code in line_codes:
        params = {
            "UserID": api_key,
            "method": "GetData",
            "DataSetName": "Regional",
            "TableName": table_name, ## table from Regional dataset 
            "LineCode": str(line_code), ## the metrics we want to extract
            "GeoFIPS": "STATE", ## geographical granularity
            "Year": 'ALL', #str(year), ## which year
            "ResultFormat": "JSON"
        }

        r = requests.get(url, params=params)
        r.raise_for_status()
        meta = r.json()


        if "Results" not in meta["BEAAPI"] or "Data" not in meta["BEAAPI"]["Results"]:
            raise ValueError(f"Unable to retrieve BEA data with LineCode {line_code} from {table_name} table")

        df_line_code = (pl.DataFrame(meta["BEAAPI"]["Results"]["Data"])
                          .rename({'GeoName':'STATE', 'TimePeriod':'YEAR', 'DataValue':'VALUE'})
                          .with_columns(UNIT_MULT = d.UNIT_MULT.cast(pl.Int64),
                                        VALUE = d.VALUE.cast(pl.Float64),
                                        LINE_CODE = line_code,    
                                        YEAR = d.YEAR.cast(pl.Int32)            
                                       )

                          .with_columns(VALUE_MULT = d.VALUE*(10**d.UNIT_MULT))
                          [['STATE', 'YEAR', 'LINE_CODE', 'UNIT_MULT', 'VALUE', 'VALUE_MULT']]
                        )

        dfs.append(df_line_code)

        print(f"\nLine code {line_code}: {meta['BEAAPI']['Results']['Statistic']} with unit {meta['BEAAPI']['Results']['UnitOfMeasure']}")

        if metric_note:
            note = meta['BEAAPI']['Results']['Notes'][0]['NoteText']
            wrapped_note = textwrap.fill(note.strip(), width=100)
            print("\nShort description:")
            print(wrapped_note)
            
    df = pl.concat(dfs)

    return df
