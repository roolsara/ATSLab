import polars as pl   ## Polars is a fast DataFrame library (like pandas but faster)
import requests       ## To make HTTP requests to the Google Places API
import time           ## To add pauses between requests (avoid hitting API too fast)

#########################
# CONFIGURATION
#########################

## Personal Google API key (to replace)
API_KEY = "replace_here"

## Input CSV file (must contain columns: APT_CODE and APT_NAME)
INPUT_FILE = "apt_scrapping_data.csv"

## Output CSV file where results will be saved
OUTPUT_FILE = "output_airport_ratings.csv"


#########################
# HELPER FUNCTIONS
#########################

def get_place_id(query: str) -> str | None:
    """
    Use Google Places 'Find Place' API to search for an airport by name or code.
    
    Parameters:
        query (str): The search text (e.g. 'CDG Airport' or 'Paris Charles de Gaulle Airport')
    
    Returns:
        str: The Google 'place_id' (unique identifier for that location), 
             or None if not found.
    """
    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {
        "input": query,           ## the text we are searching
        "inputtype": "textquery", ## tells API that we are passing a text string
        "fields": "place_id",     ## we only want the place_id
        "key": API_KEY            ## personal API key
    }
    ## Send the HTTP GET request to Google API
    resp = requests.get(url, params=params).json()
    
    ## If Google found candidates (possible matches)
    if resp.get("candidates"):
        return resp["candidates"][0]["place_id"]  ## return the first candidate's place_id
    return None  ## return None if nothing found


def get_airport_details(place_id: str) -> dict:
    """
    Use Google Places 'Details' API to fetch airport information (rating, reviews).
    
    Parameters:
        place_id (str): The Google unique identifier for the place
    
    Returns:
        dict: A dictionary with 'name', 'rating', and 'user_ratings_total'
              (or empty if request fails)
    """
    url_details = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,                        ## the unique Google identifier
        "fields": "name,rating,user_ratings_total",  ## we only request what we need (name, rating and user ratings total)
        "key": API_KEY                               ## personal API key
    }
    ## Send request to API
    resp = requests.get(url_details, params=params).json()
    
    ## Extract the "result" part of the response
    return resp.get("result", {})


#########################
# MAIN SCRIPT
#########################

def main():
    ## Load the CSV file into a Polars DataFrame
    df = pl.read_csv(INPUT_FILE)

    ## List to collect results (later converted to DataFrame)
    results = []

    ## Count how many airports we need to process
    total = len(df)

    ## Loop through each row of the DataFrame
    for i, row in enumerate(df.iter_rows(named=True), start=1):
        code, name = row["APT_CODE"], row["APT_NAME"]

        ## First try with "CODE Airport"
        query = f"{code} Airport"
        print(f"[{i}/{total}] Searching: {query}")

        place_id = get_place_id(query)

        ## If not found, use APT_NAME directly
        if not place_id:
            print(f"Not found with code. Retrying with full name: {name}")
            place_id = get_place_id(name)

        ## If still not found, save None
        if not place_id:
            results.append({"APT_CODE": code, "APT_NAME": name, "rating": None, "reviews": None})
            continue

        ## Fetch details using the place_id (rating, number of reviews)
        details = get_airport_details(place_id)

        ## Save the result
        results.append({
            "APT_CODE": code,                          ## airport code from input CSV
            "APT_NAME": name,                          ## original name from CSV
            "GOOGLE_NAME": details.get("name"),        ## Google's official place name (can be None)
            "rating": details.get("rating"),           ## average rating (float)
            "reviews": details.get("user_ratings_total") ## number of reviews (int)
        })


        ## Pause for 0.2 seconds to avoid hitting Google API rate limits (to stay safe)
        # time.sleep(0.2)

    ## Convert results to Polars DataFrame
    out_df = pl.DataFrame(results)

    ## Save results to CSV
    out_df.write_csv(OUTPUT_FILE)
    print(f"\n Finished! Results saved in {OUTPUT_FILE}")


#########################
# RUN
#########################
if __name__ == "__main__":
    main()