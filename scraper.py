

import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os

# Load API key and base URL from .env file
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

HEADERS = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
}


# Step 1 - Get list of recent matches from the API
def get_recent_matches():
    print("Fetching recent matches...")
    url = BASE_URL + "/matches/v1/recent"
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print("Error fetching matches:", response.status_code)
        return []

    data = response.json()
    return data.get("typeMatches", [])


# Step 2 - Get the top scorer for a specific match
def get_top_scorer(match_id):
    url = BASE_URL + "/mcenter/v1/" + str(match_id) + "/hscard"

    try:
        response = requests.get(url, headers=HEADERS)

        if response.status_code != 200:
            return "N/A", "N/A"

        data = response.json()
        innings_list = data.get("scorecard", [])

        best_runs = -1
        top_name = "N/A"
        top_score = "N/A"

        # Loop through all innings to find the highest scorer
        for innings in innings_list:
            batsmen = innings.get("batsman", [])
            for player in batsmen:
                runs = player.get("runs", 0)
                if runs is None:
                    runs = 0
                if runs > best_runs:
                    best_runs = runs
                    top_name = player.get("name", "N/A")
                    top_score = runs

        return top_name, top_score

    except Exception as e:
        print("Could not get scorecard for match", match_id, "-", e)
        return "N/A", "N/A"


# Step 3 - Go through all matches and pick only completed ones
def get_completed_matches(type_matches):
    completed = []

    for type_match in type_matches:
        for series_wrap in type_match.get("seriesMatches", []):
            series = series_wrap.get("seriesAdWrapper", {})
            for match in series.get("matches", []):
                info = match.get("matchInfo", {})

                # Skip if match is not completed
                if "complete" not in info.get("state", "").lower():
                    continue

                team1 = info.get("team1", {}).get("teamName", "").strip()
                team2 = info.get("team2", {}).get("teamName", "").strip()

                # Convert date from timestamp to readable format
                raw_date = info.get("startDate", "")
                try:
                    date_str = datetime.fromtimestamp(int(raw_date) / 1000).strftime("%Y-%m-%d")
                except:
                    date_str = str(raw_date)

                # Get venue name
                venue_info = info.get("venueInfo", {})
                venue = venue_info.get("ground", "") or venue_info.get("city", "N/A")

                # Figure out who won from the status text
                status = info.get("status", "")
                if team1.lower() in status.lower():
                    result = team1
                elif team2.lower() in status.lower():
                    result = team2
                else:
                    result = status.strip()

                completed.append({
                    "matchId": info.get("matchId", ""),
                    "date": date_str,
                    "team1": team1,
                    "team2": team2,
                    "venue": venue,
                    "result": result
                })

    print("Total completed matches found:", len(completed))
    return completed[:10]  # Only take first 10 matches


# Main function - runs everything
def main():
    print("=" * 50)
    print("Task 1 - Cricket Match Data Collector")
    print("=" * 50)

    # Get matches
    type_matches = get_recent_matches()
    if not type_matches:
        print("No data received. Please check your API key.")
        return

    matches = get_completed_matches(type_matches)
    if not matches:
        print("No completed matches found.")
        return

    # For each match, fetch the top scorer
    print("\nFetching top scorer for each match...")
    final_data = []

    for match in matches:
        print("  ->", match["team1"], "vs", match["team2"], "(", match["date"], ")")
        top_scorer, top_score = get_top_scorer(match["matchId"])

        final_data.append({
            "date": match["date"],
            "team1": match["team1"],
            "team2": match["team2"],
            "venue": match["venue"],
            "result": match["result"],
            "top_scorer": top_scorer,
            "top_score": top_score
        })

    # Save to CSV file
    df = pd.DataFrame(final_data)
    df.to_csv("match_data.csv", index=False)

    print("\n" + "=" * 50)
    print("Done! Saved", len(df), "matches to match_data.csv")
    print("=" * 50)
    print(df.to_string(index=False))


main()
