# fetch_daily.py —— 自動抓取最近 7 天所有聯賽的比賽
import os
import pandas as pd
import requests
from datetime import datetime, timedelta

LEAGUES = {
    "Premier League (ENG)": "39",
    "La Liga (ESP)": "140",
    "Bundesliga (GER)": "78",
    "Serie A (ITA)": "135",
    "Ligue 1 (FRA)": "61"
}
SEASON = "2024"
LOCAL_FILE = "football_data.csv"

def fetch_matches(league_id, league_name):
    api_key = os.getenv("sport_api")
    if not api_key:
        raise ValueError("Missing RAPID_API_KEY")

    headers = {
        "X-RapidAPI-Key": api_key.strip(),
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
    }

    # 抓最近 7 天已完成的比賽
    from_date = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")
    to_date = datetime.today().strftime("%Y-%m-%d")

    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
    params = {
        "league": league_id,
        "season": SEASON,
        "from": from_date,
        "to": to_date
    }

    try:
        res = requests.get(url, headers=headers, params=params, timeout=15)
        if res.status_code != 200:
            print(f"⚠️ {league_name} API error:", res.status_code)
            return []

        matches = []
        for m in res.json().get("response", []):
            if m["fixture"]["status"]["short"] == "FT":
                home = m["teams"]["home"]["name"]
                away = m["teams"]["away"]["name"]
                fthg = m["goals"]["home"]
                ftag = m["goals"]["away"]
                if fthg is None or ftag is None:
                    continue

                # 抓賠率
                odds = {"Home_Odds": None, "Draw_Odds": None, "Away_Odds": None}
                if "odds" in m and m["odds"]:
                    for book in m["odds"]:
                        if book.get("bookmaker", {}).get("id") == 8:  # Bet365
                            for bet in book.get("bets", []):
                                if bet["name"] == "Match Winner":
                                    vmap = {v["value"]: float(v["odd"]) for v in bet["values"]}
                                    odds["Home_Odds"] = vmap.get(home)
                                    odds["Draw_Odds"] = vmap.get("Draw")
                                    odds["Away_Odds"] = vmap.get(away)

                matches.append({
                    "League": league_name,
                    "Season": SEASON,
                    "HomeTeam": home,
                    "AwayTeam": away,
                    "FTHG": fthg,
                    "FTAG": ftag,
                    **odds
                })
        return matches
    except Exception as e:
        print(f"❌ Error fetching {league_name}: {e}")
        return []

def main():
    all_matches = []
    for name, lid in LEAGUES.items():
        print(f"📥 Fetching {name}...")
        all_matches.extend(fetch_matches(lid, name))

    if not all_matches:
        print("ℹ️ No new matches found.")
        return

    df_new = pd.DataFrame(all_matches)
    
    # 若已有舊數據，合併並去重（以 Fixture 為準）
    if os.path.exists(LOCAL_FILE):
        df_old = pd.read_csv(LOCAL_FILE)
        df_combined = pd.concat([df_old, df_new]).drop_duplicates(
            subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"], keep="last"
        )
    else:
        df_combined = df_new

    df_combined.to_csv(LOCAL_FILE, index=False, encoding='utf-8')
    print(f"✅ Saved {len(df_combined)} total matches to {LOCAL_FILE}")

if __name__ == "__main__":
    main()
