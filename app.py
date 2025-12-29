import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from scipy.stats import poisson
import random
import warnings
warnings.filterwarnings("ignore")

# === 設定 ===
LOCAL_DATA_FILE = "football_data.csv"
CACHE_EXPIRY_HOURS = 6

LEAGUES = {
    "Premier League (ENG)": "39",
    "La Liga (ESP)": "140",
    "Bundesliga (GER)": "78",
    "Serie A (ITA)": "135",
    "Ligue 1 (FRA)": "61",
    "Eredivisie (NED)": "88",
    "Primeira Liga (POR)": "94"
}
SEASONS = ["2024", "2023", "2022", "2021"]

# === 工具函數 ===
def is_cache_valid(filepath, expiry_hours=CACHE_EXPIRY_HOURS):
    if not os.path.exists(filepath):
        return False
    mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
    return datetime.now() - mod_time < timedelta(hours=expiry_hours)

def fetch_with_retry(url, headers, params, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"⚠️ Rate limited. Retrying in {wait:.1f}s...")
                import time; time.sleep(wait)
            elif response.status_code == 403:
                raise Exception("API subscription required (403 Forbidden). Please subscribe on RapidAPI.")
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text[:100]}")
        except requests.exceptions.Timeout:
            print(f"⏳ Timeout on attempt {attempt+1}")
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            import time; time.sleep(2 ** attempt)
    raise Exception("Max retries exceeded")

# === 主客場統計 ===
def calculate_team_stats(df, team):
    home_games = df[df['HomeTeam'] == team]
    away_games = df[df['AwayTeam'] == team]
    
    home_goals_for = home_games['FTHG'].mean() if len(home_games) > 0 else 1.2
    home_goals_against = home_games['FTAG'].mean() if len(home_games) > 0 else 1.2
    
    away_goals_for = away_games['FTAG'].mean() if len(away_games) > 0 else 1.0
    away_goals_against = away_games['FTHG'].mean() if len(away_games) > 0 else 1.0
    
    return {
        'home_attack': home_goals_for,
        'home_defense': home_goals_against,
        'away_attack': away_goals_for,
        'away_defense': away_goals_against
    }

def predict_1x2_poisson_advanced(df, home_team, away_team):
    stats_h = calculate_team_stats(df, home_team)
    stats_a = calculate_team_stats(df, away_team)
    
    home_lambda = (stats_h['home_attack'] + stats_a['away_defense']) / 2
    away_lambda = (stats_a['away_attack'] + stats_h['home_defense']) / 2
    
    prob_home = prob_draw = prob_away = 0.0
    for h in range(6):
        for a in range(6):
            p = poisson.pmf(h, home_lambda) * poisson.pmf(a, away_lambda)
            if h > a: prob_home += p
            elif h == a: prob_draw += p
            else: prob_away += p
    return prob_home, prob_draw, prob_away

# === 核心：抓取歷史數據 ===
def manual_fetch_historical_data(league_name: str, season: str):
    if is_cache_valid(LOCAL_DATA_FILE):
        return "✅ 使用緩存數據（6 小時內已更新）"

    league_id = LEAGUES[league_name]
    print(f"\n📥 Fetching {league_name} ({season}) from API-Sports...")
    
    api_key = os.getenv("sport_api")
    if not api_key:
        return "❌ 'sport_api' not found in secrets."

    headers = {
        "X-RapidAPI-Key": api_key.strip(),
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
    }

    try:
        matches_res = fetch_with_retry(
            "https://api-football-v1.p.rapidapi.com/v3/fixtures",
            headers,
            {"league": league_id, "season": season, "last": "30"}
        )
        matches_data = matches_res.get("response", [])
        finished_matches = [m for m in matches_data if m["fixture"]["status"]["short"] == "FT"]
        if not finished_matches:
            return "⚠️ No finished matches found."

        records = []
        fixture_ids = []
        for m in finished_matches:
            home = m["teams"]["home"]["name"]
            away = m["teams"]["away"]["name"]
            fthg = m["goals"]["home"]
            ftag = m["goals"]["away"]
            fid = m["fixture"]["id"]
            fixture_ids.append(fid)
            records.append({
                "League": league_name,
                "Season": season,
                "FixtureID": fid,
                "HomeTeam": home,
                "AwayTeam": away,
                "FTHG": fthg,
                "FTAG": ftag,
                "Home_Odds": None,
                "Draw_Odds": None,
                "Away_Odds": None
            })

        # 抓賠率
        odds_res = fetch_with_retry(
            "https://api-football-v1.p.rapidapi.com/v3/odds",
            headers,
            {"fixture": ",".join(map(str, fixture_ids)), "bookmaker": "8"}
        )
        odds_map = {}
        for o in odds_res.get("response", []):
            fid = o["fixture"]["id"]
            for book in o.get("bookmakers", []):
                if book["id"] == 8:
                    for bet in book.get("bets", []):
                        if bet["name"] == "Match Winner":
                            vmap = {v["value"]: float(v["odd"]) for v in bet["values"]}
                            home_name = o["teams"]["home"]["name"]
                            away_name = o["teams"]["away"]["name"]
                            odds_map[fid] = (
                                vmap.get(home_name, 0),
                                vmap.get("Draw", 0),
                                vmap.get(away_name, 0)
                            )

        final_records = []
        for rec in records:
            fid = rec["FixtureID"]
            if fid in odds_map:
                h, d, a = odds_map[fid]
                if all(x > 0 for x in (h, d, a)):
                    rec.update({"Home_Odds": h, "Draw_Odds": d, "Away_Odds": a})
                    final_records.append(rec)

        if not final_records:
            return "⚠️ No valid odds data found."

        df = pd.DataFrame(final_records)
        df.to_csv(LOCAL_DATA_FILE, index=False, encoding='utf-8')
        return f"✅ 成功抓取 {len(df)} 場比賽（{league_name}, {season}）！"

    except Exception as e:
        msg = str(e)
        if "subscription" in msg:
            return "❌ 請先到 RapidAPI 訂閱 API-Sports（免費）！"
        elif "rate limit" in msg:
            return "⏳ 請稍後再試（API 配額已用盡）"
        else:
            return f"❌ 抓取失敗: {msg}"

# === 回測功能 ===
def run_backtest():
    if not os.path.exists(LOCAL_DATA_FILE):
        return "❌ 請先抓取或上傳歷史數據", None, None
    
    try:
        df = pd.read_csv(LOCAL_DATA_FILE, encoding='utf-8')
        required = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
        if not all(c in df.columns for c in required):
            return "❌ 數據缺少必要欄位（FTHG/FTAG）", None, None
        
        df_test = df.dropna(subset=['FTHG', 'FTAG']).tail(30)
        if len(df_test) == 0:
            return "❌ 無有效比賽數據", None, None

        correct = 0
        acc_list = []
        for i, row in df_test.iterrows():
            ph, pd_, pa = predict_1x2_poisson_advanced(df, row['HomeTeam'], row['AwayTeam'])
            ah, aa = row['FTHG'], row['FTAG']
            pred = np.argmax([ph, pd_, pa])
            actual = 0 if ah > aa else (1 if ah == aa else 2)
            if pred == actual:
                correct += 1
            acc_list.append(correct / (i + 1))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(acc_list)+1), acc_list, marker='o', color='#1f77b4')
        ax.set_title('Poisson Model Rolling Accuracy (Last 30 Matches)')
        ax.set_xlabel('Match Index')
        ax.set_ylabel('Cumulative Accuracy')
        ax.grid(True, alpha=0.5)
        plt.tight_layout()

        summary = f"🎯 最終準確率: {acc_list[-1]:.2%} ({correct}/{len(df_test)})"
        detail_df = pd.DataFrame({
            'Match': df_test['HomeTeam'] + ' vs ' + df_test['AwayTeam'],
            'Result': df_test['FTHG'].astype(str) + '-' + df_test['FTAG'].astype(str),
            'Accuracy': [f"{a:.1%}" for a in acc_list]
        })
        return summary, detail_df, fig

    except Exception as e:
        return f"❌ 回測錯誤: {str(e)}", None, None

# === 模型比較圖 ===
def compare_models(home_team, away_team, eu_odds_str):
    if not os.path.exists(LOCAL_DATA_FILE):
        return plt.figure(), "❌ 請先載入歷史數據"
    try:
        df = pd.read_csv(LOCAL_DATA_FILE)
        eu_odds = list(map(float, eu_odds_str.split(',')))
        if len(eu_odds) != 3:
            raise ValueError("請輸入三個賠率（主,平,客）")

        p_poisson = predict_1x2_poisson_advanced(df, home_team, away_team)
        implied = [1/o for o in eu_odds]
        total_imp = sum(implied)
        implied = [p/total_imp for p in implied]

        labels = ['Home Win', 'Draw', 'Away Win']
        x = np.arange(len(labels))
        width = 0.25

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - width, p_poisson, width, label='Poisson (Advanced)', color='#1f77b4')
        ax.bar(x, implied, width, label='Market', color='#d62728')

        ax.set_ylabel('Probability')
        ax.set_title(f'{home_team} vs {away_team} — Model Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        plt.tight_layout()
        return fig, ""
    except Exception as e:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
        return fig, ""

# === 上傳自訂數據 ===
def upload_custom_data(file_obj):
    try:
        df = pd.read_csv(file_obj.name, encoding='utf-8')
        required_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
        if not all(col in df.columns for col in required_cols):
            missing = set(required_cols) - set(df.columns)
            return f"❌ 缺少必要欄位: {missing}", None
        
        df.to_csv(LOCAL_DATA_FILE, index=False, encoding='utf-8')
        return f"✅ 成功上傳 {len(df)} 場比賽！", df.head(10)
    except Exception as e:
        return f"❌ 上傳失敗: {str(e)}", None

# === Gradio UI ===
with gr.Blocks(title="Football Value Betting Analysis System") as app:
    
    @app.load
    def init_app():
        print("✅ Application loaded.")

    with gr.Tab("📊 Historical Backtest"):
        btn = gr.Button("🔄 執行 Poisson 模型回測")
        result_txt = gr.Textbox(label="回測結果", lines=2)
        result_table = gr.Dataframe(label="詳細結果")
        result_plot = gr.Plot(label="準確率趨勢")
        btn.click(run_backtest, outputs=[result_txt, result_table, result_plot])
    
    with gr.Tab("🔍 模型比較"):
        gr.Markdown("### 輸入一場比賽，比較 Poisson 與市場賠率")
        home_in = gr.Textbox(label="主隊", value="Manchester City")
        away_in = gr.Textbox(label="客隊", value="Arsenal")
        eu_odds_in = gr.Textbox(label="歐洲賠率 (主,平,客)", value="1.72,3.90,4.60")
        compare_btn = gr.Button("📊 生成比較圖")
        model_fig = gr.Plot()
        error_msg = gr.Textbox(label="狀態", interactive=False)
        compare_btn.click(
            lambda h,a,o: compare_models(h,a,o)[:1][0],
            inputs=[home_in, away_in, eu_odds_in],
            outputs=model_fig
        )
    
    with gr.Tab("🔧 抓取官方數據"):
        gr.Markdown("### 從 API-Sports 抓取歷史比賽（需訂閱）")
        with gr.Row():
            league_dropdown = gr.Dropdown(choices=list(LEAGUES.keys()), value="Premier League (ENG)", label="聯賽")
            season_dropdown = gr.Dropdown(choices=SEASONS, value="2024", label="賽季")
        fetch_btn = gr.Button("📥 抓取歷史數據")
        fetch_output = gr.Textbox(label="結果", interactive=False, lines=3)
        fetch_btn.click(manual_fetch_historical_data, inputs=[league_dropdown, season_dropdown], outputs=fetch_output)
    
    with gr.Tab("📤 上傳自訂數據"):
        gr.Markdown("### 上傳你自己的 CSV（需包含：HomeTeam, AwayTeam, FTHG, FTAG）")
        upload_file = gr.File(label="選擇 CSV 檔案", file_types=[".csv"])
        upload_btn = gr.Button("⬆️ 上傳並覆蓋本地數據")
        upload_result = gr.Textbox(label="結果", interactive=False)
        upload_preview = gr.Dataframe(label="預覽前 10 行")
        upload_btn.click(upload_custom_data, inputs=upload_file, outputs=[upload_result, upload_preview])

if __name__ == "__main__":
    app.launch()
