"""
CineMatch 2.0 - Movie Recommendation System
-------------------------------------------
Content-based movie recommender built with Python, Pandas, scikit-learn
and Streamlit.

Core features
  * Type a movie           -> similar movies (genre, director, cast, language)
  * Type genre/actor/director/language -> matching movies
  * Platform + official watch links for every movie

New in 2.0
  1. Watchlist             - save movies, stored in watchlist.json
  2. Mood-based picks      - choose a mood, get matching movies
  3. Multi-movie recommend - pick 2-5 movies you love, get one combined list
  4. Compare two movies    - similarity score + side-by-side table
  5. Surprise Me           - random movie that fits your filters

Run:
    python -m pip install -r requirements.txt
    python -m streamlit run app.py
"""

import os
import json
import html
import random
import difflib
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------
# 1. PAGE CONFIG + THEME
# ----------------------------------------------------------------------
st.set_page_config(page_title="CineMatch 2.0 | Movie Recommender",
                   page_icon="🎬", layout="wide")

CUSTOM_CSS = """
<style>
.stApp {
    background: radial-gradient(circle at top left, #1b1038 0%, #0b0d17 45%, #070810 100%);
    color: #e8e8f0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #14162b 0%, #0c0e1c 100%);
    border-right: 1px solid #262a4a;
}
.hero {
    padding: 28px 32px; border-radius: 20px;
    background: linear-gradient(120deg, #ff3d6e 0%, #8e2de2 55%, #4a00e0 100%);
    box-shadow: 0 10px 40px rgba(142, 45, 226, 0.35); margin-bottom: 22px;
}
.hero h1 { margin: 0; font-size: 2.4rem; color: #fff; }
.hero p  { margin: 6px 0 0 0; color: #f3e8ff; font-size: 1.05rem; }
.card {
    background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 16px; padding: 18px 20px; margin-top: 14px;
    transition: transform .15s ease, border-color .15s ease;
}
.card:hover { transform: translateY(-2px); border-color: #ff3d6e; }
.card-top { display: flex; justify-content: space-between; align-items: center; gap: 10px; }
.title { font-size: 1.25rem; font-weight: 700; color: #ffffff; }
.year { color: #a9a9c4; font-weight: 400; font-size: 1rem; }
.rating { background: #ffc107; color: #1a1a1a; padding: 3px 11px; border-radius: 999px;
          font-weight: 700; font-size: .9rem; white-space: nowrap; }
.chip { display: inline-block; margin: 8px 6px 4px 0; padding: 2px 11px; border-radius: 999px;
        font-size: .78rem; background: rgba(142, 45, 226, 0.25); border: 1px solid #8e2de2; color: #e9d5ff; }
.meta { margin-top: 6px; font-size: .92rem; color: #cfd0e6; }
.meta b { color: #ff8fab; }
.overview { margin-top: 8px; font-size: .92rem; color: #b8b9d3; font-style: italic; }
.badge { display: inline-block; margin-top: 10px; margin-right: 8px; padding: 4px 13px;
         border-radius: 8px; font-weight: 700; font-size: .85rem; color: #fff; }
.match { display: inline-block; margin-top: 10px; padding: 4px 13px; border-radius: 8px; font-size: .82rem;
         background: rgba(0, 230, 118, 0.15); color: #69f0ae; border: 1px solid #00e676; }
.links { margin-top: 12px; }
a.btn { display: inline-block; margin: 0 8px 6px 0; padding: 6px 14px; border-radius: 8px;
        font-size: .85rem; font-weight: 600; text-decoration: none !important;
        color: #ffffff !important; background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.25); }
a.btn:hover { background: #ff3d6e; border-color: #ff3d6e; }
a.btn.primary { background: linear-gradient(90deg, #ff3d6e, #8e2de2); border: none; }
.section-title { font-size: 1.4rem; font-weight: 700; margin: 18px 0 2px 0; color: #fff; }
.small-note { color: #9a9bb8; font-size: .85rem; }
div[data-testid="stMetric"] { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
                              padding: 14px; border-radius: 14px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# 2. CONSTANTS
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "movie.csv")
WATCHLIST_PATH = os.path.join(BASE_DIR, "watchlist.json")

PLATFORM_SEARCH = {
    "Netflix": "https://www.netflix.com/search?q=",
    "Prime Video": "https://www.primevideo.com/search/?phrase=",
    "JioHotstar": "https://www.hotstar.com/in/search?q=",
    "ZEE5": "https://www.zee5.com/search?q=",
    "SonyLIV": "https://www.sonyliv.com/search?q=",
}
PLATFORM_COLORS = {
    "Netflix": "#e50914", "Prime Video": "#00a8e1", "JioHotstar": "#1f3b8c",
    "ZEE5": "#8230c6", "SonyLIV": "#2b2b2b",
}

# Mood -> genres that fit that mood
MOODS = {
    "😄 Happy / Feel-good": ["Comedy", "Family", "Musical", "Animation"],
    "🔥 Need Adrenaline": ["Action", "Thriller", "Crime", "Adventure"],
    "💖 Romantic": ["Romance", "Musical", "Drama"],
    "😢 Emotional": ["Drama", "Family", "Biography"],
    "😱 Want to be Scared": ["Horror", "Mystery", "Thriller"],
    "🧠 Mind-Bending": ["Sci-Fi", "Mystery", "Thriller", "Fantasy"],
    "💪 Inspired / Motivated": ["Sports", "Biography", "History", "Drama"],
    "🌍 Adventurous": ["Adventure", "Fantasy", "Sci-Fi", "Action"],
}

SORT_OPTIONS = ["Rating (high to low)", "Year (newest first)", "Title (A-Z)"]


# ----------------------------------------------------------------------
# 3. DATA LOADING + WATCHLIST STORAGE
# ----------------------------------------------------------------------
def make_token(text):
    """'Christopher Nolan' -> 'christophernolan' so a name stays ONE token."""
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH).fillna("")
    df["genre_list"] = df["genre"].apply(lambda x: [g.strip() for g in x.split("|") if g.strip()])
    df["director_list"] = df["director"].apply(lambda x: [d.strip() for d in x.split("|") if d.strip()])
    df["cast_list"] = df["cast"].apply(lambda x: [c.strip() for c in x.split("|") if c.strip()])
    df["language"] = df["language"].str.strip()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    return df.reset_index(drop=True)


def load_watchlist():
    try:
        with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [t for t in data if isinstance(t, str)]
    except (OSError, ValueError):
        return []


def save_watchlist(titles):
    try:
        with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
            json.dump(titles, f, indent=2)
    except OSError:
        pass


def toggle_watchlist(title):
    wl = st.session_state.watchlist
    if title in wl:
        wl.remove(title)
    else:
        wl.append(title)
    save_watchlist(wl)


def clear_watchlist():
    st.session_state.watchlist = []
    save_watchlist([])


# ----------------------------------------------------------------------
# 4. RECOMMENDATION ENGINE (TF-IDF + cosine similarity)
# ----------------------------------------------------------------------
@st.cache_data
def build_similarity_matrix(df):
    """
    Each movie becomes a 'soup' of tokens. Repeating a token gives it more
    weight: genre x3, director x3, cast x2, language x2.
    """
    soups = []
    for _, r in df.iterrows():
        genres = [make_token(g) for g in r["genre_list"]] * 3
        directors = [make_token(d) for d in r["director_list"]] * 3
        cast = [make_token(c) for c in r["cast_list"]] * 2
        language = [make_token(r["language"])] * 2
        soups.append(" ".join(genres + directors + cast + language))
    matrix = TfidfVectorizer().fit_transform(soups)
    return cosine_similarity(matrix, matrix)


def explain_match(base, other):
    """Readable reason why two movies are similar."""
    reasons = []
    g = [x for x in base["genre_list"] if x in other["genre_list"]]
    d = [x for x in base["director_list"] if x in other["director_list"]]
    c = [x for x in base["cast_list"] if x in other["cast_list"]]
    if g:
        reasons.append("Genre: " + ", ".join(g))
    if d:
        reasons.append("Director: " + ", ".join(d))
    if c:
        reasons.append("Cast: " + ", ".join(c))
    if base["language"] == other["language"]:
        reasons.append("Language: " + base["language"])
    return "  |  ".join(reasons) if reasons else "Similar overall profile"


def recommend(df, sim, movie_idx, allowed_idx, top_n):
    """Movies most similar to ONE movie."""
    ranked = sorted(enumerate(sim[movie_idx]), key=lambda x: x[1], reverse=True)
    base = df.iloc[movie_idx]
    results = []
    for i, score in ranked:
        if i == movie_idx or i not in allowed_idx or score <= 0:
            continue
        results.append((df.iloc[i], score, explain_match(base, df.iloc[i])))
        if len(results) == top_n:
            break
    return results


def recommend_multi(df, sim, selected_idx, allowed_idx, top_n):
    """Movies most similar to SEVERAL movies at once (average similarity)."""
    avg_scores = sim[selected_idx].mean(axis=0)
    order = np.argsort(avg_scores)[::-1]
    results = []
    for i in order:
        i = int(i)
        if i in selected_idx or i not in allowed_idx or avg_scores[i] <= 0:
            continue
        closest = max(selected_idx, key=lambda s: sim[s][i])      # which liked movie it resembles most
        why = f"Closest to {df.iloc[closest]['title']}  |  " + explain_match(df.iloc[closest], df.iloc[i])
        results.append((df.iloc[i], float(avg_scores[i]), why))
        if len(results) == top_n:
            break
    return results


def mood_picks(df_allowed, mood, top_n):
    """Movies whose genres overlap with the chosen mood."""
    mood_genres = set(MOODS[mood])
    scored = []
    for _, r in df_allowed.iterrows():
        shared = [g for g in r["genre_list"] if g in mood_genres]
        if shared:
            scored.append((len(shared), r["rating"], r, shared))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [(r, shared) for _, _, r, shared in scored[:top_n]]


# ----------------------------------------------------------------------
# 5. SEARCH HELPERS
# ----------------------------------------------------------------------
def find_movie(df, query):
    """Return the index of a movie if the query looks like a title, else None."""
    q = query.strip().lower()
    titles = df["title"].str.lower().tolist()
    if q in titles:
        return titles.index(q)
    close = difflib.get_close_matches(q, titles, n=1, cutoff=0.8)     # typo tolerant
    if close:
        return titles.index(close[0])
    partial = [i for i, t in enumerate(titles) if q in t]
    if len(partial) == 1 and len(q) >= 3:
        return partial[0]
    return None


def search_movies(df, query):
    """Every word must appear somewhere in title/genre/director/cast/language/platform/year."""
    words = query.lower().split()
    rows = []
    for _, r in df.iterrows():
        blob = " ".join([r["title"], r["genre"].replace("|", " "), r["director"].replace("|", " "),
                         r["cast"].replace("|", " "), r["language"], r["platform"], str(r["year"])]).lower()
        if all(w in blob for w in words):
            rows.append(r)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=df.columns)


def matched_fields(row, query):
    words = query.lower().split()
    fields = {
        "Title": row["title"], "Genre": row["genre"].replace("|", " "),
        "Director": row["director"].replace("|", " "), "Cast": row["cast"].replace("|", " "),
        "Language": row["language"], "Platform": row["platform"],
    }
    hit = [name for name, val in fields.items() if any(w in val.lower() for w in words)]
    return "Matched on: " + ", ".join(hit) if hit else ""


def suggestions(df, query, n=5):
    return difflib.get_close_matches(query.lower(), df["title"].str.lower().tolist(), n=n, cutoff=0.4)


def sort_df(d, how):
    if d.empty:
        return d
    if how == SORT_OPTIONS[1]:
        return d.sort_values(["year", "rating"], ascending=False)
    if how == SORT_OPTIONS[2]:
        return d.sort_values("title")
    return d.sort_values("rating", ascending=False)


# ----------------------------------------------------------------------
# 6. UI COMPONENTS
# ----------------------------------------------------------------------
def watch_links(row):
    q = quote_plus(row["title"])
    official = PLATFORM_SEARCH.get(row["platform"], "https://www.justwatch.com/in/search?q=") + q
    justwatch = "https://www.justwatch.com/in/search?q=" + q
    trailer = "https://www.youtube.com/results?search_query=" + quote_plus(
        f"{row['title']} {row['year']} official trailer")
    return official, justwatch, trailer


def render_card(row, key_prefix, extra_text=None, match_text=None):
    """One movie card + watchlist button. key_prefix keeps button keys unique."""
    e = html.escape
    chips = "".join(f'<span class="chip">{e(g)}</span>' for g in row["genre_list"])
    color = PLATFORM_COLORS.get(row["platform"], "#555555")
    official, justwatch, trailer = watch_links(row)

    parts = [
        '<div class="card">',
        '<div class="card-top">',
        f'<div class="title">{e(row["title"])} <span class="year">({row["year"]})</span></div>',
        f'<div class="rating">⭐ {row["rating"]}</div>',
        '</div>',
        f'<div>{chips}</div>',
        f'<div class="meta"><b>🎬 Director:</b> {e(", ".join(row["director_list"]))}</div>',
        f'<div class="meta"><b>👥 Cast:</b> {e(", ".join(row["cast_list"]))}</div>',
        f'<div class="meta"><b>🗣️ Language:</b> {e(row["language"])}</div>',
        f'<div class="overview">{e(row["overview"])}</div>',
        f'<span class="badge" style="background:{color}">📺 {e(row["platform"])}</span>',
    ]
    if extra_text:
        parts.append(f'<span class="match">{e(extra_text)}</span>')
    if match_text:
        parts.append(f'<span class="match">{e(match_text)}</span>')
    parts.append(
        '<div class="links">'
        f'<a class="btn primary" href="{e(official)}" target="_blank">▶ Watch on {e(row["platform"])}</a>'
        f'<a class="btn" href="{e(justwatch)}" target="_blank">🔎 Other legal options</a>'
        f'<a class="btn" href="{e(trailer)}" target="_blank">🎞 Trailer</a>'
        '</div></div>'
    )
    st.markdown("".join(parts), unsafe_allow_html=True)

    in_list = row["title"] in st.session_state.watchlist
    st.button("✅ In Watchlist (click to remove)" if in_list else "➕ Add to Watchlist",
              key=f"{key_prefix}_wl_{row['id']}",
              on_click=toggle_watchlist, args=(row["title"],))


def show_results(rows_df, key_prefix, query=""):
    for _, r in rows_df.iterrows():
        render_card(r, key_prefix, match_text=matched_fields(r, query) if query else None)


def show_recs(recs, key_prefix):
    if not recs:
        st.warning("No matching movies found with the current filters. Try relaxing them.")
    for row, score, why in recs:
        render_card(row, key_prefix, extra_text=f"{score * 100:.0f}% match  •  {why}")


def section(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------
# 7. MAIN APP
# ----------------------------------------------------------------------
def main():
    if not os.path.exists(CSV_PATH):
        st.error(f"movie.csv not found at: {CSV_PATH}")
        st.stop()

    df = load_data()
    sim = build_similarity_matrix(df)
    all_titles = sorted(df["title"].tolist())
    title_to_idx = {t: i for i, t in enumerate(df["title"])}

    # Watchlist is loaded once per session from watchlist.json
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = [t for t in load_watchlist() if t in title_to_idx]

    # ---------------- Hero ----------------
    st.markdown(
        '<div class="hero"><h1>🎬 CineMatch 2.0</h1>'
        '<p>Search, mood picks, multi-movie recommendations, comparisons and a personal watchlist.</p></div>',
        unsafe_allow_html=True,
    )

    # ---------------- Sidebar ----------------
    st.sidebar.markdown("## ⚙️ Filters")
    languages = sorted(df["language"].unique())
    platforms = sorted(df["platform"].unique())
    sel_lang = st.sidebar.multiselect("Language", languages, default=languages)
    sel_plat = st.sidebar.multiselect("Platform", platforms, default=platforms)
    min_rating = st.sidebar.slider("Minimum rating", 0.0, 10.0, 0.0, 0.1)
    top_n = st.sidebar.slider("Number of results", 3, 15, 6)
    sort_by = st.sidebar.selectbox("Sort lists by", SORT_OPTIONS)
    st.sidebar.markdown("---")
    st.sidebar.metric("📌 Watchlist", f"{len(st.session_state.watchlist)} movies")
    st.sidebar.markdown(
        "<span class='small-note'>Platform data can change over time. Use 'Other legal options' "
        "to check where a movie is streaming today.</span>", unsafe_allow_html=True)

    mask = (df["language"].isin(sel_lang) & df["platform"].isin(sel_plat) & (df["rating"] >= min_rating))
    filtered_df = df[mask]
    allowed_idx = set(filtered_df.index)

    # ---------------- Tabs ----------------
    (tab_search, tab_mood, tab_multi, tab_cmp, tab_surprise,
     tab_watch, tab_browse, tab_stats, tab_about) = st.tabs([
        "🔍 Search", "🎭 Mood", "🎯 Multi-Movie", "⚖️ Compare", "🎲 Surprise Me",
        "📌 Watchlist", "🗂️ Browse", "📊 Dashboard", "ℹ️ About"])

    # ===== TAB 1 : SMART SEARCH =====
    with tab_search:
        query = st.text_input(
            "Search", label_visibility="collapsed",
            placeholder="Try: Inception | Christopher Nolan | Tom Cruise | Sci-Fi | Telugu | Aamir Khan comedy")

        if not query.strip():
            st.info("👆 Type a **movie name** to get similar movies, or a **genre / actor / "
                    "director / language** to see matching movies.")
            section("🔥 Top rated picks")
            show_results(filtered_df.sort_values("rating", ascending=False).head(top_n), "s_top")
        else:
            movie_idx = find_movie(df, query)
            if movie_idx is not None:
                section("🎥 You searched for")
                render_card(df.iloc[movie_idx], "s_base")
                section("✨ Movies you may also like")
                show_recs(recommend(df, sim, movie_idx, allowed_idx, top_n), "s_rec")
            else:
                results = sort_df(search_movies(filtered_df, query), sort_by)
                if results.empty:
                    st.warning(f"No results for '{query}'.")
                    sug = suggestions(df, query)
                    if sug:
                        st.write("Did you mean: " + ", ".join(f"**{s.title()}**" for s in sug) + " ?")
                else:
                    section(f'Found {len(results)} movie(s) for "{html.escape(query)}"')
                    show_results(results, "s_res", query)

    # ===== TAB 2 : MOOD-BASED PICKS =====
    with tab_mood:
        section("🎭 How are you feeling today?")
        mood = st.radio("Mood", list(MOODS.keys()), horizontal=True, label_visibility="collapsed")
        st.caption("Genres for this mood: " + ", ".join(MOODS[mood]))
        picks = mood_picks(filtered_df, mood, top_n)
        if not picks:
            st.warning("No movies match this mood with the current filters.")
        for row, shared in picks:
            render_card(row, "mood", extra_text="Mood match: " + ", ".join(shared))

    # ===== TAB 3 : MULTI-MOVIE RECOMMENDATION =====
    with tab_multi:
        section("🎯 Pick 2 to 5 movies you love")
        st.caption("The system averages their profiles and finds movies that fit ALL of them.")
        chosen = st.multiselect("Your favourite movies", all_titles, max_selections=5, key="multi_sel")
        if len(chosen) < 2:
            st.info("Select at least 2 movies to get combined recommendations.")
        else:
            sel_idx = [title_to_idx[t] for t in chosen]
            section("✨ Recommended for your taste")
            show_recs(recommend_multi(df, sim, sel_idx, allowed_idx, top_n), "multi")

    # ===== TAB 4 : COMPARE TWO MOVIES =====
    with tab_cmp:
        section("⚖️ Compare two movies")
        c1, c2 = st.columns(2)
        t1 = c1.selectbox("First movie", all_titles, index=0, key="cmp_1")
        t2 = c2.selectbox("Second movie", all_titles, index=1, key="cmp_2")

        if t1 == t2:
            st.warning("Please choose two different movies.")
        else:
            i, j = title_to_idx[t1], title_to_idx[t2]
            a, b = df.iloc[i], df.iloc[j]
            score = float(sim[i][j])

            if score >= 0.5:
                verdict = "Very similar movies 🤝"
            elif score >= 0.25:
                verdict = "Somewhat similar 🙂"
            else:
                verdict = "Quite different 🔀"

            m1, m2 = st.columns([1, 3])
            m1.metric("Similarity", f"{score * 100:.0f}%")
            m2.markdown(f"### {verdict}")
            m2.progress(min(max(score, 0.0), 1.0))

            def same(x, y):
                return "✅ Same" if x == y else "❌ Different"

            table = pd.DataFrame({
                "Attribute": ["Year", "Rating", "Language", "Platform", "Genre", "Director", "Cast"],
                a["title"]: [a["year"], a["rating"], a["language"], a["platform"],
                             ", ".join(a["genre_list"]), ", ".join(a["director_list"]), ", ".join(a["cast_list"])],
                b["title"]: [b["year"], b["rating"], b["language"], b["platform"],
                             ", ".join(b["genre_list"]), ", ".join(b["director_list"]), ", ".join(b["cast_list"])],
                "Match?": [same(a["year"], b["year"]), same(a["rating"], b["rating"]),
                           same(a["language"], b["language"]), same(a["platform"], b["platform"]),
                           "✅ Shares: " + ", ".join(set(a["genre_list"]) & set(b["genre_list"]))
                           if set(a["genre_list"]) & set(b["genre_list"]) else "❌ None shared",
                           "✅ Shares: " + ", ".join(set(a["director_list"]) & set(b["director_list"]))
                           if set(a["director_list"]) & set(b["director_list"]) else "❌ None shared",
                           "✅ Shares: " + ", ".join(set(a["cast_list"]) & set(b["cast_list"]))
                           if set(a["cast_list"]) & set(b["cast_list"]) else "❌ None shared"],
            })
            st.dataframe(table, hide_index=True)

            if a["rating"] != b["rating"]:
                better = a if a["rating"] > b["rating"] else b
                st.success(f"🏆 Higher rated: **{better['title']}** ({better['rating']})")

            col_a, col_b = st.columns(2)
            with col_a:
                render_card(a, "cmp_a")
            with col_b:
                render_card(b, "cmp_b")

    # ===== TAB 5 : SURPRISE ME =====
    with tab_surprise:
        section("🎲 Can't decide? Let fate pick")
        all_genres = sorted({g for lst in df["genre_list"] for g in lst})
        genre_choice = st.selectbox("Limit to a genre (optional)", ["Any genre"] + all_genres, key="surp_genre")

        if st.button("🎲 Surprise Me!", key="surp_btn"):
            pool = filtered_df if genre_choice == "Any genre" else \
                filtered_df[filtered_df["genre_list"].apply(lambda l: genre_choice in l)]
            if pool.empty:
                st.session_state.surprise_title = None
                st.warning("No movies match your filters. Try relaxing them in the sidebar.")
            else:
                st.session_state.surprise_title = random.choice(pool["title"].tolist())

        picked = st.session_state.get("surprise_title")
        if picked:
            idx = title_to_idx[picked]
            section("🍿 Tonight's pick")
            render_card(df.iloc[idx], "surp")
            section("If you like it, try these")
            show_recs(recommend(df, sim, idx, allowed_idx, 3), "surp_rec")

    # ===== TAB 6 : WATCHLIST =====
    with tab_watch:
        section("📌 My Watchlist")
        wl_titles = st.session_state.watchlist
        if not wl_titles:
            st.info("Your watchlist is empty. Click **➕ Add to Watchlist** on any movie card.")
        else:
            st.caption(f"{len(wl_titles)} movie(s) saved (stored in watchlist.json).")
            st.button("🗑️ Clear watchlist", key="wl_clear", on_click=clear_watchlist)
            for t in wl_titles:
                render_card(df.iloc[title_to_idx[t]], "wl")

            if len(wl_titles) >= 1:
                section("✨ Because of your watchlist")
                sel_idx = [title_to_idx[t] for t in wl_titles]
                show_recs(recommend_multi(df, sim, sel_idx, allowed_idx, top_n), "wl_rec")

    # ===== TAB 7 : BROWSE =====
    with tab_browse:
        section("Browse by category")
        by = st.radio("Browse by", ["Genre", "Actor", "Director", "Language", "Platform"], horizontal=True)

        if by == "Genre":
            options = sorted({g for lst in df["genre_list"] for g in lst})
            choice = st.selectbox("Select genre", options)
            res = filtered_df[filtered_df["genre_list"].apply(lambda l: choice in l)]
        elif by == "Actor":
            options = sorted({c for lst in df["cast_list"] for c in lst})
            choice = st.selectbox("Select actor", options)
            res = filtered_df[filtered_df["cast_list"].apply(lambda l: choice in l)]
        elif by == "Director":
            options = sorted({d for lst in df["director_list"] for d in lst})
            choice = st.selectbox("Select director", options)
            res = filtered_df[filtered_df["director_list"].apply(lambda l: choice in l)]
        elif by == "Language":
            choice = st.selectbox("Select language", languages)
            res = filtered_df[filtered_df["language"] == choice]
        else:
            choice = st.selectbox("Select platform", platforms)
            res = filtered_df[filtered_df["platform"] == choice]

        res = sort_df(res, sort_by)
        st.caption(f"{len(res)} movie(s) found")
        show_results(res, "browse")

    # ===== TAB 8 : DASHBOARD =====
    with tab_stats:
        section("Dataset overview")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Movies", len(df))
        m2.metric("Languages", df["language"].nunique())
        m3.metric("Directors", len({d for l in df["director_list"] for d in l}))
        m4.metric("Avg rating", f"{df['rating'].mean():.2f}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top 10 rated movies**")
            st.bar_chart(df.nlargest(10, "rating").set_index("title")["rating"])
        with c2:
            st.markdown("**Movies per platform**")
            st.bar_chart(df["platform"].value_counts())
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Movies per language**")
            st.bar_chart(df["language"].value_counts())
        with c4:
            st.markdown("**Movies per genre**")
            st.bar_chart(pd.Series([g for l in df["genre_list"] for g in l]).value_counts())

        with st.expander("View full dataset"):
            st.dataframe(df[["title", "year", "genre", "director", "cast", "language", "platform", "rating"]])

    # ===== TAB 9 : ABOUT =====
    with tab_about:
        st.markdown("""
### About this project
**CineMatch 2.0** is a content-based movie recommendation system.

**How the recommendation works**
1. Every movie is converted into a text "profile" of its genres, director, cast and language.
2. Important features get more weight (genre and director x3, cast and language x2).
3. Profiles are converted to numbers with **TF-IDF**.
4. **Cosine similarity** measures how close two movies are; the closest ones are recommended.

**Features**
- **Smart Search**: a movie title gives similar movies (typos tolerated); genre, actor, director, language or platform gives matching movies.
- **Mood picks**: each mood is mapped to genres; movies sharing the most genres and with the best rating come first.
- **Multi-Movie**: the similarity scores of 2-5 favourite movies are averaged to find movies that fit all of them.
- **Compare**: shows the cosine similarity of two movies and a side-by-side table of shared attributes.
- **Surprise Me**: picks a random movie from your filtered list, optionally within one genre.
- **Watchlist**: saved in `watchlist.json`, and also used to generate recommendations.

**Tech stack:** Python, Pandas, NumPy, scikit-learn, Streamlit.

**About download links:** this app does not provide movie download links, because free downloads
of copyrighted films are piracy. Each movie has a button for the official streaming platform and a
JustWatch search that lists the legal places to watch or rent it.

**Future improvements:** TMDB API for a larger dataset and posters, user ratings,
collaborative filtering, user login.
""")


if __name__ == "__main__":
    main()