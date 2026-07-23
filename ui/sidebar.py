import streamlit as st


def render_sidebar(
    *,
    qparams_get,
    qp_bool,
    qparams_set,
    qparams_merge_update,
    season_options: list[str],
    season_labels: dict[str, str],
    default_season: str,
    schedule_url: str,
    rankings_url: str,
) -> tuple[bool, bool, str]:
    """Render sidebar and keep query params in sync.

    Returns:
        (compact, div_only, selected_season)

    Note: Other filters (opp/ha) are stored in query params; the main app can
    re-read them from qparams_get() to preserve current behavior.
    """

    with st.sidebar:
        st.title("HS Soccer")
        if st.button("Dashboard (Home)"):
            qparams_set()
            st.rerun()

        qp_init = qparams_get()
        requested_season = str(qp_init.get("season", default_season))
        selected_season = requested_season if requested_season in season_options else default_season
        selected_season = st.selectbox(
            "Season",
            season_options,
            index=season_options.index(selected_season),
            format_func=lambda value: season_labels.get(value, f"{value} season"),
        )

        COMPACT_DEFAULT = True
        compact_init = qp_bool(qp_init.get("compact"), COMPACT_DEFAULT)
        div_only_init = qp_bool(qp_init.get("div_only"), False)

        compact = st.toggle("Compact mode", value=compact_init, help="Phone-friendly layout")
        div_only = st.checkbox("Division games only", value=div_only_init)

        st.subheader("Filters")
        opponent_q = st.text_input("Opponent contains", value=str(qp_init.get("opp", "")))
        ha_opt = st.selectbox(
            "Home/Away",
            ["Any", "Home", "Away"],
            index={"any": 0, "home": 1, "away": 2}.get(str(qp_init.get("ha", "any")).lower(), 0),
        )

        st.link_button("Open Schedule", schedule_url)
        st.link_button("Open Rankings (D2)", rankings_url)

        # Sync toggles/filters to query params only when they differ
        try:
            desired = {
                "season": selected_season,
                "compact": str(compact).lower(),
                "div_only": str(div_only).lower(),
                "opp": opponent_q.strip(),
                # Store full text so "Any" is not mistaken for Away
                "ha": ha_opt.lower() if ha_opt else "any",
            }

            diffs = []
            for k, v in desired.items():
                curv = qp_init.get(k)
                if isinstance(curv, list):
                    curv = curv[0] if curv else None
                if (curv or "") != (v or ""):
                    diffs.append(k)

            if diffs:
                qparams_merge_update(**desired)
                st.rerun()
        except Exception:
            pass

    return compact, div_only, selected_season
