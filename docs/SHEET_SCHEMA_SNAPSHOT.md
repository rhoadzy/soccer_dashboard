# Sheet schema snapshot (read-only)

Source sheet: https://docs.google.com/spreadsheets/d/1gRgooEJzb6ibKPSGnq3HAxOC6WFy_euo9iMpbR-oFeg/edit

## Worksheets

### seasons (2 rows)
Columns:
- season_id
- label
- active

### matches (18 rows)
Columns:
- match_id
- season_id
- date
- opponent
- home_away
- division_game
- result
- goals_for
- goals_against
- shots
- shots_target
- shots_against
- shots_against_target
- saves
- url

### players (20 rows)
Columns:
- player_id
- player_status
- name
- position
- jersey

### events (175 rows)
Columns:
- event_id
- season_id
- match_id
- player_id
- goals
- assists
- shots
- fouls

### plays (188 rows)
Columns:
- match_id
- season_id
- set_piece
- play_call_id
- taker_id
- play type
- goal_created

### goals_allowed (24 rows)
Columns:
- match_id
- season_id
- goal_id
- description
- goalie_player_id
- minute
- situation

### summary (18 rows)
Columns:
- match_id
- season_id
- opp_formation
- our_formation
- opp_style_notes
- opp_key_players
- opp_seniors_count
- next_year_players_to_watch
- our_player_of_game
- injuries_absences
- tactical_adjustments
- misc_notes
