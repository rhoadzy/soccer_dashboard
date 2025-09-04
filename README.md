# HS Soccer Dashboard

A comprehensive analytics dashboard for high school soccer teams, built with Streamlit and powered by Google Sheets integration and AI insights.

## 🚀 Features

### 📊 **Team Analytics**
- **Season Overview**: Games played, goals for/against, shots, saves, conversion rates
- **Trend Analysis**: Rolling 3-game performance metrics
- **Division Rankings**: Integration with SI.com rankings
- **Mobile-Responsive**: Optimized for coaches on the go

### 🎯 **Set-Piece Analysis**
- **Performance Tracking**: Corners, penalties, direct/indirect free kicks
- **Taker Analysis**: Individual player effectiveness on set pieces
- **AI Insights**: Automated recommendations for set-piece strategy
- **Visual Charts**: Success rates and performance trends

### 🛡️ **Defensive Analysis**
- **Goals Allowed Tracking**: Detailed breakdown by situation and minute
- **Goalie Performance**: Individual goalkeeper statistics
- **AI Defensive Insights**: Automated defensive recommendations
- **Pattern Recognition**: Identify defensive vulnerabilities

### 🎮 **Game Drill-Down**
- **Individual Match Analysis**: Detailed game-by-game breakdowns
- **Player Performance**: Per-player statistics and contributions
- **Game Recording Links**: Direct access to recorded game videos
- **Coach Notes Integration**: Game-specific coaching insights
- **AI Game Summaries**: Automated match recaps and takeaways

### 🤖 **AI-Powered Insights**
- **Gemini 2.0 Flash Lite**: Fast, reliable AI analysis
- **Set-Piece Strategy**: Recommendations for corner kicks, free kicks, penalties
- **Defensive Coaching**: Automated defensive improvement suggestions
- **Game Analysis**: Match-specific insights and training priorities

## 🛠️ Setup

### Prerequisites
- Python 3.8+
- Google Cloud Project with Sheets API enabled
- Gemini API key (optional, for AI features)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rhoadzy/soccer_dashboard.git
   cd soccer_dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   
   Create a `.env` file or set Streamlit secrets:
   ```bash
   SPREADSHEET_KEY=your_google_sheet_id_or_url
   GEMINI_API_KEY=your_gemini_api_key  # Optional
   GOOGLE_SERVICE_ACCOUNT_JSON=path_to_service_account.json
   ```

4. **Set up Google Sheets**
   
   Your Google Sheet should have these tabs:
   - **matches**: Game results and statistics (include optional 'url' column for game recordings)
   - **players**: Team roster information
   - **events**: Individual player performance
   - **plays**: Set-piece attempts and results
   - **goals_allowed**: Defensive statistics
   - **summaries**: Coach notes and game summaries

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📋 Data Structure

### Matches Tab
| Column | Type | Description |
|--------|------|-------------|
| match_id | string | Unique game identifier |
| date | date | Game date |
| opponent | string | Opposing team |
| home_away | string | H/A for home/away |
| goals_for | integer | Goals scored |
| goals_against | integer | Goals conceded |
| shots_for | integer | Shots taken |
| shots_against | integer | Shots faced |
| saves | integer | Goalkeeper saves |
| result | string | W/L/D for win/loss/draw |
| division_game | boolean | Division game flag |
| url | string | Game recording URL (optional) |

### Plays Tab (Set-Pieces)
| Column | Type | Description |
|--------|------|-------------|
| match_id | string | Game identifier |
| set_piece | string | Type (corner, penalty, fk_direct, fk_indirect) |
| play_call_id | string | Play name/identifier |
| taker_id | string | Player taking the set-piece |
| goal_created | boolean | Whether it resulted in a goal |
| play_type | string | Additional play details |

### Goals Allowed Tab
| Column | Type | Description |
|--------|------|-------------|
| match_id | string | Game identifier |
| goal_id | string | Unique goal identifier |
| minute | integer | Minute of goal |
| situation | string | Goal situation (set piece, open play, etc.) |
| goalie_player_id | string | Goalkeeper player ID |
| description | string | Goal description |

## 🎨 Features in Detail

### Mobile-First Design
- **Responsive KPI Cards**: Clean, scannable statistics on mobile
- **Compact Mode**: Optimized layout for smaller screens
- **Touch-Friendly**: Easy navigation on tablets and phones

### AI Integration
- **Set-Piece Analysis**: Identifies most effective takers and strategies
- **Defensive Insights**: Patterns in goals conceded and improvement areas
- **Game Summaries**: Automated match recaps with coaching takeaways

### Data Visualization
- **Trend Charts**: Rolling 3-game performance metrics
- **Set-Piece Success Rates**: Visual breakdown by type and taker
- **Defensive Patterns**: Goals conceded by situation and minute

## 🔧 Configuration

### Environment Variables
- `SPREADSHEET_KEY`: Your Google Sheet ID or URL
- `GEMINI_API_KEY`: Gemini API key for AI features
- `GOOGLE_SERVICE_ACCOUNT_JSON`: Service account credentials
- `APP_PASSWORD`: Optional password protection

### Streamlit Secrets
For cloud deployment, configure these in your Streamlit secrets:
```toml
SPREADSHEET_KEY = "your_sheet_id"
GEMINI_API_KEY = "your_api_key"
GOOGLE_SERVICE_ACCOUNT_JSON = "your_service_account_json"
```

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Cloud Deployment
1. **Streamlit Cloud**: Connect your GitHub repository
2. **Heroku**: Use the provided Procfile
3. **Google Cloud Run**: Containerize with Docker

## 📱 Usage

### Dashboard Navigation
- **Home**: Season overview and team KPIs
- **Games**: Match results and statistics
- **Trends**: Rolling performance metrics
- **Set-Pieces**: Corner kicks, free kicks, penalties analysis
- **Defense**: Goals allowed and defensive patterns
- **Individual Games**: Click any game for detailed analysis
- **Game Recordings**: Access recorded game videos directly from game view

### Game Recording Integration
- **Automatic Detection**: Supports multiple URL column names (url, recording_url, game_url, video_url, link)
- **Smart Display**: Shows recording link with styled blue info box when available
- **Protocol Handling**: Automatically adds https:// if missing from URLs
- **User-Friendly**: Clean "No recording available" message when no URL is present
- **New Tab Opening**: Recording links open in new tab for better user experience

### AI Features
- Click "🔎 Generate AI Insights" buttons for automated analysis
- AI provides coaching recommendations and strategic insights
- Set-piece analysis identifies most effective players and tactics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
- Check the [Issues](https://github.com/rhoadzy/soccer_dashboard/issues) page
- Review the data structure requirements
- Ensure all environment variables are properly configured

## 🏆 Credits

Built for high school soccer coaches and analysts to improve team performance through data-driven insights and AI-powered recommendations.

---

**Made with ❤️ for the beautiful game**
