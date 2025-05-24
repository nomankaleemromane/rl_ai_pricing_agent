# Smart Pricing AI

A dynamic pricing simulator powered by Reinforcement Learning (Q-learning) that teaches an AI agent to pick the best product price at the right time. The agent learns from a simulated market environment using customer behavior factors like competitor price, day type, inventory, and clicks.

##  Features
- Q-learning agent with epsilon-greedy exploration
- Linear Regression-based demand simulator
- Revenue comparison: RL vs Static pricing
- Interactive Flask dashboard with Plotly graphs

## Tech Stack
- Python
- Flask
- Scikit-learn
- Plotly
- Bootstrap

## How it Works
1. Upload your CSV file (e.g., historical pricing data).
2. Adjust hyperparameters like α, γ, ε, and episodes.
3. Watch the RL agent learn and compare against static pricing.
4. Get revenue performance, AI-learned price, and Q-value insights.

## Sample CSV Format

```csv
Our_Price,Competitor_Price,Units_Sold,Day_Type,Inventory_Level,Clicks
8.99,9.49,10,Weekday,500,100
...
