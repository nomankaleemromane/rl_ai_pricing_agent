import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression

class AdvancedRLAgent:
    def __init__(self,
                 price_options,
                 lr=0.1,
                 gamma=0.9,
                 epsilon=0.5,
                 epsilon_min=0.01,
                 epsilon_decay=0.90):
        self.price_options = sorted([float(p) for p in price_options])
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}
        self.q_max_per_episode = []
        self.sales_model = None

    def fit_sales_model(self, df):
        X = df[["Our_Price", "Competitor_Price", "Day_Type_Enc", "Inventory_Level", "Clicks"]]
        y = df["Units_Sold"]
        lr = LinearRegression()
        lr.fit(X, y)
        self.sales_model = lr

    def state_key(self, competitor_price, day_type, inventory, clicks):
        day_enc = 1 if str(day_type).lower() == "weekend" else 0
        return (float(competitor_price), day_enc, int(inventory), int(clicks))

    def step(self, state, our_price):
        comp_price, day_enc, inventory, clicks = state
        features = np.array([[our_price, comp_price, day_enc, inventory, clicks]])
        units = max(0, self.sales_model.predict(features)[0])
        reward = our_price * units
        next_inventory = max(0, inventory - units)
        next_clicks = max(0, clicks + random.randint(-5, 5))
        next_day = day_enc if random.random() < 0.9 else 1 - day_enc
        return reward, (comp_price, next_day, next_inventory, next_clicks)

    def choose_price(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.price_options)
        return max(self.price_options, 
                   key=lambda p: self.q_table.get((p,) + state, 0), 
                   default=self.price_options[0])

    def update(self, our_price, state, reward, next_state):
        current_q = self.q_table.get((our_price,) + state, 0)
        future_q = max([self.q_table.get((p,) + next_state, 0) for p in self.price_options], default=0)
        self.q_table[(our_price,) + state] = current_q + self.lr * (reward + self.gamma * future_q - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def track_q(self):
        self.q_max_per_episode.append(max(self.q_table.values(), default=0))

def train_advanced_rl(filepath, episodes=300, lr=0.1, gamma=0.9, epsilon=0.5, epsilon_decay=0.90):
    df = pd.read_csv(filepath)
    df['Day_Type_Enc'] = df['Day_Type'].str.lower().map({'weekday': 0, 'weekend': 1})
    price_opts = df['Our_Price'].unique()
    
    agent = AdvancedRLAgent(price_opts, lr, gamma, epsilon, epsilon_decay=epsilon_decay)
    agent.fit_sales_model(df)

    static_price = np.median(price_opts)
    cumulative_rl, cumulative_static = 0, 0
    rl_revenues, static_revenues = [], []
    logs = []

    for ep in range(episodes):
        sample = df.sample().iloc[0]
        state = agent.state_key(
            sample['Competitor_Price'],
            sample['Day_Type'],
            sample['Inventory_Level'],
            sample['Clicks']
        )
        
        # RL Agent
        rl_price = agent.choose_price(state)
        rl_reward, next_state = agent.step(state, rl_price)
        cumulative_rl += rl_reward
        
        # Static Baseline
        static_reward, _ = agent.step(state, static_price)
        cumulative_static += static_reward
        
        # Update agent and track metrics
        agent.update(rl_price, state, rl_reward, next_state)
        agent.track_q()
        agent.decay_epsilon()

        rl_revenues.append(cumulative_rl)
        static_revenues.append(cumulative_static)
        
        logs.append({
            'Episode': ep+1,
            'Our_Price': rl_price,
            'Competitor_Price': state[0],
            'Inventory': state[2],
            'Clicks': state[3],
            'Revenue': rl_reward,
            'Cumulative_Revenue': cumulative_rl
        })

    best_price = max(agent.q_table, key=agent.q_table.get)[0]
    
    return rl_revenues, static_revenues, agent.q_max_per_episode, logs, {
        'Total Revenue (RL)': float(cumulative_rl),
        'Total Revenue (Static)': float(cumulative_static),
        'Best Learned Price': float(best_price),
        'Total Episodes': episodes
    }