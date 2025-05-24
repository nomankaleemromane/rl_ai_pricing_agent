from flask import Flask, render_template, request
import os
import pandas as pd
from rl_agent import train_advanced_rl

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    data_preview = None
    summary = None
    rl_revenues = static_revenues = q_max = log_data = None

    if request.method == 'POST':
        csvfile = request.files['file']
        path = os.path.join(UPLOAD_FOLDER, csvfile.filename)
        csvfile.save(path)
        df = pd.read_csv(path)
        data_preview = df.head().to_html(classes='table table-bordered', index=False)

        episodes = int(request.form.get('episodes', 300))
        eps = float(request.form.get('epsilon', 0.5))
        decay = float(request.form.get('decay', 0.90))
        lr = float(request.form.get('lr', 0.1))
        gamma = float(request.form.get('gamma', 0.9))

        rl_revenues, static_revenues, q_max, log_data, summary = train_advanced_rl(
            path,
            episodes=episodes,
            lr=lr,
            gamma=gamma,
            epsilon=eps,
            epsilon_decay=decay
        )

    return render_template('index.html',
        data_preview=data_preview,
        summary=summary,
        rl_revenues=rl_revenues,
        static_revenues=static_revenues,
        q_max=q_max,
        log_data=log_data
    )

if __name__ == '__main__':
    app.run(debug=True)