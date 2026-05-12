# house-price-prediction
This project predicts house sale prices using the Kaggle House Prices dataset. The goal is to build machine learning models that accurately estimate the price of a house.
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>House Prices Prediction</title>
  <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap" rel="stylesheet"/>
  <style>
    :root {
      --bg: #0d0f14;
      --surface: #13161e;
      --border: #1f2433;
      --accent: #e8c87a;
      --accent2: #7a9ee8;
      --accent3: #7ae8b8;
      --text: #e8eaf0;
      --muted: #6b7280;
    }

    * { margin: 0; padding: 0; box-sizing: border-box; }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Outfit', sans-serif;
      font-weight: 300;
      min-height: 100vh;
      overflow-x: hidden;
    }

    /* Background grid */
    body::before {
      content: '';
      position: fixed;
      inset: 0;
      background-image:
        linear-gradient(rgba(232,200,122,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(232,200,122,0.03) 1px, transparent 1px);
      background-size: 40px 40px;
      pointer-events: none;
      z-index: 0;
    }

    .container {
      max-width: 860px;
      margin: 0 auto;
      padding: 60px 32px;
      position: relative;
      z-index: 1;
    }

    /* Header */
    .header {
      border-left: 3px solid var(--accent);
      padding-left: 24px;
      margin-bottom: 60px;
      animation: fadeUp 0.8s ease both;
    }

    .header .tag {
      font-family: 'DM Mono', monospace;
      font-size: 11px;
      color: var(--accent);
      letter-spacing: 3px;
      text-transform: uppercase;
      margin-bottom: 12px;
    }

    .header h1 {
      font-family: 'DM Serif Display', serif;
      font-size: clamp(2.2rem, 5vw, 3.4rem);
      line-height: 1.1;
      color: var(--text);
      margin-bottom: 16px;
    }

    .header h1 em {
      font-style: italic;
      color: var(--accent);
    }

    .header p {
      font-size: 1rem;
      color: var(--muted);
      line-height: 1.7;
      max-width: 520px;
    }

    /* Badges */
    .badges {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 20px;
    }

    .badge {
      font-family: 'DM Mono', monospace;
      font-size: 11px;
      padding: 4px 12px;
      border-radius: 2px;
      border: 1px solid;
      letter-spacing: 1px;
    }

    .badge.gold  { border-color: var(--accent);  color: var(--accent);  background: rgba(232,200,122,0.06); }
    .badge.blue  { border-color: var(--accent2); color: var(--accent2); background: rgba(122,158,232,0.06); }
    .badge.green { border-color: var(--accent3); color: var(--accent3); background: rgba(122,232,184,0.06); }

    /* Section */
    .section {
      margin-bottom: 48px;
      animation: fadeUp 0.8s ease both;
    }

    .section-label {
      font-family: 'DM Mono', monospace;
      font-size: 10px;
      letter-spacing: 3px;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 20px;
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .section-label::after {
      content: '';
      flex: 1;
      height: 1px;
      background: var(--border);
    }

    /* Steps */
    .steps {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .step {
      display: flex;
      align-items: flex-start;
      gap: 16px;
      padding: 16px 20px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 4px;
      transition: border-color 0.2s, transform 0.2s;
    }

    .step:hover {
      border-color: rgba(232,200,122,0.3);
      transform: translateX(4px);
    }

    .step-num {
      font-family: 'DM Mono', monospace;
      font-size: 11px;
      color: var(--accent);
      min-width: 24px;
      margin-top: 2px;
    }

    .step-text {
      font-size: 0.9rem;
      line-height: 1.6;
      color: var(--text);
    }

    .step-text span {
      color: var(--muted);
      font-size: 0.82rem;
    }

    /* Models grid */
    .models {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }

    .model-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 24px;
      transition: border-color 0.2s;
    }

    .model-card:hover { border-color: rgba(122,158,232,0.4); }

    .model-card .icon {
      font-size: 1.6rem;
      margin-bottom: 12px;
    }

    .model-card h3 {
      font-family: 'DM Serif Display', serif;
      font-size: 1.15rem;
      margin-bottom: 8px;
      color: var(--accent2);
    }

    .model-card p {
      font-size: 0.82rem;
      color: var(--muted);
      line-height: 1.6;
    }

    .model-card .params {
      margin-top: 12px;
      font-family: 'DM Mono', monospace;
      font-size: 10px;
      color: var(--accent3);
      letter-spacing: 0.5px;
    }

    /* Libraries */
    .libs {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    .lib {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 10px 18px;
      font-family: 'DM Mono', monospace;
      font-size: 12px;
      color: var(--text);
      transition: border-color 0.2s, color 0.2s;
    }

    .lib:hover {
      border-color: var(--accent3);
      color: var(--accent3);
    }

    /* Output files */
    .files {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .file {
      display: flex;
      align-items: center;
      gap: 14px;
      padding: 14px 20px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 4px;
    }

    .file-icon {
      font-size: 1.2rem;
    }

    .file-name {
      font-family: 'DM Mono', monospace;
      font-size: 13px;
      color: var(--accent3);
    }

    .file-desc {
      font-size: 0.82rem;
      color: var(--muted);
      margin-left: auto;
    }

    /* Dataset link */
    .dataset-link {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 14px 24px;
      background: var(--surface);
      border: 1px solid var(--accent);
      border-radius: 4px;
      color: var(--accent);
      text-decoration: none;
      font-family: 'DM Mono', monospace;
      font-size: 12px;
      letter-spacing: 1px;
      transition: background 0.2s;
    }

    .dataset-link:hover {
      background: rgba(232,200,122,0.08);
    }

    /* Footer */
    .footer {
      margin-top: 80px;
      padding-top: 24px;
      border-top: 1px solid var(--border);
      font-family: 'DM Mono', monospace;
      font-size: 11px;
      color: var(--muted);
      display: flex;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 8px;
    }

    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(20px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    .section:nth-child(2) { animation-delay: 0.1s; }
    .section:nth-child(3) { animation-delay: 0.2s; }
    .section:nth-child(4) { animation-delay: 0.3s; }
    .section:nth-child(5) { animation-delay: 0.4s; }
    .section:nth-child(6) { animation-delay: 0.5s; }

    @media (max-width: 600px) {
      .models { grid-template-columns: 1fr; }
      .file-desc { display: none; }
    }
  </style>
</head>
<body>
  <div class="container">

    <!-- Header -->
    <div class="header">
      <div class="tag">Machine Learning · Regression</div>
      <h1>House Prices<br/><em>Prediction</em></h1>
      <p>Predict sales prices and practice feature engineering, Random Forests, and Gradient Boosting using the Kaggle House Prices dataset.</p>
      <div class="badges">
        <span class="badge gold">Python</span>
        <span class="badge blue">Scikit-learn</span>
        <span class="badge green">Kaggle Competition</span>
        <span class="badge gold">Regression</span>
      </div>
    </div>

    <!-- Steps -->
    <div class="section">
      <div class="section-label">Pipeline</div>
      <div class="steps">
        <div class="step">
          <div class="step-num">01</div>
          <div class="step-text">Load & clean data <span>— removed ID columns and dropped features with &gt;15% missing values</span></div>
        </div>
        <div class="step">
          <div class="step-num">02</div>
          <div class="step-text">Handle missing values <span>— used KNN Imputer to fill remaining nulls intelligently</span></div>
        </div>
        <div class="step">
          <div class="step-num">03</div>
          <div class="step-text">Feature encoding <span>— applied One-Hot Encoding to all categorical columns via get_dummies()</span></div>
        </div>
        <div class="step">
          <div class="step-num">04</div>
          <div class="step-text">Feature selection <span>— removed multicollinear features using Variance Inflation Factor (VIF &gt; 10)</span></div>
        </div>
        <div class="step">
          <div class="step-num">05</div>
          <div class="step-text">Model training <span>— trained Random Forest and Gradient Boosting regressors</span></div>
        </div>
        <div class="step">
          <div class="step-num">06</div>
          <div class="step-text">Submission <span>— generated CSV predictions for Kaggle leaderboard evaluation</span></div>
        </div>
      </div>
    </div>

    <!-- Models -->
    <div class="section">
      <div class="section-label">Models</div>
      <div class="models">
        <div class="model-card">
          <div class="icon">🌲</div>
          <h3>Random Forest</h3>
          <p>Ensemble of decision trees built independently in parallel. Fast and robust to overfitting.</p>
          <div class="params">n_estimators=100 · random_state=42</div>
        </div>
        <div class="model-card">
          <div class="icon">📈</div>
          <h3>Gradient Boosting</h3>
          <p>Trees built sequentially, each correcting the previous. High accuracy with careful tuning.</p>
          <div class="params">n_estimators=50 · lr=0.1 · max_depth=3</div>
        </div>
      </div>
    </div>

    <!-- Libraries -->
    <div class="section">
      <div class="section-label">Libraries</div>
      <div class="libs">
        <div class="lib">pandas</div>
        <div class="lib">numpy</div>
        <div class="lib">scikit-learn</div>
        <div class="lib">statsmodels</div>
        <div class="lib">KNNImputer</div>
        <div class="lib">matplotlib</div>
      </div>
    </div>

    <!-- Output Files -->
    <div class="section">
      <div class="section-label">Output Files</div>
      <div class="files">
        <div class="file">
          <div class="file-icon">📄</div>
          <div class="file-name">submission_rf.csv</div>
          <div class="file-desc">Random Forest predictions</div>
        </div>
        <div class="file">
          <div class="file-icon">📄</div>
          <div class="file-name">submission_gb.csv</div>
          <div class="file-desc">Gradient Boosting predictions</div>
        </div>
      </div>
    </div>

    <!-- Dataset -->
    <div class="section">
      <div class="section-label">Dataset</div>
      <a class="dataset-link" href="https://www.kaggle.com/c/house-prices-advanced-regression-techniques" target="_blank">
        ↗ Kaggle — House Prices: Advanced Regression Techniques
      </a>
    </div>

    <!-- Footer -->
    <div class="footer">
      <span>house-prices-prediction</span>
      <span>lujai · 2025</span>
    </div>

  </div>
</body>
</html>
