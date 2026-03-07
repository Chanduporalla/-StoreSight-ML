from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, roc_auc_score
import warnings, os, json

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ─── GLOBAL STATE ───────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), 'Sample_-_Superstore.csv')
df_global      = None
clf_model      = None
reg_model      = None
label_encoders = {}
model_metrics  = {}
FEATURES       = []

# ─── TRAIN ON STARTUP ───────────────────────────────────────────
def load_and_train():
    global df_global, clf_model, reg_model, label_encoders, model_metrics, FEATURES

    df = pd.read_csv(DATA_PATH, encoding='latin1')
    df['Order Date']       = pd.to_datetime(df['Order Date'], format='%m/%d/%Y')
    df['Ship Date']        = pd.to_datetime(df['Ship Date'],  format='%m/%d/%Y')
    df['Ship Days']        = (df['Ship Date'] - df['Order Date']).dt.days
    df['Year']             = df['Order Date'].dt.year
    df['Month']            = df['Order Date'].dt.month
    df['Quarter']          = df['Order Date'].dt.quarter
    df['Revenue_Per_Unit'] = df['Sales'] / df['Quantity']
    df['Profit_Margin']    = df['Profit'] / (df['Sales'] + 1e-6)
    df['Discount_Flag']    = (df['Discount'] > 0.2).astype(int)
    df['High_Value']       = (df['Sales'] > df['Sales'].quantile(0.75)).astype(int)
    df['IsQ4']             = (df['Quarter'] == 4).astype(int)
    df['Is_Profitable']    = (df['Profit'] > 0).astype(int)

    cat_cols = ['Ship Mode','Segment','Region','Category','Sub-Category']
    for col in cat_cols:
        le = LabelEncoder()
        df[col+'_enc'] = le.fit_transform(df[col])
        label_encoders[col] = le

    FEATURES[:] = ['Sales','Quantity','Discount','Ship Days','Year','Month','Quarter',
                   'Revenue_Per_Unit','Profit_Margin','Discount_Flag','High_Value','IsQ4',
                   'Ship Mode_enc','Segment_enc','Region_enc','Category_enc','Sub-Category_enc']

    X     = df[FEATURES]
    y_cls = df['Is_Profitable']
    y_reg = df['Profit']

    X_tr, X_te, yc_tr, yc_te = train_test_split(X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
    Xr_tr,Xr_te,yr_tr,yr_te  = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    clf_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    clf_model.fit(X_tr, yc_tr)

    reg_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    reg_model.fit(Xr_tr, yr_tr)

    cls_preds  = clf_model.predict(X_te)
    cls_proba  = clf_model.predict_proba(X_te)[:,1]
    reg_preds  = reg_model.predict(Xr_te)

    feat_imp = pd.Series(clf_model.feature_importances_, index=FEATURES).sort_values(ascending=False)

    model_metrics.update({
        'accuracy':  round(accuracy_score(yc_te, cls_preds) * 100, 2),
        'auc':       round(roc_auc_score(yc_te, cls_proba), 4),
        'r2':        round(r2_score(yr_te, reg_preds), 4),
        'rmse':      round(np.sqrt(mean_squared_error(yr_te, reg_preds)), 2),
        'total_rows':len(df),
        'total_sales':   round(df['Sales'].sum(), 2),
        'total_profit':  round(df['Profit'].sum(), 2),
        'profit_margin': round((df['Profit'].sum() / df['Sales'].sum()) * 100, 2),
        'feature_importance': feat_imp.head(10).round(4).to_dict(),
    })

    df_global = df
    print("✅ Model trained successfully!")

# ─── ROUTES ─────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', metrics=model_metrics)

@app.route('/api/kpis')
def api_kpis():
    return jsonify({
        'total_sales':    f"${model_metrics['total_sales']:,.0f}",
        'total_profit':   f"${model_metrics['total_profit']:,.0f}",
        'accuracy':       f"{model_metrics['accuracy']}%",
        'auc':            str(model_metrics['auc']),
        'total_rows':     f"{model_metrics['total_rows']:,}",
        'profit_margin':  f"{model_metrics['profit_margin']}%",
    })

@app.route('/api/sales_by_category')
def sales_by_category():
    data = df_global.groupby('Category')[['Sales','Profit']].sum().round(2)
    return jsonify({
        'labels':  data.index.tolist(),
        'sales':   data['Sales'].tolist(),
        'profit':  data['Profit'].tolist(),
    })

@app.route('/api/monthly_trend')
def monthly_trend():
    df_global['YearMonth'] = df_global['Order Date'].dt.to_period('M').astype(str)
    trend = df_global.groupby('YearMonth')[['Sales','Profit']].sum().round(2)
    return jsonify({
        'labels':  trend.index.tolist(),
        'sales':   trend['Sales'].tolist(),
        'profit':  trend['Profit'].tolist(),
    })

@app.route('/api/quarterly_trend')
def quarterly_trend():
    df_global['YQ'] = df_global['Order Date'].dt.to_period('Q').astype(str)
    qt = df_global.groupby('YQ')[['Sales','Profit']].sum().round(2)
    return jsonify({
        'labels':  qt.index.tolist(),
        'sales':   qt['Sales'].tolist(),
        'profit':  qt['Profit'].tolist(),
    })

@app.route('/api/subcategory_profit')
def subcategory_profit():
    data = df_global.groupby('Sub-Category')['Profit'].sum().sort_values().round(2)
    return jsonify({
        'labels':  data.index.tolist(),
        'profit':  data.tolist(),
    })

@app.route('/api/discount_impact')
def discount_impact():
    df_global['Disc_Band'] = pd.cut(df_global['Discount'],
        bins=[-0.01,0,0.1,0.2,0.3,0.5,1.0],
        labels=['0%','1-10%','11-10%','21-30%','31-50%','51%+'])
    data = df_global.groupby('Disc_Band')['Profit'].mean().round(2)
    return jsonify({'labels': data.index.tolist(), 'profit': data.tolist()})

@app.route('/api/region_profit')
def region_profit():
    data = df_global.groupby('Region')[['Sales','Profit']].sum().round(2)
    return jsonify({
        'labels': data.index.tolist(),
        'sales':  data['Sales'].tolist(),
        'profit': data['Profit'].tolist(),
    })

@app.route('/api/segment_profit')
def segment_profit():
    data = df_global.groupby('Segment')['Profit'].sum().round(2)
    return jsonify({'labels': data.index.tolist(), 'profit': data.tolist()})

@app.route('/api/feature_importance')
def feature_importance():
    fi = model_metrics['feature_importance']
    return jsonify({'labels': list(fi.keys()), 'values': list(fi.values())})

@app.route('/api/state_profit')
def state_profit():
    data = df_global.groupby('State')['Profit'].sum().sort_values()
    top5  = data.tail(5)
    bot5  = data.head(5)
    combined = pd.concat([bot5, top5])
    return jsonify({
        'labels': combined.index.tolist(),
        'profit': combined.round(2).tolist(),
    })

@app.route('/api/future_sales')
def future_sales():
    """Predict future sales for next 6 months using trend extrapolation + GB model"""
    df_global['YearMonth'] = df_global['Order Date'].dt.to_period('M').astype(str)
    monthly = df_global.groupby('YearMonth')['Sales'].sum().reset_index()
    monthly.columns = ['period','sales']

    # Simple trend: use last 12 months average growth
    last12 = monthly.tail(12)['sales'].values
    growth = np.mean(np.diff(last12) / last12[:-1])  # avg monthly growth rate

    last_sales = monthly['sales'].iloc[-1]
    last_period = pd.Period(monthly['period'].iloc[-1], freq='M')

    future_labels = []
    future_sales  = []
    future_optimistic = []
    future_pessimistic = []

    for i in range(1, 7):
        p = last_period + i
        projected = last_sales * ((1 + growth) ** i)
        future_labels.append(str(p))
        future_sales.append(round(projected, 2))
        future_optimistic.append(round(projected * 1.15, 2))
        future_pessimistic.append(round(projected * 0.85, 2))

    # Historical last 12 months
    hist_labels = monthly.tail(12)['period'].tolist()
    hist_sales  = monthly.tail(12)['sales'].round(2).tolist()

    return jsonify({
        'hist_labels':       hist_labels,
        'hist_sales':        hist_sales,
        'future_labels':     future_labels,
        'future_sales':      future_sales,
        'future_optimistic': future_optimistic,
        'future_pessimistic':future_pessimistic,
        'growth_rate':       round(growth * 100, 2),
    })

@app.route('/api/top_products')
def top_products():
    top = df_global.groupby('Product Name')['Profit'].sum().sort_values(ascending=False).head(8).round(2)
    bot = df_global.groupby('Product Name')['Profit'].sum().sort_values().head(5).round(2)
    return jsonify({
        'top_labels':  top.index.tolist(),
        'top_profit':  top.tolist(),
        'bot_labels':  bot.index.tolist(),
        'bot_profit':  bot.tolist(),
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        sales    = float(data.get('sales', 0))
        quantity = int(data.get('quantity', 1))
        discount = float(data.get('discount', 0))
        ship_mode  = data.get('ship_mode', 'Standard Class')
        segment    = data.get('segment', 'Consumer')
        region     = data.get('region', 'West')
        category   = data.get('category', 'Technology')
        sub_cat    = data.get('sub_category', 'Phones')
        ship_days  = int(data.get('ship_days', 3))
        year       = int(data.get('year', 2024))
        month      = int(data.get('month', 6))

        quarter    = (month - 1) // 3 + 1
        rev_unit   = sales / max(quantity, 1)
        prof_margin= 0.15
        disc_flag  = 1 if discount > 0.2 else 0
        high_val   = 1 if sales > 209.94 else 0
        isq4       = 1 if quarter == 4 else 0

        sm_enc  = label_encoders['Ship Mode'].transform([ship_mode])[0]
        seg_enc = label_encoders['Segment'].transform([segment])[0]
        reg_enc = label_encoders['Region'].transform([region])[0]
        cat_enc = label_encoders['Category'].transform([category])[0]
        sc_enc  = label_encoders['Sub-Category'].transform([sub_cat])[0]

        row = [[sales, quantity, discount, ship_days, year, month, quarter,
                rev_unit, prof_margin, disc_flag, high_val, isq4,
                sm_enc, seg_enc, reg_enc, cat_enc, sc_enc]]

        is_prof  = int(clf_model.predict(row)[0])
        profit   = round(float(reg_model.predict(row)[0]), 2)
        prob     = round(float(clf_model.predict_proba(row)[0][1]) * 100, 1)

        return jsonify({
            'is_profitable': is_prof,
            'profit':        profit,
            'probability':   prob,
            'label':         '✅ Profitable' if is_prof else '❌ Loss',
            'color':         '#59a14f' if is_prof else '#e15759',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    global DATA_PATH
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    f = request.files['file']
    path = os.path.join(os.path.dirname(__file__), f.filename)
    f.save(path)
    DATA_PATH = path
    load_and_train()
    return jsonify({'success': True, 'message': f'Loaded {f.filename} and retrained model'})

if __name__ == '__main__':
    load_and_train()
    app.run(debug=True, port=5000)
