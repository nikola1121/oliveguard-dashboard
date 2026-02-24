# 🫒 OliveGuard Dashboard — Deployment Guide

## 🏗️ არქიტექტურა

```
შენი კომპიუტერი                    ☁️ Cloud
┌─────────────┐                 ┌──────────────┐
│ run_once.py  │ ──sync──────→  │  Neon DB      │
│ PostgreSQL   │                │  (უფასო)      │
│ (localhost)  │                └──────┬───────┘
└─────────────┘                       │
                                      │ კითხულობს
                                ┌─────▼────────┐
                                │  Streamlit    │
                                │  Cloud        │
                                │  (უფასო)      │
                                └──────┬───────┘
                                       │
                                  ლინკი ↓
                                https://oliveguard.streamlit.app
```

## 📋 ნაბიჯი 1 — Neon ონლაინ ბაზა (5 წუთი)

1. შედი: https://neon.tech → Sign Up (GitHub-ით)
2. **Create Project** → სახელი: `oliveguard`
3. Database-ს დაარქვი: `gis_db`
4. **დააკოპირე Connection string:**
   ```
   postgresql://gis_db_owner:xxxx@ep-xxx-123.us-east-2.aws.neon.tech/gis_db
   ```
5. გახსენი **SQL Editor** → ჩასვი:
   ```sql
   CREATE EXTENSION IF NOT EXISTS postgis;
   ```

## 📋 ნაბიჯი 2 — ბაზის სინქრონიზაცია

1. `sync_to_cloud.py`-ში CLOUD dict-ში ჩასვი Neon-ის მონაცემები:
   ```python
   CLOUD = {
       "host": "ep-xxx-123.us-east-2.aws.neon.tech",
       "port": "5432",
       "database": "gis_db",
       "user": "gis_db_owner",
       "password": "neon_password_here"
   }
   ```

2. გაუშვი:
   ```
   python sync_to_cloud.py
   ```

3. ყოველ `run_once.py`-ს შემდეგ გაუშვი `sync_to_cloud.py` ან Task Scheduler-ით ავტომატიზება.

## 📋 ნაბიჯი 3 — GitHub რეპო (3 წუთი)

1. შედი: https://github.com → New Repository
2. სახელი: `oliveguard-dashboard`
3. **Public** ან **Private**
4. ტერმინალში:
   ```
   cd olive_dashboard
   git init
   git add .
   git commit -m "OliveGuard Dashboard"
   git remote add origin https://github.com/YOUR_USER/oliveguard-dashboard.git
   git push -u origin main
   ```

## 📋 ნაბიჯი 4 — Streamlit Cloud (3 წუთი)

1. შედი: https://share.streamlit.io → GitHub-ით
2. **New app** →
   - Repository: `YOUR_USER/oliveguard-dashboard`
   - Branch: `main`
   - Main file: `streamlit_app.py`
3. **Advanced settings → Secrets** → ჩასვი:
   ```toml
   [database]
   host = "ep-xxx-123.us-east-2.aws.neon.tech"
   port = 5432
   database = "gis_db"
   user = "gis_db_owner"
   password = "neon_password_here"
   ```
4. **Deploy!**

## 🔗 შედეგი

შენი dashboard ლინკი:
```
https://YOUR_USER-oliveguard-dashboard-streamlit-app-xxxxx.streamlit.app
```

ამ ლინკს ვინც გინდა გაუგზავნე — ნებისმიერ მოწყობილობაზე გაიხსნება.

## 🔄 განახლების ციკლი

```
python run_once.py          ← მონაცემების შეგროვება (ლოკალურად)
python sync_to_cloud.py     ← Neon-ზე გაგზავნა
→ Dashboard ავტომატურად განახლდება (5 წუთი cache)
```

## 💰 ფასები

| სერვისი | გეგმა | ფასი |
|---------|-------|------|
| Neon PostgreSQL | Free tier | $0 (0.5GB) |
| Streamlit Cloud | Community | $0 |
| **სულ** | | **$0** |
