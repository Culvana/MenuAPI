# MenuAPI

> **Status:** Beta – RESTful micro‑service for managing menu items, categories, and availability.

`MenuAPI` powers Culvanaʼs customer‑facing menu pages and internal Menu Assistant. It lets you **create, read, update, and soft‑delete** menu entities, attach images, handle seasonal availability, and expose SEO‑friendly slugs that front‑end clients can cache.

---

## ✨ Feature Matrix

| Feature               | Endpoint                                                                      | Notes                                                           |
| --------------------- | ----------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **CRUD menu items**   | `GET/POST /menu/items`<br>`GET/PATCH/DELETE /menu/items/{id}`                 | JSON schema in `schemas/menu_item.json`.                        |
| **Category tree**     | `GET /menu/categories`                                                        | Recursive structure for nested sections (e.g., Lunch → Salads). |
| **Publishing toggle** | `POST /menu/items/{id}/publish`                                               | Draft ↔ Published status.                                       |
| **Images upload**     | `POST /menu/items/{id}/media`                                                 | Stores in Azure Blob, returns CDN URL.                          |
| **Allergen tagging**  | Auto‑detects allergens via GPT review of ingredients; saved to `allergens[]`. |                                                                 |
| **Global search**     | `GET /menu/search?q=pizza&limit=10`                                           | Full‑text + fuzzy search using SQLite FTS5.                     |

---

## 📁 Structure Overview

```
MenuAPI/
├── app/
│   ├── main.py          # FastAPI app instance
│   ├── models.py        # SQLModel ORM entities
│   ├── routers/
│   │   ├── items.py     # /menu/items
│   │   └── categories.py
│   ├── services/
│   │   ├── media.py     # Azure Blob helpers
│   │   └── search.py    # FTS wrapper
│   └── core/config.py   # Pydantic settings
├── alembic/             # DB migrations
├── requirements.txt
└── docker-compose.yml   # Dev DB + API
```

---

## 🛠 Prerequisites

* Python 3.11+
* SQLite (default) or Postgres
* Azure Storage account for media uploads

---

## 🚀 Local Development

```bash
git clone https://github.com/Culvana/MenuAPI.git
cd MenuAPI
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload
```

### Env Vars (`.env`)

| Key                  | Purpose                      | Default               |
| -------------------- | ---------------------------- | --------------------- |
| `DATABASE_URL`       | SQLAlchemy connection string | `sqlite:///./menu.db` |
| `AZURE_STORAGE_CONN` | Blob upload target           | n/a                   |
| `CDN_BASE_URL`       | Prepended to media URLs      | optional              |
| `ALLOWED_ORIGINS`    | Comma list for CORS          | `*`                   |

---

## 🧪 Testing

```
pytest -v tests/
```

---

## ⬆️ Deploying

1. **Azure App Service (Code):**

   ```bash
   az webapp up -n culvana-menu-api -g culvana-rg -l eastus
   ```
2. **Container:** Build image from `Dockerfile` and push to ACR.

---

## 📝 License

MIT © Culvana 2025
