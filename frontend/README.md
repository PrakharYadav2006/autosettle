# Auto-Settle AI - frontend (Vite + React + TypeScript + Tailwind)

Two separate applications in one build:

| URL      | App              | Login                                                         |
|----------|------------------|---------------------------------------------------------------|
| `/`      | Customer portal  | "Launch demo session" -> signed customer token (synthetic data) |
| `/admin` | Staff console    | username + password checked by the server                     |

Admin password: set `AUTOSETTLE_ADMIN_PASSWORD` before `python api_server.py`; if you don't,
the server prints a random one in its terminal at start-up. Customer tokens are rejected by
every `/api/admin/*` endpoint (server-side check), so changing the URL does not grant access.

The customer API and UI never expose the settlement ceiling, the offer ladder or the
negotiation step - only the current authorised offer. Those appear only in the staff console.

## Run
```
npm install
npm run dev          # http://localhost:5173   (backend: python api_server.py on :8000)
```
Backend URL: login page -> "Server connection", or `VITE_API_URL` at build time.

## Vercel
Root directory `frontend`, framework Vite. `vercel.json` rewrites all paths to index.html so
`/admin` works. Set `VITE_API_URL` to your tunnel URL (cloudflared tunnel --url http://localhost:8000).
