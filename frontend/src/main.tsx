import React, { Suspense, lazy } from "react";
import ReactDOM from "react-dom/client";
import "./index.css";

// Two separate applications. The admin bundle is only loaded on /admin, and every admin
// API call is authorised by the server - hiding routes here is not the security boundary.
const CustomerApp = lazy(() => import("./CustomerApp"));
const AdminApp = lazy(() => import("./admin/AdminApp"));
const isAdmin = window.location.pathname.startsWith("/admin");

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Suspense fallback={null}>{isAdmin ? <AdminApp /> : <CustomerApp />}</Suspense>
  </React.StrictMode>,
);

// PWA shell for the customer app only (never caches API responses or claim data).
if ("serviceWorker" in navigator && import.meta.env.PROD && !isAdmin) {
  window.addEventListener("load", () => navigator.serviceWorker.register("/sw.js").catch(() => {}));
}
