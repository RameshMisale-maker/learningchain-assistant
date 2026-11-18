# Deliverable 11 — Integrating the Assistant into the Sapphirus Company Portal

The Streamlit app is a pilot/admin client. The portal integrates with the **same REST API** — nothing in the
backend changes.

## 1. Recommended topology

```
Browser ─▶ Portal (React)  ─▶  Portal backend / API gateway (SSO session)  ─▶  Assistant API (private network)
                                     adds X-Portal-User, X-Portal-Roles, X-Proxy-Secret
```

* Keep the assistant API on a private network; only the portal backend/gateway calls it.
* Set `AUTH_MODE=proxy_headers` and a long `PROXY_SHARED_SECRET`. The gateway authenticates the user with the
  portal's SSO (Azure AD / Okta / Keycloak via OIDC or SAML) and forwards the identity and roles.
* Map IdP groups to assistant roles at the gateway, e.g. `AAD-All-Employees → employee`, `AAD-HR → hr`,
  `AAD-Assistant-Admins → admin`; external customers → no user header (→ `public`).
* Alternative for system-to-system use: `AUTH_MODE=token` with per-application tokens
  (`python -m backend.security.manage_users create --username portal-service --roles employee`).

## 2. API contract (OpenAPI at `/docs`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/me` | current user, roles, accessible KBs |
| GET | `/api/v1/suggestions` | role-filtered suggested questions |
| POST | `/api/v1/chat` | `{message, session_id?}` → answer, citations, KBs, answer_type, message_id, session_id |
| GET / DELETE | `/api/v1/chat/sessions/{id}[/messages]` | history / clear conversation |
| POST | `/api/v1/feedback` | `{message_id, rating: helpful|not_helpful, comment?}` |
| * | `/api/v1/admin/...` | KB stats, documents upload/replace/delete, sync/re-index, feedback, audit, users, retention (admin only) |

Example response (abridged):

```json
{
  "answer": "Employees are entitled to 8 days of casual leave per calendar year ... [S1]",
  "answer_type": "generated",
  "knowledge_bases": ["hr_policies"],
  "citations": [{"id": "S1", "doc_name": "HR-POL-003 Casual Leave Policy", "page": null,
                 "section": "3. Policy Rules", "source_file": "knowledge_bases/hr_policies/HR-POL-003_Casual_Leave_Policy.docx",
                 "version": 1, "is_sample": true}],
  "session_id": "4f0c…", "message_id": 812, "confidence": 0.82, "contains_sample_data": true
}
```

## 3. React widget (drop-in example)

```tsx
// SapphirusAssistant.tsx - minimal chat widget calling the portal backend proxy (/assistant-api -> assistant API)
import { useState } from "react";

type Citation = { id: string; doc_name: string; page?: number | null; section?: string };
type Msg = { role: "user" | "assistant"; text: string; citations?: Citation[]; messageId?: number };

export default function SapphirusAssistant() {
  const [msgs, setMsgs] = useState<Msg[]>([{ role: "assistant",
    text: "Welcome to Sapphirus AI Assistant. Ask me anything about our company, products, services, and policies." }]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function send() {
    const message = input.trim();
    if (!message || busy) return;
    setMsgs(m => [...m, { role: "user", text: message }]);
    setInput(""); setBusy(true);
    try {
      const res = await fetch("/assistant-api/api/v1/chat", {
        method: "POST", credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, session_id: sessionId }),
      });
      if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText);
      const data = await res.json();
      setSessionId(data.session_id);
      setMsgs(m => [...m, { role: "assistant", text: data.answer, citations: data.citations, messageId: data.message_id }]);
    } catch (e: any) {
      setMsgs(m => [...m, { role: "assistant", text: `Sorry, something went wrong: ${e.message}` }]);
    } finally { setBusy(false); }
  }

  async function rate(messageId: number, rating: "helpful" | "not_helpful") {
    await fetch("/assistant-api/api/v1/feedback", { method: "POST", credentials: "include",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message_id: messageId, rating }) });
  }

  return (
    <div className="assistant">
      {msgs.map((m, i) => (
        <div key={i} className={`msg ${m.role}`}>
          <div>{m.text}</div>
          {m.citations?.length ? <ul className="sources">{m.citations.map(c =>
            <li key={c.id}>[{c.id}] {c.doc_name}{c.page ? `, p. ${c.page}` : ""}{c.section ? ` - ${c.section}` : ""}</li>)}</ul> : null}
          {m.messageId ? <div className="feedback">
            <button onClick={() => rate(m.messageId!, "helpful")}>👍</button>
            <button onClick={() => rate(m.messageId!, "not_helpful")}>👎</button></div> : null}
        </div>))}
      {busy && <div className="msg assistant">Searching the knowledge base…</div>}
      <input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === "Enter" && send()}
             placeholder="Ask about Sapphirus, our products, services or policies…" />
    </div>
  );
}
```

Render answers as Markdown (e.g. `react-markdown`) so comparison tables display correctly; sanitize HTML.

## 4. Portal backend proxy (Node/Express example)

```js
// Forwards the logged-in user's identity to the assistant API. Never expose PROXY_SHARED_SECRET to the browser.
app.use("/assistant-api", requireLogin, createProxyMiddleware({
  target: process.env.ASSISTANT_API_URL, changeOrigin: true, pathRewrite: { "^/assistant-api": "" },
  onProxyReq: (proxyReq, req) => {
    proxyReq.setHeader("X-Proxy-Secret", process.env.ASSISTANT_PROXY_SECRET);
    proxyReq.setHeader("X-Portal-User", req.user.email);
    proxyReq.setHeader("X-Portal-Roles", mapGroupsToRoles(req.user.groups).join(","));  // e.g. "employee,hr"
    proxyReq.removeHeader("authorization");
  },
}));
```

Strip any client-supplied `X-Portal-*` headers at the gateway so users cannot spoof roles.

## 5. Integration checklist

- [ ] `AUTH_MODE=proxy_headers`, `PROXY_SHARED_SECRET` set; API reachable only from the gateway
- [ ] Group → role mapping agreed with HR/IT (who may see HR policies?)
- [ ] `CORS_ORIGINS` limited to the portal origin (only needed if the browser calls the API directly)
- [ ] Display the `[Sample data]` / `contains_sample_data` flag until real datasets are loaded
- [ ] Feedback buttons wired to `/feedback`; admins review in the admin console
- [ ] Chat retention period approved by Legal/HR (`CHAT_RETENTION_DAYS`)
