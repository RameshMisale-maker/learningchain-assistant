/*
 * Assistant - website embed.
 *
 *   <script src="https://<assistant-host>/widget.js" data-product="<product_id>" defer></script>
 *
 * data-product   : product id (config/products/<id>.yaml); omit for the default product.
 * data-assistant : legacy scope ("products" / "company" / "all"), mapped through the product's `scopes`.
 * data-position  : "right" (default) or "left".
 * data-label     : launcher button text (default: the product's ui.launcher_label).
 *
 * The chat runs in an iframe served by the assistant backend, so no API keys or tokens ever reach the host page.
 * The backend must list the host site in WIDGET_ALLOWED_ORIGINS (frame-ancestors) for the iframe to load.
 */
(function () {
  var script = document.currentScript;
  if (!script || window.__assistantWidget) return;
  window.__assistantWidget = true;
  var origin = new URL(script.src).origin;
  var assistant = script.getAttribute("data-assistant") || "";
  var product = script.getAttribute("data-product") || "";
  var side = script.getAttribute("data-position") === "left" ? "left" : "right";
  var label = script.getAttribute("data-label") || "Ask us";
  var query = [];
  if (assistant) query.push("assistant=" + encodeURIComponent(assistant));
  if (product) query.push("product=" + encodeURIComponent(product));
  var src = origin + "/chat" + (query.length ? "?" + query.join("&") : "");

  var btn = document.createElement("button");
  btn.type = "button";
  btn.textContent = label;
  if (!script.getAttribute("data-label") && window.fetch) {   // default label from the product config
    fetch(origin + "/api/v1/ui" + (query.length ? "?" + query.join("&") : "")).then(function (r) { return r.ok ? r.json() : null; })
      .then(function (ui) { if (ui && ui.launcher_label) btn.textContent = ui.launcher_label; }).catch(function () {});
  }
  btn.setAttribute("aria-expanded", "false");
  btn.style.cssText = "position:fixed;bottom:20px;" + side + ":20px;z-index:2147483000;background:#0f52ba;color:#fff;" +
    "border:0;border-radius:999px;padding:12px 18px;font:600 14px 'Segoe UI',system-ui,sans-serif;" +
    "box-shadow:0 6px 20px rgba(15,82,186,.35);cursor:pointer";

  var frame = null;
  btn.addEventListener("click", function () {
    if (!frame) {
      frame = document.createElement("iframe");
      frame.src = src;
      frame.title = "Assistant";
      frame.setAttribute("allow", "clipboard-write");
      frame.style.cssText = "position:fixed;bottom:80px;" + side + ":20px;z-index:2147483000;width:min(400px,calc(100vw - 40px));" +
        "height:min(620px,calc(100vh - 110px));border:0;border-radius:14px;box-shadow:0 12px 40px rgba(0,0,0,.25);background:#fff";
      document.body.appendChild(frame);
      btn.setAttribute("aria-expanded", "true");
      return;
    }
    var open = frame.style.display !== "none";
    frame.style.display = open ? "none" : "block";
    btn.setAttribute("aria-expanded", String(!open));
  });
  document.body.appendChild(btn);
})();
