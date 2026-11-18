/*
 * Sapphirus Assistant - website embed.
 *
 *   <script src="https://chat.sapphirus.in/widget.js" data-assistant="products" defer></script>
 *
 * data-assistant : "products" (LearningChain help), "company" (company information) or omit for a switchable chat.
 * data-position  : "right" (default) or "left".
 * data-label     : launcher button text (default "Ask Sapphirus").
 *
 * The chat runs in an iframe served by the assistant backend, so no API keys or tokens ever reach the host page.
 * The backend must list the host site in WIDGET_ALLOWED_ORIGINS (frame-ancestors) for the iframe to load.
 */
(function () {
  var script = document.currentScript;
  if (!script || window.__sapphirusWidget) return;
  window.__sapphirusWidget = true;
  var origin = new URL(script.src).origin;
  var assistant = script.getAttribute("data-assistant") || "";
  var side = script.getAttribute("data-position") === "left" ? "left" : "right";
  var label = script.getAttribute("data-label") || "Ask Sapphirus";
  var src = origin + "/chat" + (assistant ? "?assistant=" + encodeURIComponent(assistant) : "");

  var btn = document.createElement("button");
  btn.type = "button";
  btn.textContent = label;
  btn.setAttribute("aria-expanded", "false");
  btn.style.cssText = "position:fixed;bottom:20px;" + side + ":20px;z-index:2147483000;background:#0f52ba;color:#fff;" +
    "border:0;border-radius:999px;padding:12px 18px;font:600 14px 'Segoe UI',system-ui,sans-serif;" +
    "box-shadow:0 6px 20px rgba(15,82,186,.35);cursor:pointer";

  var frame = null;
  btn.addEventListener("click", function () {
    if (!frame) {
      frame = document.createElement("iframe");
      frame.src = src;
      frame.title = "Sapphirus Assistant";
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
