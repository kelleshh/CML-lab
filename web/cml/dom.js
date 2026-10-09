let nextId = 0;

export function id(prefix = 'cml') { return `${prefix}-${++nextId}`; }

export function element(tag, properties = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(properties)) {
    if (value === undefined || value === null) continue;
    if (key === 'text') node.textContent = value;
    else if (key === 'className') node.className = value;
    else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2).toLowerCase(), value);
    else if (key in node && !key.startsWith('aria')) node[key] = value;
    else node.setAttribute(key, String(value));
  }
  for (const child of Array.isArray(children) ? children : [children]) {
    if (child !== undefined && child !== null) node.append(child instanceof node.ownerDocument.defaultView.Node ? child : String(child));
  }
  return node;
}

export function clear(node, children = []) { node.replaceChildren(...children.filter(Boolean)); }

export function formatNumber(value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
  return new Intl.NumberFormat('ru', { maximumFractionDigits: 5 }).format(Number(value));
}

export function badge(text, className = '') { return element('span', { className: `cml-badge ${className}`, text }); }
