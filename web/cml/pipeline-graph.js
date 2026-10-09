import { element, id } from './dom.js';

const namespace = 'http://www.w3.org/2000/svg';
const childKey = tree => tree?.type === 'Pipeline' ? 'steps' : tree?.type === 'ColumnTransformer' ? 'transformers' : tree?.type === 'FeatureUnion' ? 'transformer_list' : null;
export { childKey as pipelineChildKey };

export function pipelineTreeAt(tree, path = []) {
  return path.reduce((node, index) => node?.[childKey(node)]?.[index]?.transformer, tree);
}

export function pipelineColumnsLabel(columns) {
  if (columns?.$tuple) return pipelineColumnsLabel(columns.$tuple);
  if (columns?.$array) return pipelineColumnsLabel(columns.$array);
  if (columns?.$call) return `${columns.$call}(${Object.entries(columns.params || {}).map(([key, value]) => `${key}=${JSON.stringify(value)}`).join(', ')})`;
  if (columns?.$slice) return `slice(${columns.$slice.map(value => value ?? 'None').join(', ')})`;
  if (Array.isArray(columns)) return columns.map(value => typeof value === 'boolean' ? String(value) : value).join(', ') || '[]';
  return columns === undefined ? 'all features' : String(columns);
}

/** The diagram records transformation order and feature routing, never inferred output widths. */
export function pipelineGraphData(tree) {
  const nodes = [{ id: 'input', label: 'Input features', kind: 'boundary', path: null }];
  const edges = [];
  const connect = (sources, target, label = '') => sources.forEach(source => edges.push({ source, target, label }));
  const walk = (node, incoming, path, stepName, columns) => {
    const nodeId = `node-${path.join('-') || 'root'}`;
    const type = typeof node === 'string' ? node : node?.type || 'Transformer';
    nodes.push({ id: nodeId, label: type, name: stepName, path, kind: childKey(node) ? 'container' : 'transformer', params: typeof node === 'object' ? node.params || {} : {} });
    connect(incoming, nodeId, columns === undefined ? '' : pipelineColumnsLabel(columns));
    const key = childKey(node);
    if (!key) return [nodeId];
    const children = node[key] || [];
    if (key === 'steps') {
      let previous = [nodeId];
      children.forEach((item, index) => { previous = walk(item.transformer, previous, [...path, index], item.name); });
      return previous;
    }
    const outputs = children.flatMap((item, index) => walk(item.transformer, [nodeId], [...path, index], item.name, key === 'transformers' ? item.columns : undefined));
    if (key === 'transformers') {
      const remainder = node.params?.remainder ?? 'drop';
      const remainderId = `${nodeId}-remainder`;
      nodes.push({ id: remainderId, label: typeof remainder === 'string' ? `remainder: ${remainder}` : `remainder: ${remainder.type || 'Transformer'}`, kind: 'boundary', path: null });
      edges.push({ source: nodeId, target: remainderId, label: 'unselected features' });
      if (remainder !== 'drop') outputs.push(remainderId);
    }
    const output = `${nodeId}-concat`;
    nodes.push({ id: output, label: 'Concatenate features', kind: 'boundary', path: null });
    connect(outputs.length ? outputs : [nodeId], output);
    return [output];
  };
  connect(walk(tree, ['input'], [], ''), 'output');
  nodes.push({ id: 'output', label: 'Output features', kind: 'boundary', path: null });
  return { nodes, edges };
}

function svgElement(tag, attrs = {}, text) {
  const node = document.createElementNS(namespace, tag);
  for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}

export function renderPipelineGraph(container, tree, { selectedPath, onSelect = () => {}, onSource = () => {}, onMove = () => {}, readOnly = false } = {}) {
  const data = pipelineGraphData(tree);
  const ranks = new Map([['input', 0]]);
  for (let pass = 0; pass < data.nodes.length; pass++) {
    for (const edge of data.edges) if (ranks.has(edge.source)) ranks.set(edge.target, Math.max(ranks.get(edge.target) || 0, ranks.get(edge.source) + 1));
  }
  const rows = new Map();
  data.nodes.forEach(node => { const rank = ranks.get(node.id) || 0; if (!rows.has(rank)) rows.set(rank, []); rows.get(rank).push(node); });
  const positions = new Map();
  let offset = 26;
  let width = 520;
  for (const [, row] of [...rows].sort((a, b) => a[0] - b[0])) {
    const columns = Math.min(4, row.length);
    width = Math.max(width, columns * 242 + 36);
    row.forEach((node, index) => positions.set(node.id, { x: 26 + (index % columns) * 242, y: offset + Math.floor(index / columns) * 105 }));
    offset += Math.ceil(row.length / columns) * 105;
  }
  // Center single-node ranks after the total width is known.
  for (const row of rows.values()) if (row.length === 1) positions.get(row[0].id).x = (width - 218) / 2;
  const titleId = id('pipeline-diagram-title');
  const svg = svgElement('svg', { viewBox: `0 0 ${width} ${offset}`, width, height: offset, role: 'group', 'aria-labelledby': titleId, class: 'cml-pipeline-svg' });
  let pointerDrag = null, suppressClick = false;
  svg.addEventListener('pointermove', event => {
    if (pointerDrag && Math.hypot(event.clientX - pointerDrag.x, event.clientY - pointerDrag.y) > 8) {
      pointerDrag.moved = true; svg.dataset.dragging = 'true';
    }
  });
  svg.addEventListener('pointerup', event => {
    if (!pointerDrag) return;
    const started = pointerDrag; pointerDrag = null; delete svg.dataset.dragging;
    if (!started.moved) return;
    suppressClick = true;
    const target = event.target.closest?.('[data-node-path]');
    if (target?.dataset.nodePath) {
      const destination = JSON.parse(target.dataset.nodePath);
      if (started.path.length && destination.length && JSON.stringify(started.path.slice(0, -1)) === JSON.stringify(destination.slice(0, -1))) onMove(started.path, destination);
    }
  });
  svg.addEventListener('pointercancel', () => { pointerDrag = null; delete svg.dataset.dragging; });
  svg.addEventListener('pointerleave', event => { if (!event.buttons) { pointerDrag = null; delete svg.dataset.dragging; } });
  svg.append(svgElement('title', { id: titleId }, 'Pipeline: направление признаков и последовательность преобразований'));
  const markerId = id('pipeline-arrow');
  const defs = svgElement('defs');
  const marker = svgElement('marker', { id: markerId, markerWidth: 9, markerHeight: 8, refX: 8, refY: 4, orient: 'auto' });
  marker.append(svgElement('path', { d: 'M0,0 L8,4 L0,8 z', class: 'cml-pipeline-arrow' }));
  defs.append(marker); svg.append(defs);
  for (const edge of data.edges) {
    const source = positions.get(edge.source), target = positions.get(edge.target);
    const startX = source.x + 109, startY = source.y + 56, endX = target.x + 109, endY = target.y;
    const midpoint = (startY + endY) / 2;
    svg.append(svgElement('path', { d: `M${startX},${startY} C${startX},${midpoint} ${endX},${midpoint} ${endX},${endY - 4}`, class: 'cml-pipeline-edge', 'marker-end': `url(#${markerId})` }));
    if (edge.label) {
      const caption = svgElement('text', { x: endX, y: endY - 13, 'text-anchor': 'middle', class: 'cml-pipeline-edge-label' }, edge.label.length > 34 ? `${edge.label.slice(0, 31)}…` : edge.label);
      caption.append(svgElement('title', {}, edge.label)); svg.append(caption);
    }
  }
  for (const node of data.nodes) {
    const point = positions.get(node.id), interactive = !readOnly && node.path !== null;
    const selected = interactive && JSON.stringify(node.path) === JSON.stringify(selectedPath);
    const group = svgElement('g', { transform: `translate(${point.x} ${point.y})`, class: `cml-pipeline-node ${node.kind}${selected ? ' selected' : ''}`, 'data-node-path': interactive ? JSON.stringify(node.path) : '', ...(interactive ? { role: 'button', tabindex: 0, 'aria-label': `${node.name ? `${node.name}: ` : ''}${node.label}; редактировать`, 'aria-pressed': selected } : {}) });
    group.append(svgElement('rect', { width: 218, height: 56, rx: 3 }));
    group.append(svgElement('text', { x: 109, y: 23, 'text-anchor': 'middle' }, node.label));
    if (node.name) group.append(svgElement('text', { x: 109, y: 43, 'text-anchor': 'middle', class: 'cml-pipeline-step-name' }, node.name));
    group.append(svgElement('title', {}, `${node.name || node.label}\n${JSON.stringify(node.params || {})}`));
    if (interactive) {
      group.addEventListener('pointerdown', event => { if (event.button === 0) pointerDrag = { path: [...node.path], x: event.clientX, y: event.clientY, moved: false }; });
      group.addEventListener('click', () => { if (suppressClick) { suppressClick = false; return; } onSelect([...node.path]); });
      group.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect([...node.path]); } if (event.key === 'ContextMenu' || (event.shiftKey && event.key === 'F10')) { event.preventDefault(); onSource(node.label, event); } });
      group.addEventListener('contextmenu', event => { event.preventDefault(); onSource(node.label, event); });
    }
    svg.append(group);
  }
  const description = element('details', { className: 'cml-pipeline-graph-description' }, [element('summary', { text: 'Направления признаков текстом' }), element('ul', {}, data.edges.map(edge => {
    const from = data.nodes.find(node => node.id === edge.source), to = data.nodes.find(node => node.id === edge.target);
    return element('li', { text: `${from.name || from.label} → ${to.name || to.label}${edge.label ? `: ${edge.label}` : ''}` });
  }))]);
  container.replaceChildren(element('div', { className: 'cml-pipeline-graph-scroll' }, [svg]), description);
  return data;
}
