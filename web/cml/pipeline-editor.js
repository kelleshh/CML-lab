import { element, id } from './dom.js';
import { helpButton } from './help.js';
import { ApiClient } from './api.js';
import { pipelineChildKey, pipelineTreeAt, renderPipelineGraph } from './pipeline-graph.js';

const defaultSource = 'from sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\n\npipeline = Pipeline([\n    ("scale", StandardScaler()),\n])\n';
const clone = value => value === undefined ? value : structuredClone(value);
const escapeHTML = value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

export function highlightPipelinePython(source) {
  const pattern = /(#.*$)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|\b(from|import|as|True|False|None|pipeline)\b|\b([A-Z][A-Za-z0-9_]*)\b|\b(\d+(?:\.\d+)?(?:e[-+]?\d+)?)\b/gm;
  let result = '', start = 0;
  for (const match of source.matchAll(pattern)) {
    result += escapeHTML(source.slice(start, match.index));
    const tokenClass = match[1] ? 'comment' : match[2] ? 'string' : match[3] ? 'keyword' : match[4] ? 'class' : 'number';
    result += `<span class="cml-python-${tokenClass}">${escapeHTML(match[0])}</span>`;
    start = match.index + match[0].length;
  }
  return result + escapeHTML(source.slice(start)) + '\n';
}

function constructorAt(source, position) {
  const stack = [];
  let quote = null, comment = false;
  for (let index = 0; index < position; index++) {
    const char = source[index];
    if (comment) { if (char === '\n') comment = false; continue; }
    if (quote) { if (char === '\\') index++; else if (char === quote) quote = null; continue; }
    if (char === '#') comment = true;
    else if (char === '"' || char === "'") quote = char;
    else if ('([{'.includes(char)) {
      const name = char === '(' ? source.slice(0, index).match(/([A-Za-z_]\w*)\s*$/)?.[1] : null;
      stack.push(name);
    } else if (')]}'.includes(char)) stack.pop();
  }
  return [...stack].reverse().find(Boolean);
}

export function pipelineCompletions(source, position, classes = []) {
  const before = source.slice(0, position), prefix = before.match(/[A-Za-z_]\w*$/)?.[0] || '';
  const constructor = classes.find(item => (item.name || item.type) === constructorAt(source, position));
  const paramNames = (constructor?.params || []).map(param => ({ label: param.key || param.name, text: `${param.key || param.name}=`, kind: 'parameter', description: param.description || param.help || '' }));
  const names = classes.map(item => ({ label: item.name || item.type, text: `${item.name || item.type}()`, kind: 'constructor', description: item.description || '' }));
  const literals = ['True', 'False', 'None'].map(text => ({ label: text, text, kind: 'literal' }));
  return { prefix, start: position - prefix.length, items: [...paramNames, ...names, ...literals].filter(item => item.label && item.label.startsWith(prefix) && (prefix || item.kind === 'parameter')).slice(0, 40) };
}

function readParameter(control, schema) {
  if (schema.type === 'bool' || schema.type === 'boolean') return control.checked;
  const value = control.value.trim();
  if (control.dataset.encoding === 'json') return value ? JSON.parse(value) : null;
  if (control.dataset.encoding === 'choice') return JSON.parse(value);
  if (['int', 'integer', 'float', 'number'].includes(schema.type)) {
    if (!value) { if (schema.nullable || schema.default === null) return null; throw new Error(`${schema.key || schema.name}: укажите число`); }
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || (['int', 'integer'].includes(schema.type) && !Number.isInteger(numeric))) throw new Error(`${schema.key || schema.name}: требуется ${schema.type === 'int' ? 'целое ' : ''}число`);
    return numeric;
  }
  return value === '' && schema.nullable ? null : control.value;
}

function uiParameterSchema(schema) {
  const type = String(schema.type || ''), plain = type.split(', default')[0];
  if (['bool', 'boolean', 'int', 'integer', 'float', 'number', 'str', 'string', 'text', 'select', 'choice', 'enum'].includes(type)) return schema;
  const options = [...plain.matchAll(/['"]([^'"]+)['"]/g)].map(match => match[1]);
  if (plain.startsWith('{') && options.length && !plain.includes('callable')) return { ...schema, type: 'choice', options: [...new Set(options), ...(plain.includes('None') ? [null] : [])] };
  if (/^bool(?:\s|$)/.test(plain) && typeof schema.default === 'boolean') return { ...schema, type: 'bool' };
  if (/^int(?:\s|$)/.test(plain) && !/float|str|array|tuple|list|dict/.test(plain)) return { ...schema, type: 'int', nullable: plain.includes('None') || schema.default === null };
  if (/^float(?:\s|$)/.test(plain) && !/str|array|tuple|list|dict/.test(plain)) return { ...schema, type: 'float', nullable: plain.includes('None') || schema.default === null };
  if (/^str(?:\s|$)/.test(plain) && !/callable|array|tuple|list|dict/.test(plain)) return { ...schema, type: 'str', nullable: plain.includes('None') || schema.default === null };
  return { ...schema, type: 'json' };
}

function defaultEquivalent(value, expected) {
  const unwrap = item => {
    if (item && typeof item === 'object' && '$tuple' in item) return item.$tuple.map(unwrap);
    if (Array.isArray(item)) return item.map(unwrap);
    if (item && typeof item === 'object') return Object.fromEntries(Object.keys(item).sort().map(key => [key, unwrap(item[key])]));
    return item;
  };
  return JSON.stringify(unwrap(value)) === JSON.stringify(unwrap(expected));
}

function parameterControl(schema, value) {
  const type = schema.type;
  if (type === 'bool' || type === 'boolean') return element('input', { type: 'checkbox', checked: Boolean(value) });
  if (['select', 'choice', 'enum'].includes(type) && (schema.options || schema.choices)?.length) {
    const select = element('select'); select.dataset.encoding = 'choice';
    for (const choice of schema.options || schema.choices) {
      const item = choice && typeof choice === 'object' && Object.hasOwn(choice, 'value') ? choice.value : choice;
      select.append(element('option', { value: JSON.stringify(item), text: item === null ? 'None' : String(item) }));
    }
    const encoded = JSON.stringify(value);
    if (![...select.options].some(option => option.value === encoded)) select.append(element('option', { value: encoded, text: value === null ? 'None' : String(value) }));
    select.value = encoded; return select;
  }
  if (['int', 'integer', 'float', 'number'].includes(type)) return element('input', { type: 'number', value: value ?? '', step: ['int', 'integer'].includes(type) ? 1 : 'any', min: schema.min, max: schema.max, placeholder: schema.nullable ? 'None' : undefined });
  if (['str', 'string', 'text'].includes(type) && (value === null || typeof value === 'string')) return element('input', { type: 'text', value: value ?? '', placeholder: schema.nullable ? 'None: пустое поле' : undefined });
  const control = element('textarea', { rows: 2, value: JSON.stringify(value ?? null), placeholder: 'JSON: null, true, [1, 2], {"key": "value"}' });
  control.dataset.encoding = 'json'; return control;
}

function addButton(label, action, props = {}) { return element('button', { type: 'button', className: 'cml-button', text: label, onclick: action, ...props }); }

/** Server validation is the only Python parser. GUI and source edits share the same canonical tree. */
export function renderPipelineEditor(container, {
  api = new ApiClient(), source, spec, tree, columns = [], catalogue,
  onChange = () => {}, onDraftChange = () => {}, onSave = () => {}, onError = () => {}, lessonId = 'cml-declarative-pipeline', name = 'pipeline',
} = {}) {
  let classes = catalogue?.classes || [], currentTree = clone(tree), currentResult = null, selectedPath = [], destroyed = false, requestNumber = 0;
  let dirty = false, sourceValid = false, busy = false, completion = null, completionIndex = 0, escapeTab = false, dragPath = null;
  const cleanup = [], downloadName = name.replace(/[^A-Za-z0-9_-]/g, '_') || 'pipeline';
  const status = element('p', { className: 'cml-pipeline-status', role: 'status', 'aria-live': 'polite', text: 'Загрузка каталога transformers…' });
  const errors = element('pre', { className: 'cml-pipeline-errors', role: 'alert', hidden: true });
  const toolbar = element('div', { className: 'cml-pipeline-toolbar' });
  const editorId = id('pipeline-python'), suggestionsId = id('pipeline-completions');
  const textarea = element('textarea', { id: editorId, className: 'cml-pipeline-code-input', rows: 18, spellcheck: false, autocapitalize: 'off', autocomplete: 'off', 'aria-label': 'Pipeline Python source', 'aria-autocomplete': 'list', 'aria-controls': suggestionsId, 'aria-expanded': 'false' });
  const highlighted = element('pre', { className: 'cml-pipeline-code-highlight', 'aria-hidden': 'true' });
  const lineNumbers = element('div', { className: 'cml-pipeline-line-numbers', 'aria-hidden': 'true' });
  const completionBox = element('ul', { id: suggestionsId, className: 'cml-pipeline-completions', role: 'listbox', 'aria-label': 'Python completions', hidden: true });
  const graph = element('div', { className: 'cml-pipeline-graph' });
  const inspector = element('section', { className: 'cml-pipeline-inspector', 'aria-label': 'Настройки выбранного transformer' });
  const outline = element('div', { className: 'cml-pipeline-outline', 'aria-label': 'Pipeline steps' });
  const visual = element('section', { className: 'cml-pipeline-visual' }, [element('h3', { text: 'Feature flow' }), element('p', { className: 'cml-field-note', text: 'Нажмите узел, чтобы изменить transformer. В списке перетаскивайте шаги или используйте ↑ / ↓. ПКМ на узле → Source.' }), graph, outline, inspector]);
  const sourcePane = element('section', { className: 'cml-pipeline-source-pane' }, [
    element('div', { className: 'cml-field-heading' }, [element('label', { htmlFor: editorId, text: 'Python source' }), helpButton({ label: 'Pipeline Python source', help: 'Объявите pipeline = Pipeline(...). Разрешены вложенные Pipeline, ColumnTransformer и FeatureUnion; код проверяется сервером до применения.', lesson_id: lessonId })]),
    element('p', { className: 'cml-field-note', text: 'Tab: дополнить имя или вставить 4 пробела. Ctrl+Space: подсказки. Escape, затем Tab: выйти из редактора. Ctrl+Enter: применить код.' }),
    element('div', { className: 'cml-pipeline-code-shell' }, [lineNumbers, element('div', { className: 'cml-pipeline-code-layer' }, [highlighted, textarea]), completionBox]),
  ]);
  const host = element('section', { className: 'cml-pipeline-editor' }, [toolbar, status, errors, element('div', { className: 'cml-pipeline-workspace' }, [sourcePane, visual])]);
  container.replaceChildren(host);

  function setError(error) { errors.textContent = error?.message || String(error); errors.hidden = false; status.textContent = 'Изменения не применены. Исправьте код или настройки.'; onError(error); }
  function clearError() { errors.hidden = true; errors.textContent = ''; }
  function refreshCode() {
    highlighted.innerHTML = highlightPipelinePython(textarea.value);
    lineNumbers.textContent = Array.from({ length: textarea.value.split('\n').length }, (_, index) => index + 1).join('\n');
    highlighted.scrollTop = textarea.scrollTop; highlighted.scrollLeft = textarea.scrollLeft; lineNumbers.scrollTop = textarea.scrollTop;
  }
  function setBusy(value) { busy = value; apply.disabled = value; save.disabled = value || !sourceValid; host.dataset.busy = String(value); }
  function markDirty() { dirty = true; sourceValid = false; save.disabled = true; status.textContent = 'Python source изменен. Нажмите «Применить Python», чтобы обновить диаграмму и настройки.'; clearError(); refreshCode(); onDraftChange(textarea.value); }
  function closeCompletions() { completion = null; completionBox.hidden = true; textarea.setAttribute('aria-expanded', 'false'); textarea.removeAttribute('aria-activedescendant'); }
  function showCompletions(force = false) {
    completion = pipelineCompletions(textarea.value, textarea.selectionStart, classes);
    if (force && !completion.items.length && !completion.prefix) completion.items = classes.map(item => ({ label: item.name, text: `${item.name}()`, kind: 'constructor', description: item.description || '' })).slice(0, 40);
    if (!completion.items.length) { closeCompletions(); return false; }
    completionIndex = 0; completionBox.hidden = false; textarea.setAttribute('aria-expanded', 'true');
    completionBox.replaceChildren(...completion.items.map((item, index) => {
      const option = element('li', { id: `${suggestionsId}-${index}`, role: 'option', 'aria-selected': index === completionIndex }, [element('strong', { text: item.label }), element('span', { text: item.kind === 'parameter' ? 'parameter' : item.kind })]);
      option.title = typeof item.description === 'string' ? item.description : '';
      option.addEventListener('mousedown', event => { event.preventDefault(); completionIndex = index; acceptCompletion(); }); return option;
    }));
    textarea.setAttribute('aria-activedescendant', `${suggestionsId}-0`); return true;
  }
  function acceptCompletion() {
    if (!completion) return;
    const item = completion.items[completionIndex], end = textarea.selectionStart;
    textarea.setRangeText(item.text, completion.start, end, 'end');
    if (item.kind === 'constructor') textarea.setSelectionRange(textarea.selectionStart - 1, textarea.selectionStart - 1);
    closeCompletions(); markDirty(); textarea.focus();
  }

  async function validate(payload, { preserveSource = false, notify = true } = {}) {
    const generation = ++requestNumber;
    setBusy(true); clearError(); status.textContent = 'Проверка Pipeline на сервере…';
    try {
      const result = await api.request('/cml/pipelines/validate', { method: 'POST', body: payload });
      if (destroyed || generation !== requestNumber) return null;
      if (!result.tree || typeof result.source !== 'string') throw new Error('Сервер не вернул дерево Pipeline и Python source. Обновите приложение.');
      currentTree = clone(result.tree); currentResult = clone(result); sourceValid = true; dirty = false;
      if (!preserveSource) { textarea.value = result.source; refreshCode(); }
      if (!pipelineTreeAt(currentTree, selectedPath)) selectedPath = [];
      renderVisual();
      const warnings = result.warnings || [];
      status.textContent = `Pipeline проверен.${warnings.length ? ` ${warnings.map(item => typeof item === 'string' ? item : item.message || JSON.stringify(item)).join(' ')}` : ''}`;
      setBusy(false);
      if (notify) onChange(clone(result));
      return result;
    } catch (error) { if (!destroyed && generation === requestNumber) { sourceValid = false; setError(error); } return null; }
    finally { if (!destroyed && generation === requestNumber) setBusy(false); }
  }
  async function modifyTree(action, { allowDirty = false } = {}) {
    if (dirty && !allowDirty) { setError(new Error('В Python есть непримененные изменения. Сначала нажмите «Применить Python»; текст сохранен.')); return; }
    if (!currentTree) return;
    const next = clone(currentTree);
    try { action(next); } catch (error) { setError(error); return; }
    return validate({ tree: next });
  }
  function descriptor(type) { return classes.find(item => item.name === type || item.type === type); }
  function newTree(type) {
    const node = { type, module: descriptor(type)?.module, params: {} };
    const key = pipelineChildKey(node);
    if (key) node[key] = [{ name: 'identity', transformer: { type: 'FunctionTransformer', module: 'sklearn.preprocessing', params: {} }, ...(key === 'transformers' ? { columns: columns.length ? columns.map(item => typeof item === 'string' ? item : item.name) : { $call: 'make_column_selector', params: { dtype_include: 'number' } } } : {}) }];
    return node;
  }
  function replaceNode(root, path, next) {
    if (!path.length) { Object.keys(root).forEach(key => delete root[key]); Object.assign(root, next); return; }
    const parent = pipelineTreeAt(root, path.slice(0, -1)); parent[pipelineChildKey(parent)][path.at(-1)].transformer = next;
  }
  function moveStep(path, offset) {
    return modifyTree(root => {
      const parent = pipelineTreeAt(root, path.slice(0, -1)), entries = parent[pipelineChildKey(parent)], from = path.at(-1), to = from + offset;
      if (to < 0 || to >= entries.length) return;
      const [entry] = entries.splice(from, 1); entries.splice(to, 0, entry); selectedPath = [...path.slice(0, -1), to];
    });
  }
  function renderOutline(node = currentTree, path = []) {
    const key = pipelineChildKey(node); if (!key) return null;
    const list = element('ol', { className: 'cml-pipeline-step-list', 'aria-label': `${node.type} steps` });
    (node[key] || []).forEach((entry, index) => {
      const itemPath = [...path, index];
      const item = element('li', { draggable: true, className: 'cml-pipeline-step-row', 'data-step-path': JSON.stringify(itemPath) });
      item.append(addButton(`${entry.name}: ${typeof entry.transformer === 'string' ? entry.transformer : entry.transformer.type}`, () => { selectedPath = itemPath; renderVisual(); }, { className: 'cml-pipeline-step-select' }), addButton('↑', () => moveStep(itemPath, -1), { disabled: index === 0, 'aria-label': `${entry.name}: переместить выше` }), addButton('↓', () => moveStep(itemPath, 1), { disabled: index === node[key].length - 1, 'aria-label': `${entry.name}: переместить ниже` }));
      item.addEventListener('dragstart', event => { dragPath = itemPath; event.dataTransfer?.setData('text/plain', JSON.stringify(itemPath)); if (event.dataTransfer) event.dataTransfer.effectAllowed = 'move'; });
      item.addEventListener('dragover', event => { if (dragPath && JSON.stringify(dragPath.slice(0, -1)) === JSON.stringify(path)) { event.preventDefault(); item.classList.add('drop-target'); } });
      item.addEventListener('dragleave', () => item.classList.remove('drop-target'));
      item.addEventListener('drop', event => { event.preventDefault(); item.classList.remove('drop-target'); if (dragPath && JSON.stringify(dragPath.slice(0, -1)) === JSON.stringify(path)) moveStep(dragPath, index - dragPath.at(-1)); dragPath = null; });
      item.addEventListener('dragend', () => { dragPath = null; });
      const nested = renderOutline(entry.transformer, itemPath); if (nested) item.append(nested); list.append(item);
    });
    return list;
  }
  function renderVisual() {
    renderPipelineGraph(graph, currentTree, { selectedPath, onSelect: path => { selectedPath = path; renderVisual(); }, onSource: type => showSource(type), onMove: (from, to) => moveStep(from, to.at(-1) - from.at(-1)) });
    outline.replaceChildren(renderOutline() || element('p', { text: 'Выберите Pipeline, ColumnTransformer или FeatureUnion, чтобы добавить шаги.' }));
    renderInspector();
  }
  function field(label, control, explanation, lesson = lessonId) {
    const controlId = id('pipeline-field'); control.id = controlId;
    return element('div', { className: 'cml-field' }, [element('div', { className: 'cml-field-heading' }, [element('label', { htmlFor: controlId, text: label }), helpButton({ label, help: explanation, lesson_id: lesson })]), control]);
  }
  function renderInspector() {
    const node = pipelineTreeAt(currentTree, selectedPath); if (!node) return;
    const type = typeof node === 'string' ? node : node.type, info = descriptor(type);
    const classSelect = element('select', { 'aria-label': 'Transformer class' });
    const transformers = classes.filter(item => item.kind !== 'estimator');
    classSelect.append(...transformers.map(item => element('option', { value: item.name, text: item.name })));
    if (selectedPath.length) classSelect.append(element('option', { value: 'passthrough', text: 'passthrough' }), element('option', { value: 'drop', text: 'drop' }));
    if (![...classSelect.options].some(option => option.value === type)) classSelect.append(element('option', { value: type, text: type }));
    classSelect.value = type;
    classSelect.addEventListener('change', () => modifyTree(root => replaceNode(root, selectedPath, ['drop', 'passthrough'].includes(classSelect.value) ? classSelect.value : newTree(classSelect.value))));
    const children = [element('h3', { text: selectedPath.length ? 'Transformer settings' : 'Pipeline settings' }), field('class', classSelect, info?.description || 'Класс sklearn transformer. При смене класса параметры выбранного узла сбрасываются.', info?.lesson_id), addButton('Source', () => showSource(type), { disabled: ['drop', 'passthrough'].includes(type) })];
    if (selectedPath.length) {
      const parent = pipelineTreeAt(currentTree, selectedPath.slice(0, -1)), entry = parent[pipelineChildKey(parent)][selectedPath.at(-1)];
      const entryName = element('input', { value: entry.name, pattern: '[A-Za-z_][A-Za-z0-9_]*', required: true });
      entryName.addEventListener('change', () => { if (!entryName.reportValidity()) return; modifyTree(root => { const ancestor = pipelineTreeAt(root, selectedPath.slice(0, -1)); ancestor[pipelineChildKey(ancestor)][selectedPath.at(-1)].name = entryName.value; }); });
      children.push(field('step name', entryName, 'Уникальное имя шага. В sklearn вложенные параметры адресуются через имя__параметр.'));
      if (parent.type === 'ColumnTransformer') {
        const selector = element('textarea', { rows: 2, value: JSON.stringify(entry.columns), placeholder: '["age", "income"]' });
        selector.addEventListener('change', () => { try { const parsed = JSON.parse(selector.value); modifyTree(root => { pipelineTreeAt(root, selectedPath.slice(0, -1)).transformers[selectedPath.at(-1)].columns = parsed; }); } catch (error) { setError(new Error(`columns: ${error.message}. Введите JSON selector.`)); } });
        children.push(field('columns', selector, 'Имена столбцов, индексы, булева маска; selector: {"$call":"make_column_selector","params":{"dtype_include":"number"}}; slice: {"$slice":[0,3,null]}. Эти признаки идут только в выбранную ветвь.'));
        if (columns.length) children.push(element('p', { className: 'cml-field-note', text: `Доступные признаки: ${columns.map(item => typeof item === 'string' ? item : item.name).join(', ')}` }));
      }
    }
    if (typeof node === 'object') {
      const skip = new Set(['steps', 'transformers', 'transformer_list']);
      const definitions = info?.params || [];
      const params = [...definitions.filter(param => !skip.has(param.key || param.name))];
      for (const key of Object.keys(node.params || {})) if (!params.some(param => (param.key || param.name) === key)) params.push({ key, type: 'json', description: 'Параметр конструктора sklearn. Значение вводится в JSON; точный контракт проверяет сервер.' });
      for (const originalSchema of params) {
        const schema = uiParameterSchema(originalSchema);
        const key = schema.key || schema.name;
        const isSet = Object.hasOwn(node.params || {}, key), current = isSet ? node.params[key] : schema.default;
        const control = parameterControl(schema, current); control.dataset.parameter = key;
        control.addEventListener('change', () => { try { if (!control.reportValidity()) return; const value = readParameter(control, schema); modifyTree(root => { const target = pipelineTreeAt(root, selectedPath); target.params ||= {}; target.params[key] = value; }); } catch (error) { setError(new Error(`${key}: ${error.message}`)); } });
        const wrapper = field(key, control, schema.description || schema.help || 'Параметр конструктора sklearn; откройте учебный материал.', schema.lesson_id || info?.lesson_id);
        if (schema.anchor) wrapper.querySelector('.cml-help-button').href += `#${encodeURIComponent(schema.anchor)}`;
        wrapper.append(addButton('Default', () => modifyTree(root => { delete pipelineTreeAt(root, selectedPath).params[key]; }), { className: 'cml-pipeline-default', disabled: schema.required || !isSet || defaultEquivalent(current, schema.default), 'aria-label': `${key}: использовать default sklearn` })); children.push(wrapper);
      }
      const key = pipelineChildKey(node);
      if (key) {
        const classChoice = element('select', { 'aria-label': 'New transformer class' }); classChoice.append(...transformers.map(item => element('option', { value: item.name, text: item.name })));
        const preferred = transformers.find(item => item.name === 'StandardScaler') || transformers[0]; if (preferred) classChoice.value = preferred.name;
        children.push(field('new transformer', classChoice, 'Добавьте самостоятельный transformer или вложенный Pipeline/ColumnTransformer/FeatureUnion.'));
        children.push(addButton('Добавить step', () => modifyTree(root => {
          const target = pipelineTreeAt(root, selectedPath), entries = target[key] ||= [];
          let index = entries.length + 1; while (entries.some(item => item.name === `step_${index}`)) index++;
          entries.push({ name: `step_${index}`, transformer: newTree(classChoice.value), ...(key === 'transformers' ? { columns: columns.length ? [typeof columns[0] === 'string' ? columns[0] : columns[0].name] : [] } : {}) });
          selectedPath = [...selectedPath, entries.length - 1];
        }), { disabled: !classes.length }));
      }
    }
    if (selectedPath.length) children.push(addButton('Удалить step', () => modifyTree(root => {
      const parentPath = selectedPath.slice(0, -1), parent = pipelineTreeAt(root, parentPath); parent[pipelineChildKey(parent)].splice(selectedPath.at(-1), 1); selectedPath = parentPath;
    }), { className: 'cml-button danger' }));
    inspector.replaceChildren(...children);
  }

  async function showSource(type) {
    const previousFocus = document.activeElement;
    const dialog = element('dialog', { className: 'cml-pipeline-source-dialog', 'aria-label': `${type}: sklearn source` });
    const content = element('pre', { text: 'Загрузка исходника установленной библиотеки…', className: 'cml-pipeline-library-source', tabindex: 0 });
    const close = () => { dialog.close?.(); dialog.remove(); previousFocus?.focus?.(); };
    dialog.append(element('h3', { text: `${type} · sklearn source` }), addButton('Закрыть', close), content);
    host.append(dialog);
    if (dialog.showModal) dialog.showModal(); else { dialog.setAttribute('open', ''); dialog.setAttribute('role', 'dialog'); dialog.setAttribute('aria-modal', 'true'); }
    dialog.addEventListener('cancel', event => { event.preventDefault(); close(); });
    dialog.querySelector('button').focus();
    try {
      const result = await api.request(`/cml/pipelines/source?class=${encodeURIComponent(type)}`);
      content.textContent = result.source || result.code || '';
      const path = result.file || result.path || result.module;
      if (path) dialog.insertBefore(element('p', { className: 'cml-field-note', text: `${path}${result.first_line || result.line ? `:${result.first_line || result.line}` : ''}` }), content);
      if (!content.textContent) content.textContent = 'Для этого объекта исходник недоступен.';
    } catch (error) { content.textContent = error.message || String(error); }
  }
  function download(extension) {
    const exportSpec = dirty || !sourceValid ? { format: 'cml.pipeline', version: 1, source: textarea.value } : currentResult?.spec || { format: 'cml.pipeline', version: 1, source: textarea.value };
    const contents = extension === 'py' ? textarea.value : JSON.stringify(exportSpec, null, 2);
    const url = URL.createObjectURL(new Blob([contents], { type: extension === 'py' ? 'text/x-python;charset=utf-8' : 'application/json;charset=utf-8' }));
    const link = element('a', { href: url, download: `${downloadName}.${extension}` }); link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const apply = addButton('Применить Python', () => validate({ source: textarea.value }));
  const save = addButton('Сохранить Pipeline', () => { if (currentResult && sourceValid && !dirty) onSave(clone(currentResult)); }, { disabled: true });
  const copy = addButton('Copy Python', async () => {
    try { if (!globalThis.navigator?.clipboard?.writeText) throw new Error('Clipboard API недоступен'); await globalThis.navigator.clipboard.writeText(textarea.value); status.textContent = 'Python source скопирован.'; }
    catch { textarea.focus(); textarea.select(); status.textContent = 'Python source выделен. Нажмите Ctrl+C.'; }
  });
  const importFile = element('input', { type: 'file', accept: '.py,.json', hidden: true, 'aria-label': 'Import Pipeline' });
  importFile.addEventListener('change', async () => {
    const file = importFile.files?.[0]; if (!file) return;
    try {
      const text = await file.text();
      if (file.name.toLowerCase().endsWith('.json')) { const parsed = JSON.parse(text); textarea.value = parsed.source || parsed.spec?.source || textarea.value; refreshCode(); await validate(parsed.tree ? { tree: parsed.tree } : parsed.source ? { source: parsed.source } : parsed.spec || parsed); }
      else { textarea.value = text; markDirty(); await validate({ source: text }); }
    } catch (error) { setError(error); }
    finally { importFile.value = ''; }
  });
  toolbar.append(apply, save, copy, addButton('Export .py', () => download('py')), addButton('Export .json', () => download('json')), addButton('Import .py / .json', () => importFile.click()), importFile);
  textarea.value = typeof source === 'string' ? source : source?.source || spec?.source || defaultSource;
  refreshCode();
  textarea.addEventListener('input', () => { markDirty(); closeCompletions(); });
  textarea.addEventListener('scroll', refreshCode);
  textarea.addEventListener('click', closeCompletions);
  textarea.addEventListener('keydown', event => {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') { event.preventDefault(); event.stopPropagation(); closeCompletions(); validate({ source: textarea.value }); return; }
    if ((event.ctrlKey || event.metaKey) && (event.code === 'Space' || event.key === ' ')) { event.preventDefault(); showCompletions(true); return; }
    if (event.key === 'Escape') { closeCompletions(); escapeTab = true; return; }
    if (completion && ['ArrowDown', 'ArrowUp'].includes(event.key)) {
      event.preventDefault(); completionIndex = (completionIndex + (event.key === 'ArrowDown' ? 1 : -1) + completion.items.length) % completion.items.length;
      completionBox.querySelectorAll('[role=option]').forEach((option, index) => option.setAttribute('aria-selected', String(index === completionIndex)));
      textarea.setAttribute('aria-activedescendant', `${suggestionsId}-${completionIndex}`); return;
    }
    if (completion && event.key === 'Enter') { event.preventDefault(); acceptCompletion(); return; }
    if (event.key !== 'Tab') { escapeTab = false; return; }
    if (escapeTab) { escapeTab = false; closeCompletions(); return; }
    event.preventDefault();
    if (!event.shiftKey && (completion || (pipelineCompletions(textarea.value, textarea.selectionStart, classes).prefix && showCompletions()))) { acceptCompletion(); return; }
    const position = textarea.selectionStart;
    if (event.shiftKey) {
      const start = textarea.value.lastIndexOf('\n', position - 1) + 1, count = textarea.value.slice(start, start + 4).match(/^ {1,4}/)?.[0].length || 0;
      textarea.setRangeText('', start, start + count, 'preserve');
    } else textarea.setRangeText('    ', position, textarea.selectionEnd, 'end');
    markDirty();
  });
  const ready = (async () => {
    try { if (!catalogue) { const data = await api.request('/cml/pipelines/catalogue'); classes = data.classes || []; } if (destroyed) return null; return await validate(currentTree ? { tree: currentTree } : { source: textarea.value }, { notify: false }); }
    catch (error) { if (!destroyed) { setError(error); setBusy(false); } return null; }
  })();
  return {
    ready, host, textarea,
    getSource: () => textarea.value,
    getSpec: () => clone(currentResult?.spec || { format: 'cml.pipeline', version: 1, source: textarea.value }),
    getTree: () => clone(currentTree),
    isValid: () => sourceValid && !dirty && !busy,
    setSource: async value => { textarea.value = typeof value === 'string' ? value : value?.source || ''; markDirty(); return validate({ source: textarea.value }); },
    setSpec: value => value?.tree ? validate({ tree: clone(value.tree) }) : validate({ source: value?.source || '' }),
    setTree: value => validate({ tree: clone(value) }),
    setColumns: value => { columns = clone(value || []); if (currentTree) renderVisual(); },
    focus: () => textarea.focus(),
    destroy: () => { destroyed = true; requestNumber++; cleanup.forEach(callback => callback()); host.remove(); },
  };
}
