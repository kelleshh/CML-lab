import { element, id } from './dom.js';
import { helpButton } from './help.js';

function valueFrom(control, schema) {
  if (control._getValue) return control._getValue();
  if (schema.type === 'bool' || schema.type === 'boolean') return control.checked;
  const raw = control.value.trim();
  if (raw === '' && (schema.nullable || schema.default === null)) return null;
  if (['int', 'integer', 'float', 'number'].includes(schema.type)) return Number(raw);
  if (schema.type === 'select' || schema.type === 'choice') {
    const options = schema.options || schema.choices || [];
    const option = options.find(option => String(option && typeof option === 'object' ? option.value ?? '' : option ?? '') === raw);
    return option && typeof option === 'object' ? option.value : option === undefined ? raw : option;
  }
  if (['list', 'array', 'int_list', 'float_list'].includes(schema.type)) return raw.split(/[,;]+/).map(value => value.trim()).filter(Boolean).map(value => schema.type === 'int_list' || schema.type === 'float_list' || schema.default?.some?.(item => typeof item === 'number') ? Number(value) : value);
  return raw;
}

export class SchemaForm {
  constructor({ onChange = () => {}, className = 'cml-form-grid', definitions = [], task, depth = 0 } = {}) {
    this.onChange = onChange;
    this.className = className;
    this.controls = new Map();
    this.schema = [];
    this.definitions = definitions;
    this.task = task;
    this.depth = depth;
  }

  mount(container, schema = [], values = {}) {
    this.container = container;
    this.schema = schema;
    this.controls.clear();
    const grid = element('div', { className: this.className });
    for (const field of schema) {
      const key = field.key || field.name;
      if (!key) continue;
      const inputId = id(`param-${key}`);
      const label = key;
      const wrapper = element('div', { className: 'cml-field' });
      const heading = element('div', { className: 'cml-field-heading' }, [
        element('label', { htmlFor: inputId, text: label }), helpButton({ ...field, label }),
      ]);
      const current = Object.prototype.hasOwnProperty.call(values, key) ? values[key] : field.default;
      let control;
      if (field.type === 'multiselect') {
        control = element('div', { id: inputId, className: 'cml-column-checks', role: 'group', 'aria-label': label });
        const options = field.options || [];
        for (const option of options) {
          const value = typeof option === 'object' && option !== null ? option.value : option;
          const check = element('input', { type: 'checkbox', value: String(value), checked: (current || []).includes(value) });
          control.append(element('label', { className: 'cml-check' }, [check, element('span', { text: typeof option === 'object' && option !== null ? option.label : String(option) })]));
        }
        control._getValue = () => [...control.querySelectorAll('input:checked')].map(check => options.find(option => String(typeof option === 'object' && option !== null ? option.value : option) === check.value)).map(option => typeof option === 'object' && option !== null ? option.value : option);
        control._setValue = values => control.querySelectorAll('input').forEach(check => { check.checked = (values || []).map(String).includes(check.value); });
        wrapper.classList.add('full');
      } else if (field.type === 'json' && key === 'categories' && field.format !== 'json') {
        control = element('textarea', { id: inputId, rows: 3, value: Array.isArray(current) ? current.map(row => Array.isArray(row) ? row.join(', ') : String(row)).join('\n') : '', placeholder: 'Пусто: автоматически. Категории одного столбца в строке: низкий, средний, высокий.' });
        control._getValue = () => control.value.trim() ? control.value.trim().split(/\n+/).map(line => line.split(',').map(value => value.trim()).filter(Boolean)) : null;
        control._setValue = value => { control.value = Array.isArray(value) ? value.map(row => Array.isArray(row) ? row.join(', ') : String(row)).join('\n') : ''; };
        wrapper.classList.add('full');
      } else if (['json', 'estimator', 'estimators', 'kernel'].includes(field.type)) {
        const example = field.example || (field.type === 'estimator' ? { algorithm_id: 'ridge', params: { alpha: 1 } } : field.type === 'kernel' ? { class: 'RBF', params: { length_scale: 1 } } : field.type === 'estimators' ? [{ name: 'ridge', algorithm_id: 'ridge', params: { alpha: 1 } }] : null);
        control = element('textarea', { id: inputId, name: key, rows: 4, spellcheck: false, value: current === undefined ? 'null' : JSON.stringify(current, null, 2), placeholder: JSON.stringify(example, null, 2) });
        control.classList.add('cml-json-editor');
        const parse = () => {
          try { const value = JSON.parse(control.value.trim() || 'null'); control.setCustomValidity(''); return value; }
          catch { control.setCustomValidity(`${key}: введите корректный JSON.`); throw new Error(`${key}: некорректный JSON.`); }
        };
        control._getValue = parse;
        control._setValue = value => { control.value = JSON.stringify(value ?? null, null, 2); control.setCustomValidity(''); };
        control.addEventListener('input', () => { try { parse(); } catch { /* Keep invalid text editable. */ } });
        wrapper.classList.add('full');
      } else if (field.type === 'models') {
        control = this.modelsControl(current || [], field);
        control.id = inputId;
        wrapper.classList.add('full');
      } else if (['select', 'choice'].includes(field.type)) {
        control = element('select', { id: inputId, name: key });
        for (const option of field.options || field.choices || []) {
          const value = typeof option === 'object' && option !== null && Object.hasOwn(option, 'value') ? option.value : option;
          const optionLabel = option?.label ?? field.option_labels?.[String(value)] ?? (value === null ? 'None' : String(value));
          control.append(element('option', { value: value === null ? '' : String(value), text: optionLabel }));
        }
        control.value = current === null ? '' : String(current ?? '');
      } else if (['bool', 'boolean'].includes(field.type)) {
        control = element('input', { id: inputId, name: key, type: 'checkbox', checked: Boolean(current) });
        wrapper.classList.add('cml-checkbox-field');
      } else {
        const numeric = ['int', 'integer', 'float', 'number'].includes(field.type);
        control = element('input', {
          id: inputId, name: key, type: numeric ? 'number' : 'text', value: Array.isArray(current) ? current.join(', ') : current ?? '',
          min: field.min, max: field.max, step: ['int', 'integer'].includes(field.type) ? (field.step || 1) : numeric ? 'any' : undefined,
          required: field.required ?? (numeric && !field.nullable && field.default !== null), placeholder: field.nullable || field.default === null ? 'None' : undefined,
        });
      }
      control.dataset.parameter = key;
      if (field.description) {
        const descriptionId = id('description');
        control.setAttribute('aria-describedby', descriptionId);
        wrapper.append(heading, control, element('p', { id: descriptionId, className: 'cml-field-note', text: field.description }));
      } else wrapper.append(heading, control);
      control.addEventListener('change', () => { try { this.onChange(this.values(), key); } catch { control.reportValidity?.(); } });
      this.controls.set(key, { control, field });
      grid.append(wrapper);
    }
    container.replaceChildren(grid);
    return this;
  }

  values() {
    return Object.fromEntries([...this.controls].map(([key, { control, field }]) => [key, valueFrom(control, field)]));
  }

  setValues(values) {
    for (const [key, { control, field }] of this.controls) {
      if (!Object.prototype.hasOwnProperty.call(values, key)) continue;
      if (control._setValue) control._setValue(values[key]);
      else if (['bool', 'boolean'].includes(field.type)) control.checked = Boolean(values[key]);
      else control.value = Array.isArray(values[key]) ? values[key].join(', ') : values[key] ?? '';
    }
  }

  valid() { return [...this.controls.values()].every(({ control }) => control.reportValidity ? control.reportValidity() : true); }

  setDisabled(disabled) { for (const { control } of this.controls.values()) { if ('disabled' in control) control.disabled = disabled; control.querySelectorAll?.('input,select,button').forEach(input => { input.disabled = disabled; }); } }

  modelsControl(values, schema) {
    const wrapper = element('div', { className: 'cml-model-components' });
    let components = structuredClone(values);
    let forms = [];
    const compatible = this.definitions.filter(model => model.available !== false && (!this.task || model.tasks?.includes(this.task)) && (this.depth < 2 || !model.params?.some(param => param.type === 'models')));
    const render = () => {
      forms = [];
      const children = components.map((component, index) => {
        const choice = element('select', { 'aria-label': `Алгоритм компонента ${index + 1}` });
        choice.append(...compatible.map(model => element('option', { value: model.id, text: model.name })));
        choice.value = component.algorithm_id || compatible[0]?.id || '';
        component.algorithm_id = choice.value;
        const definition = compatible.find(model => model.id === choice.value);
        const container = element('div');
        const form = new SchemaForm({ onChange: () => { component.params = form.values(); this.onChange(this.values(), schema.key); }, definitions: this.definitions, task: this.task, depth: this.depth + 1 }).mount(container, definition?.params || [], component.params || {});
        forms[index] = form;
        choice.addEventListener('change', () => { components = wrapper._getValue(); components[index] = { algorithm_id: choice.value, params: {} }; render(); this.onChange(this.values(), schema.key); });
        return element('section', { className: 'cml-panel' }, [choice, container, element('button', { type: 'button', className: 'cml-button danger', text: 'Убрать компонент', onclick: () => { components = wrapper._getValue(); components.splice(index, 1); render(); this.onChange(this.values(), schema.key); } })]);
      });
      const add = element('button', { type: 'button', className: 'cml-button', text: 'Добавить модель в ансамбль', disabled: components.length >= 8 || !compatible.length || this.depth >= 3, onclick: () => { components = wrapper._getValue(); components.push({ algorithm_id: compatible[0]?.id, params: {} }); render(); this.onChange(this.values(), schema.key); } });
      wrapper.replaceChildren(...children, add);
    };
    wrapper._getValue = () => components.map((component, index) => ({ ...component, params: forms[index]?.values() || component.params || {} }));
    wrapper._setValue = value => { components = structuredClone(value || []); render(); };
    render();
    return wrapper;
  }
}
