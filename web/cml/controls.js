import { element, id } from './dom.js';
import { helpButton } from './help.js';

export function field(label, control, help = {}) {
  control.id ||= id('field');
  return element('div', { className: 'cml-field' }, [
    element('div', { className: 'cml-field-heading' }, [
      element('label', { htmlFor: control.id, text: label }), helpButton({ label, ...help }),
    ]), control,
  ]);
}

export function select(options = [], value = '', properties = {}) {
  const control = element('select', properties);
  setOptions(control, options, value);
  return control;
}

export function setOptions(control, options = [], value = control.value) {
  control.replaceChildren(...options.map(item => {
    const option = typeof item === 'object' && item !== null ? item : { value: item, label: item };
    return element('option', { value: option.value ?? option.id ?? '', text: option.label ?? option.name ?? String(option.value ?? option.id ?? ''), disabled: option.disabled || false });
  }));
  if ([...control.options].some(option => option.value === String(value))) control.value = value;
}

export function number(value, properties = {}) { return element('input', { type: 'number', value, ...properties }); }

export function checkbox(label, checked = false, help = {}) {
  const control = element('input', { type: 'checkbox', checked });
  const wrapper = element('div', { className: 'cml-field-heading' }, [
    element('label', { className: 'cml-check' }, [control, element('span', { text: label })]), helpButton({ label, ...help }),
  ]);
  return { control, wrapper };
}

export function action(text, onclick, className = '', properties = {}) {
  return element('button', { type: 'button', className: `cml-button ${className}`, text, onclick, ...properties });
}

export function heading(title, help = {}) {
  return element('div', { className: 'cml-panel-heading' }, [element('h2', { text: title }), helpButton({ label: title, ...help })]);
}

export function optionsFromColumns(metadata, { numeric = false, empty = true } = {}) {
  const columns = metadata?.columns || [];
  return [...(empty ? [{ value: '', label: 'Не используется' }] : []), ...columns.filter(column => !numeric || column.numeric).map(column => ({ value: column.name, label: column.name }))];
}

export function numericList(text, label) {
  const result = String(text).split(/[,;\s]+/).filter(Boolean).map(Number);
  if (result.some(value => !Number.isInteger(value) || value < 1)) throw new Error(`${label}: укажите положительные целые числа через запятую.`);
  return [...new Set(result)];
}

export function notice(text, error = false) { return element('p', { className: `cml-notice${error ? ' error' : ''}`, text }); }
