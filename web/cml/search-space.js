export function parseChoices(text, schema = {}) {
  const raw = text.trim();
  if (!raw) throw new Error('Введите хотя бы одно значение параметра.');
  if (raw.startsWith('[')) {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed) || !parsed.length) throw new Error('Список вариантов не должен быть пустым.');
    return parsed;
  }
  return raw.split(/[,;]+/).map(value => value.trim()).filter(Boolean).map(value => {
    if (value === 'null' || value === 'None') return null;
    if (value === 'true' || value === 'True') return true;
    if (value === 'false' || value === 'False') return false;
    if (['int', 'integer', 'float', 'number'].includes(schema.type)) {
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) throw new Error(`${schema.key}: вариант должен быть конечным числом.`);
      return numeric;
    }
    // Mixed native schemas include numbers alongside strings (gamma, alpha).
    if (value && Number.isFinite(Number(value))) return Number(value);
    if (/^".*"$/.test(value)) return JSON.parse(value);
    return value;
  });
}

export function searchDefinition(row, schema) {
  const kind = row.mode?.value || 'choices';
  if (kind === 'choices') return parseChoices(row.values.value, schema);
  const low = Number(row.low.value), high = Number(row.high.value);
  if (!Number.isFinite(low) || !Number.isFinite(high) || low >= high) throw new Error(`${row.key.value}: нужны конечные low < high.`);
  if (kind === 'int' && (!Number.isInteger(low) || !Number.isInteger(high))) throw new Error('Границы int должны быть целыми.');
  if (row.log.checked && low <= 0) throw new Error('Логарифмический диапазон требует low > 0.');
  return { type: kind, low, high, log: row.log.checked };
}
