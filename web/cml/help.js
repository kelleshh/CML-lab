import { element, id } from './dom.js';

export function lessonUrl(lessonId) { return `/lesson?id=${encodeURIComponent(lessonId || '01-prediction')}`; }

export function helpButton({ label, help, lesson_id, lessonId, help_key, helpKey } = {}) {
  const panelId = id('help');
  const url = lessonUrl(lesson_id || lessonId);
  const summary = typeof help === 'string' ? help : help?.summary || help?.text || label || 'Откройте учебный материал с объяснением этой настройки.';
  const container = element('span', { className: 'cml-help' });
  const panel = element('span', { id: panelId, className: 'cml-help-panel', role: 'tooltip' }, [
    element('strong', { text: label || 'Что означает настройка' }),
    element('span', { text: summary }),
    element('span', { className: 'cml-help-hint', text: 'Нажмите ?, чтобы открыть учебный материал в новой вкладке.' }),
  ]);
  const link = element('a', {
    className: 'cml-help-button', href: url, target: '_blank', rel: 'noopener', text: '?',
    'aria-label': `Объяснение: ${label || 'настройка'}; новая вкладка`, 'aria-describedby': panelId,
  });
  const key = help_key || helpKey;
  if (key) link.dataset.helpKey = key;
  const dismiss = () => { container.dataset.dismissed = 'true'; };
  const reopen = () => { delete container.dataset.dismissed; };
  container.addEventListener('keydown', event => { if (event.key === 'Escape') { dismiss(); event.stopPropagation(); } });
  container.addEventListener('mouseenter', reopen);
  link.addEventListener('focus', reopen);
  container.append(link, panel);
  return container;
}

export function fieldHeading(label, help = {}) {
  return element('span', { className: 'cml-field-heading' }, [element('span', { text: label }), helpButton({ label, ...help })]);
}
