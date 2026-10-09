import { element, formatNumber } from './dom.js';
import { action, field, heading, notice, select } from './controls.js';

function chartLayout(label) {
  const style = getComputedStyle(document.body);
  return { paper_bgcolor: style.getPropertyValue('--panel').trim(), plot_bgcolor: style.getPropertyValue('--panel').trim(), font: { color: style.getPropertyValue('--text').trim(), family: 'Arial, sans-serif', size: 13 }, margin: { l: 65, r: 25, t: 15, b: 55 }, xaxis: { title: 'Настоящий шаг обучения', gridcolor: style.getPropertyValue('--line').trim() }, yaxis: { title: label || 'Измеренная ошибка', gridcolor: style.getPropertyValue('--line').trim() }, legend: { orientation: 'h' }, uirevision: 'training' };
}

function lossTraces(trace, count = trace.length) {
  const frames = trace.slice(0, count);
  return ['train_loss', 'validation_loss'].filter(key => frames.some(frame => typeof frame[key] === 'number')).map((key, index) => ({
    type: 'scatter', mode: 'lines+markers', x: frames.map((frame, i) => frame.step ?? i + 1), y: frames.map(frame => frame[key]),
    name: key === 'train_loss' ? 'Обучение' : 'Выбор настроек', marker: { size: 4 }, line: { color: index ? '#8d521d' : '#315e54', width: 2 },
  }));
}

export class TracePlayer {
  constructor({ trace, plot, container, featureNames = [] }) {
    this.trace = trace; this.plot = plot; this.container = container; this.featureNames = featureNames;
    this.index = trace.length; this.playing = false; this.timer = null;
    this.slider = element('input', { type: 'range', min: 1, max: trace.length, value: trace.length, 'aria-label': 'Показать реальный шаг обучения' });
    this.label = element('span', { className: 'cml-note' });
    this.speed = select([{ value: 500, label: '0.5×' }, { value: 250, label: '1×' }, { value: 125, label: '2×' }, { value: 50, label: '5×' }], 250, { 'aria-label': 'Скорость воспроизведения шагов' });
    this.play = action('▶ Воспроизвести', () => this.toggle(), '', { 'aria-label': 'Воспроизвести сохраненные шаги обучения' });
    this.slider.addEventListener('input', () => { this.pause(); this.index = Number(this.slider.value); this.render(); });
    this.speed.addEventListener('change', () => { if (this.playing) { clearTimeout(this.timer); this.tick(); } });
    container.replaceChildren(element('div', { className: 'cml-trace-controls' }, [this.play, this.slider, this.label, this.speed]), element('p', { className: 'cml-note', text: 'Воспроизводятся сохраненные измерения реальных шагов. Новое обучение при воспроизведении не запускается.' }));
    this.render();
  }

  async render() {
    this.slider.value = this.index;
    const frame = this.trace[this.index - 1];
    this.label.textContent = `Шаг ${frame.step ?? this.index} из ${this.trace.at(-1)?.step ?? this.trace.length} · ${frame.label || 'ошибка'}: ${formatNumber(frame.train_loss)}`;
    if (globalThis.Plotly) await Plotly.react(this.plot, lossTraces(this.trace, this.index), chartLayout(frame.label), { responsive: true, displaylogo: false });
  }

  toggle() {
    if (this.playing) { this.pause(); return; }
    if (this.index >= this.trace.length) this.index = 1;
    this.playing = true; this.play.textContent = 'Ⅱ Пауза'; this.play.setAttribute('aria-label', 'Приостановить воспроизведение');
    this.tick();
  }

  tick() {
    if (!this.playing) return;
    this.render();
    if (this.index >= this.trace.length) { this.pause(); return; }
    this.timer = setTimeout(() => { this.index += 1; this.tick(); }, Number(this.speed.value));
  }

  pause() { this.playing = false; clearTimeout(this.timer); this.play.textContent = '▶ Воспроизвести'; this.play.setAttribute('aria-label', 'Воспроизвести сохраненные шаги обучения'); }
  destroy() { this.pause(); }
}

export class LiveTrainingView {
  constructor({ container }) { this.container = container; this.frames = []; }

  begin(modelName) {
    this.frames = [];
    this.plot = element('div', { className: 'cml-plot' });
    this.message = element('p', { className: 'cml-note', 'aria-live': 'polite', text: 'Расчет поставлен в очередь.' });
    this.details = notice('График появляется, когда алгоритм передает измерения промежуточных шагов. Для алгоритмов с одним вызовом обучения отображается состояние реального расчета.');
    this.container.replaceChildren(element('section', { className: 'cml-panel' }, [heading(`Обучение: ${modelName}`, { help: 'События приходят из настоящего процесса обучения. Проверочная ошибка измеряется по части выбора; итоговая тестовая часть остается скрытой.', lesson_id: '26-cross-validation' }), this.message, this.details, this.plot]));
    this.plot.hidden = true;
  }

  async update(job) {
    const events = job.events || [];
    const trace = events.filter(event => event.type === 'frame' && event.frame).map(event => event.frame);
    this.message.textContent = job.message || events.at(-1)?.message || `Состояние: ${job.status}`;
    if (!trace.length) return;
    this.frames = trace;
    this.plot.hidden = false;
    this.details.textContent = `${trace.at(-1).label || 'Ошибка'} измерена после ${trace.at(-1).step || trace.length} реальных шагов обучения.`;
    if (globalThis.Plotly) await Plotly.react(this.plot, lossTraces(trace), chartLayout(trace.at(-1).label), { responsive: true, displaylogo: false });
  }
}
