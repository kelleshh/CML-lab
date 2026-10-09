"""Tabular dataset adapters, reproducible generators and durable dataset storage."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from io import BytesIO, StringIO
import inspect
import json
import math
from pathlib import Path
import re
from typing import Any
from uuid import uuid4
import zipfile

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn import datasets as sklearn_datasets
from scipy import sparse


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_ROWS = 100_000
MAX_COLUMNS = 1_000
MAX_CELLS = 5_000_000
MAX_CATEGORIES = 512
_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


@dataclass(frozen=True)
class DataBundle:
    X: pd.DataFrame
    y: np.ndarray
    feature_names: list[str]
    target_name: str
    name: str
    meta: dict[str, Any]


_SYNTHETIC = {
    "linear": ("Линейные данные", "Числовая цель равна сумме вкладов признаков и случайного шума."),
    "correlated": ("Похожие признаки", "Сильная корреляция признаков делает отдельные коэффициенты нестабильными."),
    "sparse": ("Много бесполезных признаков", "Часть истинных коэффициентов равна нулю. Сравните отбор признаков."),
    "nonlinear": ("Нелинейная зависимость", "К линейной зависимости добавлен квадрат первого признака."),
    "heteroscedastic": ("Неодинаковый шум", "Разброс ошибки увеличивается вместе с модулем первого признака."),
    "positive": ("Положительная цель", "Экспонента линейного предиктора с шумом: подходит для Gamma и Tweedie."),
    "counts": ("Количество событий", "Число событий из распределения Пуассона с экспоненциальной связью."),
    "grouped": ("Группы признаков", "Активные признаки объединены в группы; часть групп не влияет на цель."),
    "fused": ("Соседние коэффициенты", "Истинные коэффициенты постоянны внутри последовательных блоков."),
}
_KNOWN = {
    "load_diabetes": ("regression", True, "Диабет", "442 наблюдения; числовой показатель прогрессирования болезни."),
    "load_linnerud": ("regression", True, "Физические упражнения", "20 наблюдений, три числовые цели; выберите одну."),
    "load_iris": ("classification", True, "Ирисы", "Классы цветов не являются числовой целью регрессии."),
    "load_wine": ("classification", True, "Вино", "Тип вина является классом; для регрессии выберите числовое измерение."),
    "load_breast_cancer": ("classification", True, "Диагностика опухолей", "Диагноз является классом; доступен выбор числового измерения."),
    "load_digits": ("image", True, "Рукописные цифры", "Изображения 8×8 представлены 64 числовыми пикселями; номер цифры является классом."),
    "fetch_california_housing": ("regression", True, "Жилье в Калифорнии", "20 640 наблюдений; цель — стоимость жилья. Требуется загрузка."),
    "fetch_openml": ("mixed", True, "OpenML", "Укажите имя или ID набора. Назначение цели зависит от набора."),
    "fetch_covtype": ("classification", True, "Тип лесного покрова", "Большой набор; в интерфейс попадает воспроизводимая выборка. Тип покрова — класс."),
    "fetch_kddcup99": ("classification", True, "KDD Cup 1999", "Сетевые соединения, числовые и категориальные признаки; цель — класс."),
    "fetch_20newsgroups": ("text", False, "Новостные группы", "Требуется отдельный преобразователь текста; текущая лаборатория работает с таблицами."),
    "fetch_20newsgroups_vectorized": ("text", False, "Векторизованные новости", "Разреженная текстовая матрица не поддерживается в табличной лаборатории."),
    "fetch_lfw_people": ("image", False, "Лица LFW", "Требуется отдельная обработка изображений и явная числовая цель."),
    "fetch_lfw_pairs": ("image", False, "Пары лиц LFW", "Набор для проверки пар изображений, не табличная регрессия."),
    "fetch_olivetti_faces": ("image", False, "Лица Olivetti", "Изображения и идентификаторы людей; отдельный маршрут изображений не реализован."),
    "load_sample_image": ("image", False, "Пример изображения", "Одно изображение не задает таблицу наблюдений для регрессии."),
    "load_sample_images": ("image", False, "Примеры изображений", "Изображения не имеют числовой цели регрессии."),
    "load_files": ("text", False, "Текстовые файлы", "Загрузчик локального каталога текста; используйте импорт таблицы."),
    "fetch_rcv1": ("text", False, "Reuters RCV1", "Большая разреженная текстовая матрица с классами документов."),
    "fetch_species_distributions": ("spatial", False, "Распределения видов", "Географические сетки требуют отдельного адаптера наблюдений."),
    "load_svmlight_file": ("sparse", False, "SVMlight файл", "Импорт разреженных SVMlight-файлов пока отсутствует; поддержаны CSV и Excel."),
    "load_svmlight_files": ("sparse", False, "SVMlight файлы", "Импорт разреженных SVMlight-файлов пока отсутствует; поддержаны CSV и Excel."),
}
_MAKE_SUPPORTED = {"make_regression", "make_friedman1", "make_friedman2", "make_friedman3", "make_sparse_uncorrelated", "make_low_rank_matrix", "make_classification", "make_blobs", "make_circles", "make_moons", "make_gaussian_quantiles"}
_MAKE_REGRESSION = {"make_regression", "make_friedman1", "make_friedman2", "make_friedman3", "make_sparse_uncorrelated"}
_GENERATOR_PARAMS = [
    {"key": "n_samples", "label": "Наблюдений", "type": "int", "default": 180, "min": 12, "max": 10000, "step": 12, "help": "Каждая строка — одно наблюдение."},
    {"key": "n_features", "label": "Признаков", "type": "int", "default": 2, "min": 1, "max": 100, "step": 1, "help": "Столбцы, на основе которых модель предсказывает цель."},
    {"key": "noise", "label": "Шум", "type": "float", "default": 8, "min": 0, "max": 100, "step": 1, "help": "Стандартное отклонение случайной ошибки в единицах цели."},
    {"key": "correlation", "label": "Корреляция признаков", "type": "float", "default": 0.2, "min": 0, "max": 0.99, "step": 0.01, "help": "Ближе к 1 — признаки сильнее повторяют друг друга."},
    {"key": "outliers", "label": "Доля выбросов", "type": "float", "default": 0, "min": 0, "max": 0.4, "step": 0.01, "help": "Случайной части строк добавляется большая ошибка цели."},
    {"key": "sparsity", "label": "Доля нулевых коэффициентов", "type": "float", "default": 0.7, "min": 0, "max": 0.95, "step": 0.05, "help": "Для разреженного набора определяет долю бесполезных признаков."},
    {"key": "scale_spread", "label": "Разница масштабов", "type": "float", "default": 0, "min": 0, "max": 6, "step": 0.5, "help": "Разница порядков величины признаков; 3 означает отношение масштабов 1000."},
    {"key": "seed", "label": "Случайное зерно", "type": "int", "default": 42, "min": 0, "max": 1000000, "step": 1, "help": "Одинаковое зерно и настройки дают одинаковые данные."},
]


class DataService:
    """Own dataset IDs and never interpret client identifiers as filesystem paths."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def catalogue(self) -> list[dict[str, Any]]:
        entries = [
            {"id": name, "name": name, "label": label, "kind": "synthetic", "task": "regression", "supported": True, "requires_network": False, "description": description, "params": _GENERATOR_PARAMS}
            for name, (label, description) in _SYNTHETIC.items()
        ]
        # Introspection keeps the catalogue complete for the installed sklearn version.
        for name in sorted(dir(sklearn_datasets)):
            if not name.startswith(("load_", "fetch_", "make_")):
                continue
            try:
                loader = getattr(sklearn_datasets, name)
            except (AttributeError, ImportError):
                continue
            if not callable(loader):
                continue
            if name.startswith("make_"):
                task = "regression" if name in _MAKE_REGRESSION else "classification" if name != "make_low_rank_matrix" else "matrix"
                supported = name in _MAKE_SUPPORTED
                label = name
                description = "Генератор scikit-learn. " + ("Можно выбрать числовую цель из столбцов." if supported else "Структура результата требует отдельного адаптера; доступен в каталоге для ознакомления.")
            else:
                task, supported, label, description = _KNOWN.get(name, ("other", False, name, "Для этого формата пока нет безопасного табличного адаптера."))
            entries.append({"id": name, "name": name, "label": label, "kind": "synthetic" if name.startswith("make_") else "fetch" if name.startswith("fetch_") else "builtin", "task": task, "supported": supported, "requires_network": name.startswith("fetch_"), "description": description, "reason": None if supported else description, "source": f"https://scikit-learn.org/stable/modules/generated/sklearn.datasets.{name}.html", "params": _GENERATOR_PARAMS if name.startswith("make_") else []})
        return entries

    def load(self, spec: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(spec, dict):
            raise ValueError("Настройки набора должны быть объектом.")
        kind, name = spec.get("kind", "synthetic"), str(spec.get("name", "linear"))
        params = spec.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError("Параметры набора должны быть объектом.")
        if kind == "custom":
            rows = spec.get("rows")
            if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
                raise ValueError("Для редактирования передайте непустой список строк-объектов.")
            frame = pd.DataFrame(rows)
            target = spec.get("target")
            if target is not None and target not in frame.columns:
                raise ValueError("Указанная цель отсутствует в отредактированных данных.")
            return self._store(frame, spec.get("name", "Отредактированные данные"), target, {"source": "custom", "description": "Строки введены или изменены вручную."})
        if kind == "synthetic" and name in _SYNTHETIC:
            frame, metadata = self._synthetic(name, params)
            return self._store(frame, _SYNTHETIC[name][0], "y", metadata)
        if kind == "synthetic" and name.startswith("make_"):
            return self._make_sklearn(name, params)
        if kind in {"builtin", "fetch", "openml"}:
            return self._load_sklearn(kind, name, params)
        raise ValueError("Неизвестный источник данных. Доступны генератор, sklearn, OpenML, CSV и Excel.")

    def import_bytes(self, data: bytes, filename: str) -> dict[str, Any]:
        if not data or len(data) > MAX_UPLOAD_BYTES:
            raise ValueError("Файл пуст или превышает 25 МБ.")
        extension = Path(filename).suffix.lower()
        try:
            if extension in {".csv", ".tsv", ".txt"}:
                frame = self._read_csv(data, extension)
            elif extension in {".xlsx", ".xls"}:
                if extension == ".xlsx":
                    with zipfile.ZipFile(BytesIO(data)) as archive:
                        sizes = [entry.file_size for entry in archive.infolist()]
                        if len(sizes) > 5000 or sum(sizes) > 100 * 1024 * 1024:
                            raise ValueError("Excel-файл слишком велик после распаковки.")
                raw = pd.read_excel(BytesIO(data), header=None, nrows=MAX_ROWS + 2)
                if raw.empty or raw.iloc[0].isna().any():
                    raise ValueError("В Excel отсутствуют названия колонок.")
                headers = [str(value).strip() for value in raw.iloc[0]]
                if len(headers) != len(set(headers)):
                    raise ValueError("В Excel повторяются названия колонок.")
                frame = raw.iloc[1:].reset_index(drop=True)
                frame.columns = headers
            else:
                raise ValueError("Поддержаны CSV, TSV, XLSX и XLS. Выберите табличный файл.")
        except ImportError as exc:
            raise ValueError("Для этого формата Excel не установлен нужный модуль. Используйте XLSX или CSV.") from exc
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Не удалось прочитать таблицу: {exc}") from exc
        frame = self._validate_frame(frame)
        target = next((column for column in frame.columns if str(column).lower() in {"y", "target", "цель"} and self._numeric(frame[column])), None)
        return self._store(frame, Path(filename).name[:200], target, {"source": "upload", "description": "Ваш набор данных. Выберите числовую цель и признаки перед обучением."})

    def describe(self, dataset_id: str) -> dict[str, Any]:
        return self._read(dataset_id)[1]

    def rows(self, dataset_id: str, offset: int = 0, limit: int = 500) -> dict[str, Any]:
        """Return a bounded original-data viewport with stable zero-based indices."""
        offset = self._number({"offset": offset}, "offset", 0, 0, MAX_ROWS, True)
        limit = self._number({"limit": limit}, "limit", 500, 1, 500, True)
        frame, meta = self._read(dataset_id)
        records = json.loads(frame.iloc[offset:offset + limit].to_json(orient="records", date_format="iso", double_precision=15))
        return {"id": dataset_id, "columns": meta["columns"], "rows": records, "total": len(frame), "offset": offset, "limit": limit, "target": meta["default_target"]}

    def update_rows(self, dataset_id: str, changes: list[dict[str, Any]], additions: list[dict[str, Any]] | None = None, deletes: list[int] | None = None) -> dict[str, Any]:
        """Create a new dataset version; indices refer to the unchanged original."""
        additions = [] if additions is None else additions
        deletes = [] if deletes is None else deletes
        if not isinstance(changes, list) or len(changes) > 500:
            raise ValueError("Передайте список максимум из 500 изменений строк.")
        if not isinstance(additions, list) or len(additions) > 100:
            raise ValueError("За один запрос можно добавить максимум 100 строк.")
        if not isinstance(deletes, list) or len(deletes) > 500:
            raise ValueError("За один запрос можно удалить максимум 500 строк.")
        frame, meta = self._read(dataset_id)
        columns = set(frame.columns)
        deleted: set[int] = set()
        for index in deletes:
            if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= len(frame):
                raise ValueError("Индекс удаляемой строки отсутствует в исходном наборе.")
            if index in deleted:
                raise ValueError("Одна строка указана для удаления несколько раз.")
            deleted.add(index)
        seen: set[int] = set()
        # An object buffer allows atomic dtype validation after every edit is applied.
        edited = frame.astype(object)
        for change in changes:
            if not isinstance(change, dict):
                raise ValueError("Изменение строки должно содержать index и values.")
            index, values = change.get("index"), change.get("values")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= len(frame):
                raise ValueError("Индекс изменяемой строки отсутствует в исходном наборе.")
            if index in deleted:
                raise ValueError("Нельзя одновременно изменять и удалять одну строку.")
            if index in seen:
                raise ValueError("Объедините изменения одной строки в один объект values.")
            if not isinstance(values, dict) or not values:
                raise ValueError("Передайте непустой объект новых значений строки.")
            if set(values) - columns:
                raise ValueError("В изменениях есть неизвестная колонка.")
            seen.add(index)
            for column, value in values.items():
                if isinstance(value, (dict, list, tuple, set)):
                    raise ValueError("Ячейка должна содержать число, текст или пропуск.")
                edited.at[index, column] = value
        for row in additions:
            if not isinstance(row, dict) or set(row) != columns:
                raise ValueError("Добавляемая строка должна содержать все исходные колонки и не добавлять новых.")
        if deleted:
            edited = edited.drop(index=sorted(deleted))
        if additions:
            edited = pd.concat([edited, pd.DataFrame(additions, columns=frame.columns)], ignore_index=True)
        if not changes and not additions and not deleted:
            raise ValueError("Нет изменений для сохранения.")
        details = {key: value for key, value in meta.items() if key not in {"id", "name", "rows", "columns", "column_names", "preview", "targets", "default_target", "feature_names"}}
        details["parent_id"] = dataset_id
        details["edited"] = True
        details["edit_summary"] = {"changed": len(changes), "added": len(additions), "deleted": len(deleted)}
        return self._store(edited, meta["name"], meta["default_target"], details)

    def resolve(self, dataset_id: str, target: str | None = None, features: list[str] | None = None) -> DataBundle:
        frame, meta = self._read(dataset_id)
        target = target or meta.get("default_target")
        if target is None:
            raise ValueError("Выберите числовую целевую колонку. Для набора классификации класс автоматически не используется.")
        if target not in meta["targets"]:
            raise ValueError("Цель должна быть числовым измерением. Метки классов не подходят для обычной регрессии.")
        if frame[target].isna().any():
            raise ValueError(f"В целевой колонке «{target}» есть пропуски. Исправьте их или выберите другую цель.")
        excluded = set(meta.get("excluded_features", []))
        feature_names = [column for column in frame.columns if column != target and column not in excluded] if features is None else features
        if not isinstance(feature_names, list) or not feature_names or not all(isinstance(column, str) for column in feature_names):
            raise ValueError("Выберите хотя бы один признак.")
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("Один признак выбран несколько раз.")
        if target in feature_names:
            raise ValueError("Целевая колонка не может одновременно быть признаком: это утечка ответа.")
        unknown = set(feature_names) - set(frame.columns)
        if unknown:
            raise ValueError("В данных нет признаков: " + ", ".join(sorted(unknown)))
        for column in feature_names:
            if frame[column].isna().all():
                raise ValueError(f"Признак «{column}» полностью пуст; удалите его.")
            if not self._numeric(frame[column]) and frame[column].nunique(dropna=True) > MAX_CATEGORIES:
                raise ValueError(f"В «{column}» больше {MAX_CATEGORIES} категорий. Удалите идентификатор или сократите категории.")
        X = frame[feature_names].copy()
        for column in feature_names:
            if not self._numeric(X[column]):
                X[column] = X[column].map(lambda value: np.nan if pd.isna(value) else str(value)).astype(object)
        return DataBundle(X=X, y=frame[target].to_numpy(dtype=float), feature_names=list(feature_names), target_name=target, name=meta["name"], meta=meta)

    @staticmethod
    def _numeric(series: pd.Series) -> bool:
        return bool(is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype))

    def _read(self, dataset_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        if not isinstance(dataset_id, str) or not _ID_PATTERN.fullmatch(dataset_id):
            raise ValueError("Некорректный идентификатор набора данных.")
        path = self.root / f"{dataset_id}.json"
        if not path.is_file():
            raise FileNotFoundError("Набор данных не найден. Загрузите его заново.")
        with path.open(encoding="utf-8") as file:
            saved = json.load(file)
        frame = pd.DataFrame(saved["data"], columns=saved["metadata"]["column_names"])
        for item in saved["metadata"]["columns"]:
            if item["numeric"]:
                frame[item["name"]] = pd.to_numeric(frame[item["name"]], errors="coerce")
            elif item["dtype"] == "bool":
                frame[item["name"]] = frame[item["name"]].astype(bool)
            else:
                frame[item["name"]] = frame[item["name"]].map(lambda value: None if pd.isna(value) else str(value)).astype(object)
        return frame, saved["metadata"]

    def _store(self, frame: pd.DataFrame, name: str, default_target: str | None, details: dict[str, Any]) -> dict[str, Any]:
        frame = self._validate_frame(frame)
        excluded_targets = set(details.get("excluded_targets", []))
        targets = [column for column in frame.columns if self._numeric(frame[column]) and column not in excluded_targets]
        if not targets:
            raise ValueError("В таблице нет числовых колонок для целевой переменной регрессии.")
        if default_target is not None and default_target not in targets:
            raise ValueError("Указанная целевая колонка не является числовой.")
        numeric = frame.select_dtypes(include="number")
        if numeric.size and np.isinf(numeric.to_numpy(dtype=float)).any():
            raise ValueError("В таблице есть бесконечные значения. Замените их числом или пропуском.")
        warnings = list(details.get("warnings", []))
        for column in frame.columns:
            if not self._numeric(frame[column]) and frame[column].nunique(dropna=True) > MAX_CATEGORIES:
                warnings.append(f"В «{column}» много категорий; этот столбец нельзя использовать как признак без обработки.")
        dataset_id = uuid4().hex
        records = json.loads(frame.to_json(orient="values", date_format="iso", double_precision=15))
        columns = [{"name": column, "dtype": str(frame[column].dtype), "numeric": self._numeric(frame[column]), "missing": int(frame[column].isna().sum()), "unique": int(frame[column].nunique(dropna=True))} for column in frame.columns]
        metadata = {**details, "id": dataset_id, "name": str(name)[:200], "rows": len(frame), "columns": columns, "column_names": list(frame.columns), "preview": [dict(zip(frame.columns, row)) for row in records[:100]], "targets": targets, "default_target": default_target, "feature_names": [column for column in frame.columns if column != default_target and column not in details.get("excluded_features", [])], "warnings": warnings}
        payload = {"metadata": metadata, "data": records}
        temp = self.root / f"{dataset_id}.tmp"
        try:
            temp.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            temp.replace(self.root / f"{dataset_id}.json")
        finally:
            temp.unlink(missing_ok=True)
        return metadata

    @staticmethod
    def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame) or len(frame) < 3 or len(frame.columns) < 2:
            raise ValueError("Нужны минимум три строки и две колонки: признаки и числовая цель.")
        if len(frame) > MAX_ROWS or len(frame.columns) > MAX_COLUMNS or frame.size > MAX_CELLS:
            raise ValueError("Таблица превышает лимиты: 100 000 строк, 1000 колонок, 5 млн ячеек.")
        frame = frame.copy().reset_index(drop=True)
        names = [str(column).strip() for column in frame.columns]
        if any(not column or len(column) > 200 for column in names) or len(names) != len(set(names)):
            raise ValueError("Названия колонок должны быть непустыми, уникальными и не длиннее 200 символов.")
        frame.columns = names
        for column in names:
            series = frame[column]
            if series.dtype == object or isinstance(series.dtype, pd.CategoricalDtype):
                if series.map(lambda value: isinstance(value, (dict, list, tuple, set))).any():
                    raise ValueError("Ячейки должны содержать число, текст или пропуск, а не вложенные объекты.")
                nonempty = series.dropna()
                if len(nonempty):
                    converted = pd.to_numeric(series, errors="coerce")
                    if int(converted.notna().sum()) == len(nonempty):
                        frame[column] = converted
                    else:
                        comma_numeric = pd.to_numeric(series.astype("string").str.replace(",", ".", regex=False), errors="coerce")
                        if int(comma_numeric.notna().sum()) == len(nonempty):
                            frame[column] = comma_numeric.astype(float)
                        else:
                            frame[column] = series.map(lambda value: None if pd.isna(value) else value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value))
        return frame

    @staticmethod
    def _read_csv(data: bytes, extension: str) -> pd.DataFrame:
        try:
            content = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            content = data.decode("cp1251")
        if "\x00" in content:
            raise ValueError("Файл содержит нулевые байты и не похож на текстовую таблицу.")
        try:
            delimiter = "\t" if extension == ".tsv" else csv.Sniffer().sniff(content[:65536], delimiters=",;\t|").delimiter
        except csv.Error as exc:
            raise ValueError("Не удалось определить разделитель. Нужны колонки, разделенные запятой, точкой с запятой или табуляцией.") from exc
        headers = next(csv.reader(StringIO(content), delimiter=delimiter), [])
        if len(headers) != len(set(header.strip() for header in headers)) or any(not header.strip() for header in headers):
            raise ValueError("В CSV есть пустые или повторяющиеся названия колонок.")
        return pd.read_csv(StringIO(content), sep=delimiter, nrows=MAX_ROWS + 1, on_bad_lines="error")

    @staticmethod
    def _number(params: dict[str, Any], key: str, default: float, minimum: float, maximum: float, integer: bool = False) -> int | float:
        value = params.get(key, default)
        try:
            number = float(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Параметр «{key}» должен быть числом.") from exc
        if not math.isfinite(number) or number < minimum or number > maximum or (integer and number != int(number)):
            raise ValueError(f"Параметр «{key}» должен быть между {minimum} и {maximum}." )
        return int(number) if integer else number

    def _synthetic(self, name: str, params: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
        n = self._number(params, "n_samples", 180, 12, MAX_ROWS, True)
        p = self._number(params, "n_features", 2, 1, 100, True)
        if n * (p + 1) > MAX_CELLS:
            raise ValueError("Слишком много строк и признаков для генератора.")
        seed = self._number(params, "seed", 42, 0, 2**32 - 1, True)
        noise = self._number(params, "noise", 8, 0, 10000)
        rho = self._number(params, "correlation", 0.95 if name == "correlated" else 0.2, -0.99, 0.9999)
        outliers = self._number(params, "outliers", 0, 0, 0.5)
        spread = self._number(params, "scale_spread", 0, 0, 6)
        sparsity = self._number(params, "sparsity", 0.7, 0, 1)
        group_size = self._number(params, "group_size", 3, 1, 100, True)
        intercept = self._number(params, "intercept", 10, -1e6, 1e6)
        if p > 1 and rho <= -1 / (p - 1):
            raise ValueError(f"При {p} признаках одинаковая попарная корреляция должна быть больше {-1 / (p - 1):.3f}.")
        rng = np.random.default_rng(seed)
        covariance = np.full((p, p), rho)
        np.fill_diagonal(covariance, 1)
        raw = rng.standard_normal((n, p)) @ np.linalg.cholesky(covariance).T
        scales = np.logspace(-spread / 2, spread / 2, p) if p > 1 else np.ones(1)
        if "scales" in params:
            try:
                scales = np.asarray(params["scales"], dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError("Масштабы должны быть списком положительных чисел.") from exc
            if scales.shape != (p,) or not np.isfinite(scales).all() or (scales <= 0).any() or (scales > 1e6).any():
                raise ValueError("Укажите по одному положительному масштабу до 1 млн на каждый признак.")
        beta_raw = rng.uniform(15, 55, p) * rng.choice([-1, 1], p)
        if name == "sparse":
            inactive = rng.choice(p, min(p, int(round(p * sparsity))), replace=False)
            beta_raw[inactive] = 0
        if name == "grouped":
            for block, start in enumerate(range(0, p, group_size)):
                if block % 2:
                    beta_raw[start:start + group_size] = 0
        if name == "fused":
            for start in range(0, p, group_size):
                beta_raw[start:start + group_size] = rng.choice([-40, -15, 0, 20, 50])
        X = raw * scales
        beta = beta_raw / scales
        signal = X @ beta
        local_noise = noise * (0.25 + np.abs(raw[:, 0])) if name == "heteroscedastic" else np.full(n, noise)
        errors = rng.normal(size=n) * local_noise
        if name == "nonlinear":
            signal = signal + 35 * (raw[:, 0] ** 2 - 1)
        if name in {"positive", "counts"}:
            normalization = max(float(np.std(signal)), 1.0) * 2
            linear_predictor = signal / normalization + 1
            if name == "positive":
                y = np.exp(np.clip(linear_predictor + rng.normal(0, min(noise / 50, 2), n), -12, 12))
            else:
                y = rng.poisson(np.exp(np.clip(linear_predictor, -8, 8))).astype(float)
            beta = beta / normalization
            intercept = 1.0
        else:
            y = signal + intercept + errors
        outlier_indices = rng.choice(n, int(round(n * outliers)), replace=False)
        if len(outlier_indices):
            magnitude = max(float(np.std(y)), noise, 1) * 6
            if name in {"positive", "counts"}:
                y[outlier_indices] += np.abs(rng.normal(magnitude, magnitude / 3, len(outlier_indices)))
                if name == "counts":
                    y = np.round(y)
            else:
                y[outlier_indices] += rng.choice([-1, 1], len(outlier_indices)) * rng.uniform(magnitude, magnitude * 2, len(outlier_indices))
        names = [f"x{i + 1}" for i in range(p)]
        frame = pd.DataFrame(X, columns=names)
        frame["y"] = y
        metadata = {"source": "synthetic", "generator": name, "task": "regression", "description": _SYNTHETIC[name][1], "params": {**params, "n_samples": n, "n_features": p, "seed": seed}, "true_coefficients": beta.tolist(), "true_intercept": intercept, "true_feature_names": names, "outlier_indices": sorted(outlier_indices.tolist()), "groups": [index // group_size for index in range(p)], "link": "log" if name in {"positive", "counts"} else "identity", "warnings": ["Истинная зависимость нелинейна; обычная прямая не описывает ее целиком."] if name == "nonlinear" else []}
        return frame, metadata

    def _make_sklearn(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        if name not in _MAKE_SUPPORTED:
            raise ValueError("Этот генератор возвращает неподдерживаемую структуру. Выберите табличный генератор.")
        loader = getattr(sklearn_datasets, name)
        signature = inspect.signature(loader)
        accepted = set(signature.parameters)
        n = self._number(params, "n_samples", 180, 12, MAX_ROWS, True)
        p = self._number(params, "n_features", 5 if name == "make_friedman1" else 2, 1, 100, True)
        kwargs: dict[str, Any] = {"random_state": self._number(params, "seed", 42, 0, 2**32 - 1, True)}
        if "n_samples" in accepted:
            kwargs["n_samples"] = n
        if "n_features" in accepted:
            kwargs["n_features"] = p
        if "noise" in accepted:
            kwargs["noise"] = self._number(params, "noise", 8 if name in _MAKE_REGRESSION else 0.05, 0, 1000)
        if name == "make_regression":
            kwargs["n_informative"] = self._number(params, "n_informative", min(p, 5), 1, p, True)
        if name == "make_classification":
            kwargs.update(n_informative=min(p, 2), n_redundant=0, n_clusters_per_class=1)
        if name == "make_low_rank_matrix":
            kwargs.update(n_samples=n, n_features=p, effective_rank=min(p, 3))
        if name == "make_friedman1" and p < 5:
            raise ValueError("Генератор Friedman 1 требует минимум пять признаков.")
        try:
            generated = loader(**kwargs)
        except Exception as exc:
            raise ValueError(f"Не удалось создать набор {name}: {exc}") from exc
        if isinstance(generated, tuple):
            data, target = generated[:2]
        else:
            data, target = generated, None
        frame = pd.DataFrame(data, columns=[f"x{i + 1}" for i in range(data.shape[1])])
        is_regression = name in _MAKE_REGRESSION
        if target is not None:
            frame["y" if is_regression else "class_label"] = target if is_regression else [f"класс: {value}" for value in target]
        meta = {"source": "sklearn", "generator": name, "task": "regression" if is_regression else "classification" if target is not None else "matrix", "description": f"Набор создан sklearn.datasets.{name}.", "params": kwargs, "excluded_features": [] if is_regression else ["class_label"], "excluded_targets": [] if is_regression else ["class_label"], "warnings": [] if is_regression else ["Выберите числовое измерение как цель. Метка класса не является величиной регрессии."]}
        return self._store(frame, name, "y" if is_regression else None, meta)

    def _load_sklearn(self, kind: str, name: str, params: dict[str, Any]) -> dict[str, Any]:
        if kind == "openml":
            loader_name = "fetch_openml"
        else:
            prefix = "fetch_" if kind == "fetch" else "load_"
            loader_name = name if name.startswith(prefix) else prefix + name
        info = _KNOWN.get(loader_name)
        if not info or not info[1]:
            raise ValueError(info[3] if info else "В установленном sklearn нет поддержанного табличного загрузчика с таким именем.")
        loader = getattr(sklearn_datasets, loader_name)
        accepted = set(inspect.signature(loader).parameters)
        kwargs: dict[str, Any] = {"as_frame": True} if "as_frame" in accepted else {}
        if loader_name.startswith("fetch_"):
            kwargs.update(data_home=str(self.root / "download-cache"), download_if_missing=True)
        if loader_name == "load_diabetes":
            kwargs["scaled"] = bool(params.get("scaled", False))
        if loader_name == "load_digits" and "n_class" in params:
            kwargs["n_class"] = self._number(params, "n_class", 10, 2, 10, True)
        if loader_name == "fetch_kddcup99":
            kwargs["percent10"] = True
            kwargs["shuffle"] = True
            kwargs["random_state"] = self._number(params, "seed", 42, 0, 2**32 - 1, True)
        if loader_name == "fetch_openml":
            kwargs.pop("download_if_missing", None)
            data_id = params.get("data_id", params.get("id"))
            openml_name = params.get("name") or (name if name not in {"openml", "fetch_openml", "linear"} else None)
            if data_id is not None:
                kwargs["data_id"] = self._number({"data_id": data_id}, "data_id", 1, 1, 100000000, True)
            elif openml_name and re.fullmatch(r"[\w .+\-]{1,200}", str(openml_name)):
                kwargs["name"] = str(openml_name)
                version = params.get("version", "active")
                kwargs["version"] = version if version == "active" else self._number({"version": version}, "version", 1, 1, 100000, True)
            else:
                raise ValueError("Для OpenML укажите корректное имя или числовой data_id.")
            kwargs.update(n_retries=1, delay=0.1, parser="auto")
            if "target_column" in params:
                column = params["target_column"]
                if column is not None and not isinstance(column, (str, list)):
                    raise ValueError("Цель OpenML должна быть именем колонки или списком имен.")
                kwargs["target_column"] = column
        kwargs = {key: value for key, value in kwargs.items() if key in accepted}
        try:
            bunch = loader(**kwargs)
        except Exception as exc:
            if loader_name.startswith("fetch_"):
                raise ValueError(f"Не удалось загрузить {loader_name}: сеть недоступна, набор отсутствует или сервер отказал. Можно загрузить CSV вручную. Причина: {exc}") from exc
            raise ValueError(f"Не удалось открыть {loader_name}: {exc}") from exc
        task = info[0]
        if loader_name == "fetch_openml":
            openml_target = bunch.get("target")
            if isinstance(openml_target, pd.Series):
                task = "classification" if not self._numeric(openml_target) else "regression"
            elif isinstance(openml_target, pd.DataFrame):
                task = "regression" if all(self._numeric(openml_target[column]) for column in openml_target) else "classification"
            else:
                task = "unknown"
        return self._adapt_bunch(bunch, loader_name, task, params, info[2], info[3])

    def _adapt_bunch(self, bunch: Any, loader_name: str, task: str, params: dict[str, Any], label: str, description: str) -> dict[str, Any]:
        data = bunch.get("data")
        if sparse.issparse(data):
            raise ValueError("Этот набор хранится как разреженная матрица. Выберите табличный набор или импортируйте подготовленный CSV.")
        if not isinstance(data, pd.DataFrame):
            data = np.asarray(data)
            if data.ndim != 2:
                raise ValueError("Набор содержит изображения, текст или вложенные массивы вместо таблицы.")
            feature_names = bunch.get("feature_names", [f"x{i + 1}" for i in range(data.shape[1])])
            frame = pd.DataFrame(data, columns=[str(column) for column in feature_names])
        else:
            frame = data.copy()
            frame.columns = [str(column) for column in frame.columns]
        target = bunch.get("target")
        regression = task == "regression"
        target_columns: list[str] = []
        if target is not None:
            if isinstance(target, pd.DataFrame):
                target_frame = target.reset_index(drop=True).copy()
            elif isinstance(target, pd.Series):
                target_frame = target.reset_index(drop=True).to_frame(name=target.name or "target")
            else:
                values = np.asarray(target)
                names = bunch.get("target_names") if regression and values.ndim == 2 else None
                target_frame = pd.DataFrame(values, columns=names if names is not None else ["target"] if values.ndim == 1 else [f"target_{i + 1}" for i in range(values.shape[1])])
            for original in target_frame.columns:
                column = str(original) if regression else "class_label" if len(target_frame.columns) == 1 else f"class_{original}"
                while column in frame.columns:
                    column += "_target"
                series = target_frame[original]
                frame[column] = series.to_numpy() if regression else series.map(lambda value: None if pd.isna(value) else "класс: " + (value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value))).to_numpy()
                target_columns.append(column)
        warnings: list[str] = []
        sample_limit = self._number(params, "sample_limit", min(MAX_ROWS, 10000) if loader_name in {"fetch_covtype", "fetch_kddcup99", "fetch_openml"} else MAX_ROWS, 12, MAX_ROWS, True)
        original_rows = len(frame)
        if original_rows > sample_limit:
            seed = self._number(params, "seed", 42, 0, 2**32 - 1, True)
            indices = np.sort(np.random.default_rng(seed).choice(original_rows, sample_limit, replace=False))
            frame = frame.iloc[indices].reset_index(drop=True)
            warnings.append(f"Из {original_rows} строк взята воспроизводимая случайная выборка {len(frame)} строк.")
        if not regression:
            warnings.append("Это набор классификации или другого назначения. Выберите числовую колонку измерений; метка класса не выбирается автоматически.")
        meta = {"source": "sklearn", "loader": loader_name, "task": task, "description": description, "source_description": str(bunch.get("DESCR", ""))[:20000], "excluded_targets": [] if regression else target_columns, "excluded_features": target_columns, "original_rows": original_rows, "warnings": warnings, "source_url": f"https://scikit-learn.org/stable/modules/generated/sklearn.datasets.{loader_name}.html"}
        if loader_name == "fetch_openml":
            meta["openml"] = {key: value for key, value in (bunch.get("details") or {}).items() if key in {"id", "name", "version", "description", "url"}}
            label = "OpenML: " + str((bunch.get("details") or {}).get("name", "набор"))
        return self._store(frame, label, target_columns[0] if regression and target_columns else None, meta)
