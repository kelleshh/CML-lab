"""Translate declarative pipeline validation and portable exports."""
import json
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, Response
from cml_lab.infrastructure.ml.pipelines import (
    compile_pipeline, describe_pipeline, inspect_pipeline_source, pipeline_catalogue,
    transformer_tree, emit_pipeline_source, validate_pipeline_object,
)
from cml_lab.infrastructure.ml.search_presets import search_preset
from cml_lab.infrastructure.ml.hyperparameters import json_value


def _export(config, format):
    if format not in {'py', 'json'}:
        raise ValueError('Формат пайплайна: py или json.')
    declarative = config.get('declarative_pipeline')
    if declarative:
        overrides = config.get('pipeline_params') or {}
        if overrides:
            pipeline = compile_pipeline(declarative)
            pipeline.set_params(**overrides)
            validate_pipeline_object(pipeline)
            declarative = {'source': emit_pipeline_source(transformer_tree(pipeline))}
        document = describe_pipeline(declarative)
        value = document['source'] if format == 'py' else json.dumps(document['spec'], ensure_ascii=False, indent=2)
    else:
        document = {'format': 'cml.preparation', 'version': 2, 'preprocessing': config}
        text = json.dumps(config, ensure_ascii=False)
        value = json.dumps(document, ensure_ascii=False, indent=2) if format == 'json' else (
            '"""CML-lab legacy preparation. Supply the raw feature DataFrame X."""\n'
            'import json\nfrom cml_lab.infrastructure.ml.preparation import build_preprocessor\n\n'
            f'CONFIG = json.loads({text!r})\n\n'
            'def create_pipeline(X, task="regression", seed=42):\n'
            '    return build_preprocessor(X, CONFIG, task=task, seed=seed)\n'
        )
    return Response(value, media_type='text/x-python' if format == 'py' else 'application/json',
        headers={'Content-Disposition': f'attachment; filename="pipeline.{format}"'})


def pipeline_routes(services):
    router = APIRouter(prefix='/api/cml', tags=['Pipelines', 'Tuning'])

    @router.get('/pipelines/catalogue')
    def catalogue(): return pipeline_catalogue()

    @router.post('/pipelines/validate')
    def validate(payload: dict):
        try: return describe_pipeline(payload.get('spec', payload))
        except ValueError as error:
            detail = error.to_dict() if hasattr(error, 'to_dict') else {'message': str(error)}
            return JSONResponse({'detail': str(error), 'error': detail}, status_code=422)

    @router.get('/pipelines/source')
    def source(class_name: str = Query(alias='class')): return inspect_pipeline_source(class_name)

    @router.get('/pipelines/sources/{class_name}')
    def sources(class_name: str): return inspect_pipeline_source(class_name)

    @router.post('/pipelines/export')
    def export(payload: dict, format: str = 'py'):
        return _export({'declarative_pipeline': payload.get('spec', payload)}, format)

    @router.get('/preprocessor-recipes/{recipe_id}/export')
    def recipe_export(recipe_id: str, format: str = 'json'):
        recipe = services.recipes.get('preprocessor', recipe_id)
        return _export(recipe['config'], format)

    @router.post('/search/preset')
    def preset(payload: dict): return search_preset(services.policy.catalogue, payload)

    @router.post('/pipelines/parameters')
    def parameters(payload: dict):
        model = compile_pipeline(payload.get('spec', payload))
        classes = {item['name']: item for item in pipeline_catalogue()['classes']}
        params = model.get_params(deep=True)
        rows = []
        for key, value in params.items():
            if hasattr(value, 'fit') or key.endswith(('steps', 'transformers', 'transformer_list')): continue
            parent_key, _, name = key.rpartition('__')
            parent = params.get(parent_key, model)
            definition = next((item for item in classes.get(type(parent).__name__, {}).get('params', []) if item['key'] == (name or key)), {})
            if value is None or isinstance(value, (str, bool, int, float, list, dict, tuple)):
                rows.append({**definition, 'key': 'pipeline__' + key, 'label': 'pipeline__' + key,
                             'default': json_value(value), 'type': definition.get('type', 'json'),
                             'help': definition.get('help') or definition.get('description') or 'Параметр вложенного преобразования. Меняется внутри каждой части CV; статистики обучаются только на train.'})
        return {'params': rows}

    return router
