"""Capture public JSON from the real v2 services for DOM boundary tests."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient
from linear_lab.app import create_app, public_job
from linear_lab.models import ModelRegistry
from linear_lab.training import TrainingService
from linear_lab.search import SearchService
from linear_lab.exports import ModelExportService

with tempfile.TemporaryDirectory() as temporary:
    directory = Path(temporary)
    app = create_app(directory)
    with TestClient(app) as client:
        catalogue = client.get('/api/catalogue').json()
        data = app.state.datasets
        dataset = data.load({'kind':'synthetic','name':'linear','params':{'n_samples':180,'n_features':2,'noise':8,'seed':42}})
        classification = data.load({'kind':'builtin','name':'load_iris'})
        request = {'dataset_id':dataset['id'],'target':'y','features':['x1','x2'],'model':'ridge','params':{'alpha':1.0},'seed':42,
            'split':{'train':0.6,'validation':0.2,'test':0.2,'shuffle':True},'preprocessing':{'scale':True,'impute':True,'degree':1},
            'metrics':['mse','rmse','mae','r2'],'custom_metric':'','metric_params':{'quantile':.5,'power':1.5},
            'epochs':100,'cv':0,'regularization_path':True,'learning_curve':False,'permutation_importance':False}
        job_id = 'd'*32
        artifact = directory/'jobs'/job_id/'model.joblib'
        artifact.parent.mkdir(parents=True)
        result = TrainingService(data,ModelRegistry()).run(dict(request,artifact_path=str(artifact)))
        artifact.with_name('request.json').write_text(json.dumps(request))
        artifact.with_name('result.json').write_text(json.dumps(result))
        fixture = {'catalogue':catalogue,'dataset':dataset,'job':public_job({'id':job_id,'status':'completed','result':result}),
            'rows':data.rows(dataset['id'],0,100),'allRows':data.rows(dataset['id'],0,200),
            'library':data.list_library(),'classification':classification,
            'exportCapabilities':{'formats':ModelExportService(directory).capabilities(job_id)},'request':request}
        fixture['explorations'] = {}
        for meta in (dataset,classification):
            target = meta.get('task_target') or meta['default_target']
            numeric = [col['name'] for col in meta['columns'] if col['numeric'] and col['name'] != target]
            x,second = numeric[:2]
            z = target if target in numeric or any(c['name']==target and c['numeric'] for c in meta['columns']) else numeric[2]
            for y in (target,second):
                payload = data.explore(meta['id'],x=x,y=y,z=z,color=target,target=target,sample_size=2000)
                fixture['explorations'][meta['id']+':'+y]=payload
        response = client.post(f"/api/datasets/{dataset['id']}/preprocessing-preview",json={**request,'preprocessing':{**request['preprocessing'],'scaler':'robust','imputation':'median','missing_indicator':True,'interaction_only':True,'degree':2,'numeric_transform':'log1p','clip_quantiles':[.01,.99],'selection':'mutual_info','max_features':3},'resampling':{'method':'random_over','focus':'high'}})
        response.raise_for_status()
        fixture['preprocessingPreview'] = response.json()
        fixture['searchJobs'] = {}
        for direction in ['min','max']:
            search_request = {**request,'custom_metric':'mean(abs(error))','cv_config':{'strategy':'kfold','folds':3},
                'search':{'method':'optuna_tpe','metric':'custom','direction':direction,'trials':3,'param_space':{'alpha':{'type':'float','low':.1,'high':10,'log':True}}}}
            searched = SearchService(data,ModelRegistry()).run(search_request)
            fixture['searchJobs'][direction] = public_job({'id':direction,'status':'completed','result':searched})
        fixture['cvJob'] = public_job({'id':'cv','status':'completed','result':TrainingService(data,ModelRegistry()).run({**request,'cv_config':{'strategy':'repeated_kfold','folds':3,'repeats':2},'n_jobs':2})})
        (ROOT/'tests'/'fixtures'/'qa-fixture.json').write_text(json.dumps(fixture,ensure_ascii=False),encoding='utf-8')
        print('Captured',len(catalogue['lessons']),'lessons,',len(catalogue['models']),'models and real CV/Optuna/preprocessing/dataset results')
