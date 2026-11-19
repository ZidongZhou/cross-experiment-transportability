#!/usr/bin/env python3
"""Reproducible analysis pipeline for the ASSISTments transportability study.

Input:
  experiment_dataset_2021-09-23.zip (outer ZIP containing 88 experiment ZIPs)
Output:
  machine-readable aggregate result CSVs used by figures/manuscript.

All outcome construction, preprocessing and model fitting are performed here.
"""
from __future__ import annotations
import io, json, zipfile, hashlib, os, math, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, StratifiedShuffleSplit, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier
import statsmodels.api as sm
from scipy.special import expit, logit

SEED=20260807
N_JOBS=int(os.environ.get('TKL_N_JOBS','4'))
ROOT=Path(__file__).resolve().parents[1]
RAW=Path(os.environ.get('TKL_RAW_ZIP', ROOT/'data'/'experiment_dataset_2021-09-23.zip'))
EXPECTED_RAW_SHA256='06d8c992c9600781dc1c7c80690e29ec46588df260aea9a44ef16b7d083266ea'
OUT=ROOT/'results'; WORK=ROOT/'analysis_work'; OUT.mkdir(parents=True,exist_ok=True); WORK.mkdir(parents=True,exist_ok=True)

def sha256_file(path: Path, chunk_size=1024*1024):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()

def verify_raw_archive(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'Raw archive not found: {path}')
    actual=sha256_file(path)
    if actual != EXPECTED_RAW_SHA256:
        raise ValueError(f'Raw archive SHA256 mismatch. Expected {EXPECTED_RAW_SHA256}, got {actual}.')


# ---------- raw-data reconstruction ----------
def load_raw(raw_zip: Path):
    alogs=[]; priors=[]; slogs=[]; expstats=[]
    with zipfile.ZipFile(raw_zip) as outer:
        names=[n for n in outer.namelist() if n.endswith('.zip')]
        for n in names:
            with zipfile.ZipFile(io.BytesIO(outer.read(n))) as z:
                a=pd.read_csv(z.open('exp_alogs.csv'),low_memory=False)
                p=pd.read_csv(z.open('priors.csv'),low_memory=False)
                s=pd.read_csv(z.open('exp_slogs.csv'),usecols=['experiment_id','student_id','action','timestamp'],low_memory=False)
                ts=pd.to_datetime(s['timestamp'],format='mixed',utc=True,errors='coerce')
                horizon=ts.max()
                mask=s.action.isin(['assignment_started','assignment_finished'])
                q=s.loc[mask,['experiment_id','student_id','action']].copy(); q['ts']=ts[mask].to_numpy()
                st=q[q.action=='assignment_started'].groupby(['experiment_id','student_id'],as_index=False)['ts'].agg(slog_start='min',slog_start_count='count')
                fi=q[q.action=='assignment_finished'].groupby(['experiment_id','student_id'],as_index=False)['ts'].agg(slog_finish='min',slog_finish_count='count')
                g=st.merge(fi,on=['experiment_id','student_id'],how='outer'); g['observed_horizon']=horizon
                alogs.append(a); priors.append(p); slogs.append(g)
                expstats.append({'experiment_id':n[:-4],'n_alogs':len(a),'n_priors':len(p),'n_slogs':len(s),'observed_horizon':horizon})
    return pd.concat(alogs,ignore_index=True), pd.concat(priors,ignore_index=True), pd.concat(slogs,ignore_index=True), pd.DataFrame(expstats)

def derive_dataset(alogs, priors, slog):
    idcol='assistments_reference_assignment_log_id'
    # Duplicated assignment IDs in the source release must differ only in assigned condition.
    dup=alogs[alogs.duplicated(idcol,keep=False)]
    for col in [c for c in alogs.columns if c not in (idcol,'assigned_condition')]:
        if not dup.empty:
            assert dup.groupby(idcol)[col].nunique(dropna=False).max() == 1, f'Duplicated IDs disagree on {col}'
    a=alogs.drop_duplicates(idcol,keep='first').copy()
    a['start']=pd.to_datetime(a.start_time,format='mixed',utc=True,errors='coerce')
    a['alog_end']=pd.to_datetime(a.end_time,format='mixed',utc=True,errors='coerce')
    for c in ['slog_start','slog_finish','observed_horizon']:
        slog[c]=pd.to_datetime(slog[c],utc=True,errors='coerce')
    d=a.merge(slog,on=['experiment_id','student_id'],how='left',validate='one_to_one')
    # Strong source consistency check: assignment log and action log timestamps coincide where observed.
    d['start_source_match_seconds']=(d.start-d.slog_start).dt.total_seconds()
    d['finish_source_match_seconds']=(d.alog_end-d.slog_finish).dt.total_seconds()
    assert d.start.notna().all(), 'Unusable assignment start timestamp'
    assert d.slog_start.notna().all(), 'Missing matching assignment_started action'
    assert (d.start_source_match_seconds == 0).all(), 'Assignment start timestamps disagree across sources'
    observed_finish=d.alog_end.notna()
    assert d.loc[observed_finish,'slog_finish'].notna().all(), 'Missing matching assignment_finished action'
    assert (d.loc[observed_finish,'finish_source_match_seconds'] == 0).all(), 'Observed finish timestamps disagree across sources'
    # Fixed-horizon endpoints. An event with no finish is evaluable only if 14 observed days remain in that experiment's action log.
    for days in (7,14,30):
        mature=d.start+pd.Timedelta(days=days)
        evaluable=d.slog_finish.notna() | (mature<=d.observed_horizon)
        d[f'evaluable{days}']=evaluable
        d[f'y{days}']=np.where(evaluable, (d.slog_finish.isna() | (d.slog_finish>mature)).astype(int), np.nan)
    d['y_archive']=d.slog_finish.isna().astype(int)
    d=d.merge(priors,on=['experiment_id','student_id'],how='left',validate='one_to_one')
    # Derived pre-assignment features.
    d['student_skill_builder_completion_rate']=np.where(d.student_prior_started_skill_builder_count>0,d.student_prior_completed_skill_builder_count/d.student_prior_started_skill_builder_count,np.nan)
    d['student_problem_set_completion_rate']=np.where(d.student_prior_started_problem_set_count>0,d.student_prior_completed_problem_set_count/d.student_prior_started_problem_set_count,np.nan)
    d['no_student_prior_skill_builders']=(d.student_prior_started_skill_builder_count==0).astype(int)
    d['no_student_prior_problem_sets']=(d.student_prior_started_problem_set_count==0).astype(int)
    d['no_student_prior_completed_problems']=(d.student_prior_completed_problem_count==0).astype(int)
    d['class_prior_started_skill_builder_count']=d.class_prior_skill_builder_count*d.class_prior_skill_builder_percent_started/100.0
    d['class_prior_started_problem_set_count']=d.class_prior_problem_set_count*d.class_prior_problem_set_percent_started/100.0
    d['no_class_prior_skill_builders']=(d.class_prior_started_skill_builder_count==0).astype(int)
    d['no_class_prior_problem_sets']=(d.class_prior_started_problem_set_count==0).astype(int)
    d['no_class_prior_completed_problems']=(d.class_prior_completed_problem_count==0).astype(int)
    cc=pd.to_datetime(d.class_creation_date,format='mixed',utc=True,errors='coerce')
    tc=pd.to_datetime(d.teacher_account_creation_date,format='mixed',utc=True,errors='coerce')
    d['class_age_days']=(d.start-cc).dt.total_seconds()/86400.0
    d['teacher_account_age_days']=(d.start-tc).dt.total_seconds()/86400.0
    return d

STUDENT_ACTIVITY=[
'student_prior_started_skill_builder_count','student_prior_completed_skill_builder_count',
'student_prior_started_problem_set_count','student_prior_completed_problem_set_count',
'student_prior_completed_problem_count','student_skill_builder_completion_rate','student_problem_set_completion_rate',
'no_student_prior_skill_builders','no_student_prior_problem_sets','no_student_prior_completed_problems']
STUDENT_PERF=['student_prior_median_first_response_time','student_prior_median_time_on_task','student_prior_average_correctness','student_prior_average_attempt_count']
CLASS_ACTIVITY=['class_student_count','class_prior_skill_builder_count','class_prior_problem_set_count','class_prior_started_skill_builder_count','class_prior_started_problem_set_count','class_prior_skill_builder_percent_started','class_prior_skill_builder_percent_completed','class_prior_problem_set_percent_started','class_prior_problem_set_percent_completed','class_prior_completed_problem_count','no_class_prior_skill_builders','no_class_prior_problem_sets','no_class_prior_completed_problems']
CLASS_PERF=['class_prior_median_time_on_task','class_prior_median_first_response_time','class_prior_average_correctness','class_prior_average_attempt_count']
CONTEXT_NUM=['class_age_days','teacher_account_age_days']
CONTEXT_CAT=['location','opportunity_zone']
FEATURE_SETS={
    'Activity only': STUDENT_ACTIVITY,
    'Complete learner history': STUDENT_ACTIVITY+STUDENT_PERF,
    'Learner + class history': STUDENT_ACTIVITY+STUDENT_PERF+CLASS_ACTIVITY+CLASS_PERF,
    'Full transportable': STUDENT_ACTIVITY+STUDENT_PERF+CLASS_ACTIVITY+CLASS_PERF+CONTEXT_NUM+CONTEXT_CAT,
    'Context only': CONTEXT_NUM+CONTEXT_CAT,
    'No context': STUDENT_ACTIVITY+STUDENT_PERF+CLASS_ACTIVITY+CLASS_PERF,
}

# ---------- modelling ----------
def split_feature_types(cols):
    cats=[c for c in cols if c in CONTEXT_CAT]
    nums=[c for c in cols if c not in cats]
    return nums,cats

def make_preprocessor(cols):
    nums,cats=split_feature_types(cols); trs=[]
    if nums:
        trs.append(('num',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),nums))
    if cats:
        trs.append(('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',min_frequency=20))]),cats))
    return ColumnTransformer(trs,sparse_threshold=0.3)

def make_model(name, random_state=SEED):
    if name=='Logistic regression':
        # Original comparator retained for exact reproduction of the submitted analysis.
        return SGDClassifier(loss='log_loss',penalty='l2',alpha=1e-4,max_iter=1500,tol=1e-3,average=True,random_state=random_state)
    if name=='Logistic regression (LBFGS)':
        # Fixed L2-regularised comparator; no tuning uses held-out experiments.
        return LogisticRegression(C=1.0, solver='lbfgs', max_iter=5000, tol=1e-4)
    if name in ('LightGBM', 'LightGBM seed 20260808'):
        seed = random_state if name == 'LightGBM' else SEED + 1
        return LGBMClassifier(objective='binary',n_estimators=140,learning_rate=0.05,num_leaves=15,max_depth=5,min_child_samples=50,reg_lambda=1.0,subsample=1.0,subsample_freq=0,colsample_bytree=0.9,random_state=seed,n_jobs=1,verbosity=-1)
    if name=='ExtraTrees':
        return ExtraTreesClassifier(n_estimators=80,min_samples_leaf=10,max_features='sqrt',class_weight=None,random_state=random_state,n_jobs=1)
    raise KeyError(name)

def fit_predict(train_idx,test_idx,d,cols,model_names=('Logistic regression','LightGBM'),model_seeds=None):
    train_idx=np.asarray(train_idx,dtype=int); test_idx=np.asarray(test_idx,dtype=int)
    assert len(np.intersect1d(train_idx,test_idx))==0, 'Training and test indices overlap'
    assert len(train_idx)>0 and len(test_idx)>0, 'Empty training or test split'
    pre=make_preprocessor(cols)
    Xtr=pre.fit_transform(d.iloc[train_idx][cols]); Xte=pre.transform(d.iloc[test_idx][cols]); ytr=d.iloc[train_idx].y.to_numpy()
    out={}
    model_seeds = model_seeds or {}
    for name in model_names:
        m=make_model(name,random_state=model_seeds.get(name,SEED)); m.fit(Xtr,ytr); pred=m.predict_proba(Xte)[:,1]
        assert np.isfinite(pred).all(), f'Non-finite predictions from {name}'
        out[name]=pred
    return test_idx,out

def assign_connected_components(d,entity):
    parent={}; size={}
    def find(x):
        if x not in parent: parent[x]=x; size[x]=1
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra==rb:return
        if size[ra]<size[rb]:ra,rb=rb,ra
        parent[rb]=ra; size[ra]+=size[rb]
    for s,e in zip(d.student_id.to_numpy(),d[entity].to_numpy()): union(f's:{s}',f'{entity}:{e}')
    roots=[find(f's:{s}') for s in d.student_id.to_numpy()]
    mapping={r:i for i,r in enumerate(pd.unique(roots))}
    return np.array([mapping[r] for r in roots])

def run_kfold(d,validation,cols,model_names=('Logistic regression','LightGBM'),model_seeds=None):
    y=d.y.to_numpy(); n=len(d); idx=np.arange(n)
    if validation=='Row random':
        splitter=StratifiedKFold(5,shuffle=True,random_state=SEED); splits=list(splitter.split(idx,y))
    elif validation=='Learner grouped':
        splitter=StratifiedGroupKFold(5,shuffle=True,random_state=SEED); splits=list(splitter.split(idx,y,groups=d.student_id.to_numpy()))
    elif validation=='Learner + class grouped':
        groups=assign_connected_components(d,'class_id'); splitter=StratifiedGroupKFold(5,shuffle=True,random_state=SEED); splits=list(splitter.split(idx,y,groups=groups))
    elif validation=='Learner + teacher grouped':
        groups=assign_connected_components(d,'teacher_id'); splitter=StratifiedGroupKFold(5,shuffle=True,random_state=SEED); splits=list(splitter.split(idx,y,groups=groups))
    else: raise KeyError(validation)
    res=[fit_predict(tr,te,d,cols,model_names,model_seeds) for tr,te in splits] if N_JOBS==1 else Parallel(n_jobs=N_JOBS,backend='threading')(delayed(fit_predict)(tr,te,d,cols,model_names,model_seeds) for tr,te in splits)
    preds={m:np.full(n,np.nan) for m in model_names}
    for te,out in res:
        for m,p in out.items(): preds[m][te]=p
    for m,p in preds.items(): assert np.isfinite(p).all(), f'Missing predictions in {validation}: {m}'
    return preds

def loeo_splits(d,mode):
    exp=d.experiment_id.to_numpy(); stud=d.student_id.to_numpy(); y=d.y.to_numpy(); allidx=np.arange(len(d)); exps=pd.unique(exp)
    splits=[]; purge_stats=[]
    for k,e in enumerate(exps):
        te=allidx[exp==e]; base=allidx[exp!=e]
        test_students=np.unique(stud[te])
        purged=base[~np.isin(stud[base],test_students)]
        if mode=='Standard LOEO': tr=base
        elif mode=='Learner-purged LOEO':
            tr=purged
            assert len(np.intersect1d(np.unique(stud[tr]),test_students))==0, 'Learner purge failed'
        elif mode=='Training-size-matched LOEO':
            if len(purged)>=len(base): tr=base
            else:
                ss=StratifiedShuffleSplit(n_splits=1,train_size=len(purged),random_state=SEED+k)
                sel,_=next(ss.split(base,y[base])); tr=base[sel]
        else: raise KeyError(mode)
        splits.append((tr,te))
        purge_stats.append({'experiment_id':e,'standard_train_n':len(base),'purged_train_n':len(purged),'purged_fraction':1-len(purged)/len(base)})
    return splits,pd.DataFrame(purge_stats)

def run_loeo(d,mode,cols,model_names=('Logistic regression','LightGBM'),model_seeds=None):
    splits,pstats=loeo_splits(d,mode)
    res=[fit_predict(tr,te,d,cols,model_names,model_seeds) for tr,te in splits] if N_JOBS==1 else Parallel(n_jobs=N_JOBS,backend='threading')(delayed(fit_predict)(tr,te,d,cols,model_names,model_seeds) for tr,te in splits)
    preds={m:np.full(len(d),np.nan) for m in model_names}
    for te,out in res:
        for m,p in out.items(): preds[m][te]=p
    for m,p in preds.items(): assert np.isfinite(p).all(), f'Missing predictions in {mode}: {m}'
    return preds,pstats

def run_experiment_grouped_purged(d,cols,model_names=('Logistic regression','LightGBM'),model_seeds=None):
    """Five-fold experiment-blocked CV with learner purge across each held-out experiment block."""
    idx=np.arange(len(d)); groups=d.experiment_id.to_numpy(); splits=[]
    for tr0,te in GroupKFold(5).split(idx,d.y.to_numpy(),groups=groups):
        test_students=np.unique(d.student_id.to_numpy()[te])
        tr=tr0[~np.isin(d.student_id.to_numpy()[tr0],test_students)]
        assert len(np.intersect1d(np.unique(d.student_id.to_numpy()[tr]),test_students))==0, 'Learner purge failed in experiment-blocked CV'
        splits.append((tr,te))
    if N_JOBS==1:
        res=[fit_predict(tr,te,d,cols,model_names,model_seeds) for tr,te in splits]
    else:
        res=Parallel(n_jobs=N_JOBS,backend='threading')(delayed(fit_predict)(tr,te,d,cols,model_names,model_seeds) for tr,te in splits)
    preds={m:np.full(len(d),np.nan) for m in model_names}
    for te,out in res:
        for m,p in out.items(): preds[m][te]=p
    for m,p in preds.items(): assert np.isfinite(p).all(), f'Missing predictions in experiment-blocked learner-purged CV: {m}'
    return preds


def metrics(y,p):
    return {'auroc':roc_auc_score(y,p),'auprc':average_precision_score(y,p),'brier':brier_score_loss(y,p),'log_loss':log_loss(y,p)}

def per_experiment_metrics(d,preds,validation):
    rows=[]
    for e,g in d.groupby('experiment_id'):
        ii=g.index.to_numpy(); yy=g.y.to_numpy(); n=len(g); n1=int(yy.sum()); n0=n-n1
        eligible=n>=50 and min(n1,n0)>=10
        for model,p in preds.items():
            q=p[ii]
            row={'experiment_id':e,'validation':validation,'model':model,'n':n,'events':n1,'prevalence':yy.mean(),'eligible_auroc':eligible}
            if eligible:
                row.update({'auroc':roc_auc_score(yy,q),'auprc':average_precision_score(yy,q)})
            else: row.update({'auroc':np.nan,'auprc':np.nan})
            rows.append(row)
    return pd.DataFrame(rows)

def calibration_coefficients(y,p):
    p=np.clip(np.asarray(p),1e-6,1-1e-6); z=logit(p); y=np.asarray(y)
    try:
        mod=sm.GLM(y,sm.add_constant(z),family=sm.families.Binomial()).fit(disp=0)
        return float(mod.params[0]),float(mod.params[1])
    except Exception as exc:
        warnings.warn(f'Calibration model failed: {exc}', RuntimeWarning)
        return np.nan,np.nan

def intercept_update(y,p):
    p=np.clip(np.asarray(p),1e-6,1-1e-6); z=logit(p); y=np.asarray(y)
    try:
        mod=sm.GLM(y,np.ones((len(y),1)),family=sm.families.Binomial(),offset=z).fit(disp=0)
        return float(mod.params[0])
    except Exception as exc:
        warnings.warn(f'Calibration intercept update failed: {exc}', RuntimeWarning)
        return np.nan

def local_recalibration_cohorts(d):
    """Return eligible chronological calibration/evaluation cohorts and in-memory record keys."""
    key_cols = ['experiment_id', 'student_id', 'class_id', 'start']
    cohorts = {}
    for experiment_id, group in d.groupby('experiment_id', sort=False):
        group = group.sort_values('start', kind='mergesort')
        n = len(group)
        k = max(1, int(math.ceil(0.20 * n)))
        calibration = group.iloc[:k]
        update_time = (calibration.start + pd.Timedelta(days=14)).max()
        evaluation = group[group.start > update_time]
        if (len(calibration) < 30 or len(evaluation) < 30
                or calibration.y.nunique() < 2 or evaluation.y.nunique() < 2
                or calibration.y.value_counts().min() < 5
                or evaluation.y.value_counts().min() < 5):
            continue
        calibration_keys = tuple(sorted(map(tuple, calibration[key_cols].astype(str).to_numpy())))
        evaluation_keys = tuple(sorted(map(tuple, evaluation[key_cols].astype(str).to_numpy())))
        assert len(calibration_keys) == len(set(calibration_keys)), 'Duplicate calibration record keys'
        assert len(evaluation_keys) == len(set(evaluation_keys)), 'Duplicate evaluation record keys'
        assert set(calibration_keys).isdisjoint(evaluation_keys), 'Calibration and evaluation records overlap'
        cohorts[experiment_id] = {
            'calibration': calibration, 'evaluation': evaluation, 'update_time': update_time,
            'calibration_keys': calibration_keys, 'evaluation_keys': evaluation_keys,
        }
    return cohorts


def local_recalibration(d, preds, return_membership=False):
    rows = []
    cohorts = local_recalibration_cohorts(d)
    membership = {}
    for experiment_id, cohort in cohorts.items():
        early = cohort['calibration']
        later = cohort['evaluation']
        update_time = cohort['update_time']
        membership[experiment_id] = (cohort['calibration_keys'], cohort['evaluation_keys'])
        for model, predictions in preds.items():
            pe = predictions[early.index.to_numpy()]
            pl = predictions[later.index.to_numpy()]
            y_later = later.y.to_numpy()
            y_early = early.y.to_numpy()
            adjustment = intercept_update(y_early, pe)
            if not np.isfinite(adjustment):
                continue
            updated = expit(logit(np.clip(pl, 1e-6, 1 - 1e-6)) + adjustment)
            intercept, slope = calibration_coefficients(y_early, pe)
            updated_full = (expit(intercept + slope * logit(np.clip(pl, 1e-6, 1 - 1e-6)))
                            if np.isfinite(intercept + slope) else np.full_like(pl, np.nan))
            common = {
                'experiment_id': experiment_id, 'model': model,
                'n_calibration': len(early), 'n_evaluation': len(later),
                'update_time': update_time,
            }
            rows.extend([
                {**common, 'probability_version': 'No local update',
                 'brier': brier_score_loss(y_later, pl), 'log_loss': log_loss(y_later, pl)},
                {**common, 'probability_version': 'Intercept update',
                 'brier': brier_score_loss(y_later, updated), 'log_loss': log_loss(y_later, updated)},
                {**common, 'probability_version': 'Intercept + slope update',
                 'brier': brier_score_loss(y_later, updated_full), 'log_loss': log_loss(y_later, updated_full)},
            ])
    result = pd.DataFrame(rows)
    return (result, membership) if return_membership else result

def capacity_group(d,preds,group_cols,label):
    eligible=[]; sel={m:[] for m in preds}; allidx=[]
    groups=d.groupby(group_cols,sort=False)
    for key,g in groups:
        if len(g)<20: continue
        ii=g.index.to_numpy(); allidx.extend(ii.tolist()); eligible.append((key,len(g)))
        k=max(1,int(math.ceil(0.10*len(g))))
        for m,p in preds.items(): sel[m].extend(ii[np.argsort(-p[ii],kind='mergesort')[:k]].tolist())
    allidx=np.array(allidx,dtype=int); yy=d.loc[allidx,'y'].to_numpy(); base=yy.mean()
    rows=[]
    for m,ids in sel.items():
        ids=np.array(ids,dtype=int); ys=d.loc[ids,'y'].to_numpy(); tp=int(ys.sum()); precision=ys.mean(); total_events=int(yy.sum())
        rows.append({'ranking_unit':label,'model':m,'records':len(allidx),'groups':len(eligible),'analysis_prevalence':base,'selected_n':len(ids),'precision':precision,'recall':tp/total_events,'lift_over_analysis_prevalence':precision/base,'reviews_per_endpoint':len(ids)/tp if tp else np.nan})
    return rows

def run_horizon_logistic(full,horizon):
    if horizon=='Archive closure':
        z=full.copy(); z['y']=z.y_archive.astype(int)
    else:
        days=int(horizon.split('-')[0]); z=full[full[f'evaluable{days}']].copy(); z['y']=z[f'y{days}'].astype(int)
    z=z.reset_index(drop=True)
    pr=run_experiment_grouped_purged(z,FEATURE_SETS['Full transportable'],model_names=('Logistic regression',))
    mm=metrics(z.y.to_numpy(),pr['Logistic regression'])
    return {'analysis':horizon+' outcome','validation':'Learner-purged experiment-grouped CV','model':'Logistic regression',**mm,'n':len(z),'prevalence':z.y.mean()}

def run_horizon_lightgbm(full, horizon):
    """Fit LightGBM under the same experiment-blocked, learner-purged design for an endpoint horizon."""
    if horizon == 'Archive closure':
        cohort = full.copy()
        cohort['y'] = cohort.y_archive.astype(int)
    else:
        days = int(horizon.split('-')[0])
        cohort = full[full[f'evaluable{days}']].copy()
        cohort['y'] = cohort[f'y{days}'].astype(int)
    cohort = cohort.reset_index(drop=True)
    predictions = run_experiment_grouped_purged(
        cohort, FEATURE_SETS['Full transportable'], model_names=('LightGBM',)
    )['LightGBM']
    return {
        'analysis': horizon + ' outcome',
        'validation': 'Learner-purged experiment-grouped CV',
        'model': 'LightGBM',
        **metrics(cohort.y.to_numpy(), predictions),
        'n': len(cohort), 'prevalence': float(cohort.y.mean()),
    }

# ---------- main ----------
def main():
    t0=time.time(); print('Verifying raw archive...',flush=True); verify_raw_archive(RAW)
    print('Loading raw archive...',flush=True)
    alogs,priors,slog,expstats=load_raw(RAW); full=derive_dataset(alogs,priors,slog)
    d=full[full.evaluable14].copy().reset_index(drop=True); d['y']=d.y14.astype(int)
    for col in ['experiment_id','student_id','class_id','teacher_id']:
        assert d[col].notna().all(), f'Missing grouping identifier: {col}'
    n_archive_experiments=int(d['experiment_id'].nunique())
    # cohort/source integrity summaries
    duplicate_rows=int(alogs.duplicated('assistments_reference_assignment_log_id',keep=False).sum())
    duplicate_ids=int(alogs.loc[alogs.duplicated('assistments_reference_assignment_log_id',keep=False),'assistments_reference_assignment_log_id'].nunique())
    cohort=pd.DataFrame([
        ['Experiments in archive',full.experiment_id.nunique()],['Raw assignment-log rows',len(alogs)],['Duplicated assignment reference IDs',duplicate_ids],['Unique learner-assignment events',len(full)],['Unique learners in primary cohort',d.student_id.nunique()],['Unusable assignment start times',int(full.start.isna().sum())],['Insufficient observable follow-up for 14-day endpoint',int((~full.evaluable14).sum())],['Evaluable 14-day events',len(d)],['14-day recorded noncompletion events',int(d.y.sum())],['14-day endpoint prevalence',float(d.y.mean())]])
    cohort.columns=['quantity','value']; cohort.to_csv(OUT/'cohort_summary.csv',index=False)
    expco=d.groupby('experiment_id').agg(n=('y','size'),events=('y','sum'),prevalence=('y','mean'),learners=('student_id','nunique'),classes=('class_id','nunique'),teachers=('teacher_id','nunique')).reset_index(); expco.to_csv(OUT/'experiment_cohort.csv',index=False)
    qc={'raw_assignment_rows':len(alogs),'unique_events':len(full),'start_parse_failures':int(full.start.isna().sum()),'slog_start_parse_failures':int(full.slog_start.isna().sum()),'start_timestamp_exact_matches':int((full.start_source_match_seconds==0).sum()),'finish_timestamp_exact_matches':int((full.finish_source_match_seconds.dropna()==0).sum()),'finish_both_sources':int(full.finish_source_match_seconds.notna().sum())}
    (WORK/'source_integrity.json').write_text(json.dumps(qc,indent=2),encoding='utf-8')
    print('Primary cohort',len(d),'events',int(d.y.sum()),'prevalence',round(d.y.mean(),6),flush=True)

    # Primary validation hierarchy
    all_preds={}; perf_rows=[]; exp_rows=[]
    for validation in ['Row random','Learner grouped','Learner + class grouped','Learner + teacher grouped']:
        print('Validation:',validation,flush=True); pr=run_kfold(d,validation,FEATURE_SETS['Full transportable']); all_preds[validation]=pr
        for m,p in pr.items(): perf_rows.append({'validation':validation,'model':m,**metrics(d.y,p),'n':len(d),'prevalence':d.y.mean()})
        exp_rows.append(per_experiment_metrics(d,pr,validation))
    for validation in ['Standard LOEO','Training-size-matched LOEO','Learner-purged LOEO']:
        print('Validation:',validation,flush=True); pr,pstats=run_loeo(d,validation,FEATURE_SETS['Full transportable']); all_preds[validation]=pr
        if validation=='Learner-purged LOEO': pstats.to_csv(OUT/'purge_by_experiment.csv',index=False)
        for m,p in pr.items(): perf_rows.append({'validation':validation,'model':m,**metrics(d.y,p),'n':len(d),'prevalence':d.y.mean()})
        exp_rows.append(per_experiment_metrics(d,pr,validation))
    pd.DataFrame(perf_rows).to_csv(OUT/'validation_performance.csv',index=False)
    expperf=pd.concat(exp_rows,ignore_index=True); expperf.to_csv(OUT/'experiment_level_performance.csv',index=False)
    # Compact prediction workfile, used only to build aggregate analyses; excluded from final package later.
    pwork=d[['experiment_id','student_id','class_id','teacher_id','start','y']].copy()
    for v in ['Row random','Standard LOEO','Training-size-matched LOEO','Learner-purged LOEO']:
        key=v.lower().replace(' ','_').replace('-','_').replace('+','plus')
        for m,p in all_preds[v].items(): pwork[f'{key}__{m.lower().replace(" ","_")}']=p
    pwork.to_pickle(WORK/'predictions.pkl')

    # Compact experiment-level model-gap values are retained for the headline summary.
    # Complete paired uncertainty across all validation designs is written by additional_analyses.py.
    rows=[]
    for validation in ['Row random','Standard LOEO','Training-size-matched LOEO','Learner-purged LOEO']:
        q=expperf[(expperf.validation==validation)&expperf.eligible_auroc].pivot(index='experiment_id',columns='model',values='auroc').dropna()
        delta=q['LightGBM']-q['Logistic regression']
        rows.append({'validation':validation,'eligible_experiments':len(delta),'median_delta_auroc':delta.median(),'q1':delta.quantile(.25),'q3':delta.quantile(.75),'proportion_logistic_better':float((delta<0).mean())})
    comp=pd.DataFrame(rows)

    # Extended feature-block analyses under five-fold experiment-blocked, learner-purged validation.
    print('Feature blocks...',flush=True); fb=[]
    for fs in ['Activity only','Complete learner history','Learner + class history','Full transportable','Context only']:
        pr=run_experiment_grouped_purged(d,FEATURE_SETS[fs],model_names=('Logistic regression',))
        fb.append({'validation':'Learner-purged experiment-grouped CV','model':'Logistic regression','feature_set':fs,**metrics(d.y,pr['Logistic regression'])})
    # LightGBM context ablation under row-random and experiment-blocked learner-purged validation.
    for validation in ['Row random','Learner-purged experiment-grouped CV']:
        for fs in ['Full transportable','No context']:
            if validation=='Row random':
                pr=run_kfold(d,validation,FEATURE_SETS[fs],model_names=('LightGBM',))
            else:
                pr=run_experiment_grouped_purged(d,FEATURE_SETS[fs],model_names=('LightGBM',))
            fb.append({'validation':validation,'model':'LightGBM','feature_set':fs,**metrics(d.y,pr['LightGBM'])})
    pd.DataFrame(fb).to_csv(OUT/'feature_blocks.csv',index=False)

    # Calibration summaries from strict LOEO predictions.
    print('Calibration and local updating...',flush=True); purged=all_preds['Learner-purged LOEO']; calrows=[]
    for m,p in purged.items():
        ci,cs=calibration_coefficients(d.y,p); calrows.append({'scope':'Pooled','model':m,'metric':'Calibration intercept','value':ci,'q1':np.nan,'q3':np.nan,'n_experiments':n_archive_experiments}); calrows.append({'scope':'Pooled','model':m,'metric':'Calibration slope','value':cs,'q1':np.nan,'q3':np.nan,'n_experiments':n_archive_experiments})
        ints=[]; slopes=[]
        for e,g in d.groupby('experiment_id'):
            if len(g)<100 or g.y.sum()<50 or (len(g)-g.y.sum())<50: continue
            a,b=calibration_coefficients(g.y,p[g.index.to_numpy()]);
            if np.isfinite(a) and np.isfinite(b): ints.append(a); slopes.append(b)
        for metric,vals in [('Calibration intercept',ints),('Calibration slope',slopes)]:
            calrows.append({'scope':'Experiment-level','model':m,'metric':metric,'value':np.median(vals),'q1':np.quantile(vals,.25),'q3':np.quantile(vals,.75),'n_experiments':len(vals)})
    pd.DataFrame(calrows).to_csv(OUT/'calibration_summary.csv',index=False)
    lr=local_recalibration(d,purged); lr.to_csv(OUT/'local_recalibration_by_experiment.csv',index=False)
    lrsummary=lr.groupby(['model','probability_version']).agg(experiments=('experiment_id','nunique'),evaluation_records=('n_evaluation','sum'),mean_brier=('brier','mean'),mean_log_loss=('log_loss','mean'),median_brier=('brier','median')).reset_index(); lrsummary.to_csv(OUT/'local_recalibration.csv',index=False)

    # Capacity-constrained retrospective ranking.
    print('Capacity analysis...',flush=True); cap=[]
    # Every experiment is a ranking unit, with no minimum because LOEO creates one score per record.
    for m,p in purged.items():
        selected=[]
        for e,g in d.groupby('experiment_id',sort=False):
            ii=g.index.to_numpy(); k=max(1,int(math.ceil(.10*len(g)))); selected.extend(ii[np.argsort(-p[ii],kind='mergesort')[:k]].tolist())
        ids=np.array(selected,dtype=int); ys=d.loc[ids,'y'].to_numpy(); base=d.y.mean(); tp=int(ys.sum())
        cap.append({'ranking_unit':'Experiment','model':m,'records':len(d),'groups':d.experiment_id.nunique(),'analysis_prevalence':base,'selected_n':len(ids),'precision':ys.mean(),'recall':tp/int(d.y.sum()),'lift_over_analysis_prevalence':ys.mean()/base,'reviews_per_endpoint':len(ids)/tp})
    cap += capacity_group(d,purged,['experiment_id','class_id'],'Class within experiment')
    cap += capacity_group(d,purged,['experiment_id','teacher_id'],'Teacher within experiment')
    pd.DataFrame(cap).to_csv(OUT/'capacity_screening.csv',index=False)

    # Sensitivity: outcome horizon, repeated learners, ExtraTrees structural robustness.
    print('Sensitivity analyses...',flush=True); sens=[]
    sens.append(run_horizon_logistic(full,'7-day'))
    # Include the primary horizon under the same experiment-blocked, learner-purged design.
    p14=run_experiment_grouped_purged(d,FEATURE_SETS['Full transportable'],model_names=('Logistic regression',))
    sens.append({'analysis':'14-day outcome','validation':'Learner-purged experiment-grouped CV','model':'Logistic regression',**metrics(d.y,p14['Logistic regression']),'n':len(d),'prevalence':d.y.mean()})
    for horizon in ['30-day','Archive closure']:
        sens.append(run_horizon_logistic(full,horizon))
    horizon_lgbm = [run_horizon_lightgbm(full, horizon)
                    for horizon in ['7-day', '30-day', 'Archive closure']]
    pd.DataFrame(horizon_lgbm).to_csv(OUT / 'horizon_lightgbm.csv', index=False)
    # learner-balanced and earliest-record metrics reuse strict OOF predictions (evaluation sensitivity, no refit)
    counts=d.student_id.value_counts(); w=d.student_id.map(1/counts).to_numpy();
    # weighted AUROC via sklearn sample_weight
    for m,p in purged.items():
        sens.append({'analysis':'Learner-balanced evaluation','validation':'Learner-purged LOEO','model':m,'auroc':roc_auc_score(d.y,p,sample_weight=w),'auprc':average_precision_score(d.y,p,sample_weight=w),'brier':np.average((d.y-p)**2,weights=w),'log_loss':log_loss(d.y,p,sample_weight=w),'n':len(d),'prevalence':np.average(d.y,weights=w)})
        ear=d.sort_values('start').drop_duplicates('student_id',keep='first'); ii=ear.index.to_numpy(); sens.append({'analysis':'Earliest record per learner','validation':'Learner-purged LOEO','model':m,**metrics(d.loc[ii,'y'],p[ii]),'n':len(ii),'prevalence':d.loc[ii,'y'].mean()})
    # ExtraTrees robustness under the same five-fold experiment-blocked, learner-purged design
    et=run_experiment_grouped_purged(d,FEATURE_SETS['Full transportable'],model_names=('ExtraTrees',)); sens.append({'analysis':'Structural robustness model','validation':'Learner-purged experiment-grouped CV','model':'ExtraTrees',**metrics(d.y,et['ExtraTrees']),'n':len(d),'prevalence':d.y.mean()})
    pd.DataFrame(sens).to_csv(OUT/'sensitivity_summary.csv',index=False)

    # Model specification & exact feature list.
    specs=[
        {'component':'Logistic regression','setting':'estimator','value':'SGDClassifier(loss=log_loss, penalty=l2, alpha=1e-4, average=True, max_iter=1500, tol=1e-3, random_state=20260807)'},
        {'component':'Logistic regression (LBFGS sensitivity)','setting':'estimator','value':'LogisticRegression(solver=lbfgs, L2 penalty, C=1.0, max_iter=5000, tol=1e-4); fixed without held-out tuning'},
        {'component':'LightGBM','setting':'estimator','value':'n_estimators=140; learning_rate=0.05; num_leaves=15; max_depth=5; min_child_samples=50; reg_lambda=1.0; subsample=1.0; subsample_freq=0; colsample_bytree=0.9; random_state=20260807'},
        {'component':'LightGBM alternate-seed shortlist reference','setting':'random_state','value':'20260808; same hyperparameters as primary LightGBM (random_state=20260807)'},
        {'component':'Numeric preprocessing','setting':'pipeline','value':'Training-fold median imputation with missing indicators, then standardisation'},
        {'component':'Categorical preprocessing','setting':'pipeline','value':'Training-fold most-frequent imputation and one-hot encoding; unseen categories ignored; categories with frequency <20 grouped'},
        {'component':'Validation','setting':'random_seed','value':str(SEED)},
        {'component':'Capacity','setting':'top_decile','value':'ceil(0.10*n) per ranking group; stable descending sort for ties'},
        {'component':'Local recalibration','setting':'eligibility','value':'Earliest 20% calibration cohort; update after all 14-day outcomes mature; >=30 calibration and later records, >=5 observations in each outcome class in both sets'}]
    for fs,cols in FEATURE_SETS.items(): specs.append({'component':'Feature set','setting':fs,'value':'; '.join(cols)})
    pd.DataFrame(specs).to_csv(OUT/'model_specification.csv',index=False)

    # concise headline metrics for document builder
    vp=pd.DataFrame(perf_rows).set_index(['validation','model'])
    vc=comp.set_index('validation')
    cs=pd.DataFrame(calrows)
    capdf=pd.DataFrame(cap).set_index(['ranking_unit','model'])
    h={
      'n_raw':len(alogs),'n_unique':len(full),'n_evaluable':len(d),'n_censored':int((~full.evaluable14).sum()),'events':int(d.y.sum()),'prevalence':float(d.y.mean()),'learners':int(d.student_id.nunique()),'experiments':int(d.experiment_id.nunique()),
      'eligible_auroc_experiments':int(expperf[(expperf.validation=='Learner-purged LOEO')&expperf.eligible_auroc].experiment_id.nunique()),
      'row_log_auc':float(vp.loc[('Row random','Logistic regression'),'auroc']),'row_lgb_auc':float(vp.loc[('Row random','LightGBM'),'auroc']),
      'standard_log_auc':float(vp.loc[('Standard LOEO','Logistic regression'),'auroc']),'standard_lgb_auc':float(vp.loc[('Standard LOEO','LightGBM'),'auroc']),
      'matched_log_auc':float(vp.loc[('Training-size-matched LOEO','Logistic regression'),'auroc']),'matched_lgb_auc':float(vp.loc[('Training-size-matched LOEO','LightGBM'),'auroc']),
      'purged_log_auc':float(vp.loc[('Learner-purged LOEO','Logistic regression'),'auroc']),'purged_lgb_auc':float(vp.loc[('Learner-purged LOEO','LightGBM'),'auroc']),
      'purged_log_auprc':float(vp.loc[('Learner-purged LOEO','Logistic regression'),'auprc']),'purged_lgb_auprc':float(vp.loc[('Learner-purged LOEO','LightGBM'),'auprc']),
      'row_delta_median':float(vc.loc['Row random','median_delta_auroc']),'purged_delta_median':float(vc.loc['Learner-purged LOEO','median_delta_auroc']),'purged_log_better':float(vc.loc['Learner-purged LOEO','proportion_logistic_better']),
      'purge_fraction_median':float(pstats.purged_fraction.median()),'purge_fraction_mean':float(pstats.purged_fraction.mean()),'purge_fraction_max':float(pstats.purged_fraction.max()),
      'local_recal_experiments':int(lr.experiment_id.nunique()) if len(lr) else 0,
      'cap_log_precision':float(capdf.loc[('Experiment','Logistic regression'),'precision']),'cap_lgb_precision':float(capdf.loc[('Experiment','LightGBM'),'precision'])}
    pd.DataFrame([h]).to_csv(OUT/'headline_metrics.csv',index=False)
    print('DONE in',round((time.time()-t0)/60,1),'minutes',flush=True); print(json.dumps(h,indent=2),flush=True)

if __name__=='__main__': main()
