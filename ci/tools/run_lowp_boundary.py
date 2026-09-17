#!/usr/bin/env python3
"""Execute the fixed lowp-boundary baseline/candidate GPU jobs sequentially."""
import argparse,hashlib,json,os,pathlib,subprocess,time
import run_mtp_teacher_forced_r9700 as service
ROOT=pathlib.Path(__file__).resolve().parents[2]
UUIDS={'gfx1030':'GPU-76a08c022586fed6','gfx1201':'GPU-a8e9ddefa2d60f55'}
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=pathlib.Path);a=p.parse_args();m=json.loads(a.manifest.read_text());target=m['target'];uuid=UUIDS[target];out=pathlib.Path(m['output']);out.mkdir(parents=True,exist_ok=False)
 devices=[p.parent for p in pathlib.Path('/sys/class/drm').glob('card[0-9]*/device/unique_id') if p.read_text().strip().lower()==uuid[4:].lower()];assert len(devices)==1
 level=devices[0]/'power_dpm_force_performance_level';r={'state':'running','target':target,'uuid':uuid,'manifest_sha256':sha(a.manifest),'started':time.time(),'performance_level_before':level.read_text().strip(),'jobs':[]};stopped=False
 def save():
  temp=out/'execution.tmp';temp.write_text(json.dumps(r,indent=2)+'\n');temp.replace(out/'execution.json')
 try:
  if target=='gfx1030':
   status=subprocess.check_output(['/home/homelab1/.local/bin/qwen38-subagent-server','status'],text=True)
   if 'process:  stopped' not in status:raise RuntimeError('Qwen pair must be idle and stopped')
  else:
   r['service_was_active']=service.active();r['service_hashes_before']=service.service_hashes()
   if r['service_was_active']:
    clients=subprocess.check_output(['ss','-Htn','state','established','( sport = :8000 )'],text=True)
    if clients.strip() or service.health('/readyz')!=200:raise RuntimeError('resident service busy or unready')
    stopped=True;subprocess.run(['systemctl','--user','stop',service.UNIT],check=True)
  save()
  for job in m['jobs']:
   d=out/job['name'];d.mkdir();env={k:v for k,v in os.environ.items() if not k.startswith('SLLM_') and k not in ['ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES']};env.update(ROCR_VISIBLE_DEVICES=uuid,LD_LIBRARY_PATH='/opt/rocm/lib');env.update(job.get('env',{}))
   if job.get('make_prefix'):
    perf=json.loads((out/'nvfp4-performance/report.json').read_text());tokens=list(map(int,(ROOT/'.local-artifacts/phase85/inputs/coding-8192.csv').read_text().strip().split(',')));assert len(tokens)==8192
    row=perf['rows'][0];run=next(x for x in row['runs'] if x['sample_kind']=='measured');generated=run['generated_tokens'];assert len(generated)==128
    prefix={'schema':'phase85-a16-mtp-prefix-v1','entries':[{'case_id':'lowp-boundary-8192-128','seed':123,'prompt_tokens':tokens,'output_prefix_tokens':generated}]};(out/'nvfp4-prefix.json').write_text(json.dumps(prefix)+'\n')
   command=job['command'];item={'name':job['name'],'command':command,'environment':job.get('env',{}),'binary_sha256':sha(command[0]),'started':time.time()};r['jobs'].append(item)
   with (d/'report.json').open('w') as stdout,(d/'stderr.log').open('w') as stderr:
    proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr);item['pid']=proc.pid;save();print(target,job['name'],proc.pid,flush=True);item['exit_code']=proc.wait()
   item['elapsed_seconds']=time.time()-item['started'];item['report_sha256']=sha(d/'report.json');save()
   if item['exit_code']!=0:raise RuntimeError(f"{job['name']} exit {item['exit_code']}")
   if job.get('capture'):
    capture=pathlib.Path(job['capture']);doc=json.loads(capture.read_text());item['capture_sha256']=sha(capture);item['capture_file']=str(capture)
    rep=doc['repeat'];assert rep['bitwise_identical'] and doc['cleanup']['final_cleanup_empty']
    for key in ['primary_dispatch','repeat_dispatch']:
     audit=rep[key];assert audit['selected_backend']=='hip' and audit['target']==target and audit['all_dispatches_hip'] and not audit['fallback_used'] and audit['kernel_dispatch_count']>0
    save()
   elif job.get('json',True):
    doc=json.loads((d/'report.json').read_text())
    if doc.get('state')!='PASS':raise RuntimeError(f"{job['name']} non-PASS report")
   else:
    if 'PASS' not in (d/'report.json').read_text():raise RuntimeError('text evidence has no PASS marker')
   print(target,job['name'],'PASS',round(item['elapsed_seconds'],1),flush=True)
  r['state']='complete'
 except Exception as error:r['state']='failed';r['error']=str(error)
 finally:
  if stopped:
   subprocess.run(['systemctl','--user','start',service.UNIT],check=False)
   for _ in range(180):
    if service.active() and service.health('/healthz')==service.health('/readyz')==200:break
    time.sleep(1)
   r['restored_health']=service.health('/healthz');r['restored_ready']=service.health('/readyz')
  if target=='gfx1201' and 'service_was_active' in r:
   r['service_hashes_after']=service.service_hashes();r['service_restored']=(service.active()==r['service_was_active'] and r['service_hashes_before']==r['service_hashes_after'] and (not stopped or r.get('restored_health')==r.get('restored_ready')==200))
   if not r['service_restored']:r['state']='failed'
  r['performance_level_restored']=level.read_text().strip()==r['performance_level_before']
  if not r['performance_level_restored']:r['state']='failed'
  r['ended']=time.time();save()
 print(r['state'],r.get('error',''),flush=True);return 0 if r['state']=='complete' else 1
if __name__=='__main__':raise SystemExit(main())
