"""Select tile profiles by validation gain, an asset budget and a dot-cost proxy."""
import argparse
import json
import math
from displacement_analyze import OUT
from displacement_assets import sha
from project_paths import REPO_ROOT


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('screen')
    ap.add_argument('--name',default='stage11_profiles');args=ap.parse_args()
    path=REPO_ROOT/'docs'/(args.screen+'.json');r=json.loads(path.read_text());groups={}
    for curve in r['curves']:
        for item in curve:
            key=item['factors'].rsplit('_',1)[1]
            value=dict(item)
            value['net_dot_dimensions']=item['expected_extra_dot_dimensions']-(128*item['coverage'] if item['direct'] else 0)
            groups.setdefault(key,[]).append(value)
    profiles=[];unit=65536
    for budget in (16_000_000,20_000_000,24_000_000):
        for penalty in (0.,.002,.005,.01):
            capacity=(budget-4*266305-32-16*len(groups))//unit
            states={0:(0.,[])}
            for key,options in sorted(groups.items()):
                updated={}
                for used,(score,selected) in states.items():
                    for item in options:
                        cost=math.ceil(item['asset_bytes_excluding_map']/unit);total=used+cost
                        if total>capacity:continue
                        value=score+item['valid_proxy_gain']-penalty*item['net_dot_dimensions']
                        if total not in updated or value>updated[total][0]:updated[total]=(value,selected+[item])
                states=updated
            assert states,'budget cannot hold any complete profile'
            value,selected=max(states.values(),key=lambda x:x[0]);gain=sum(s['valid_proxy_gain'] for s in selected)
            name=f'{args.name}_b{budget//1000000}_p{int(penalty*1000)}'
            profile=dict(name=name,asset_budget=budget,dot_penalty=penalty,parts=selected,
                         valid_proxy_accuracy=r['baseline_valid_accuracy']+gain,
                         local_bytes=4*266305+32+16*len(selected)+sum(s['asset_bytes_excluding_map'] for s in selected),
                         net_dot_dimensions=sum(s['net_dot_dimensions'] for s in selected),test_labels_used=False)
            profiles.append(profile);print(name,profile['valid_proxy_accuracy'],profile['local_bytes'],profile['net_dot_dimensions'],flush=True)
    report=dict(screen=args.screen,screen_sha256=sha(path),selected_branch_proxy=True,test_labels_used=False,profiles=profiles)
    (REPO_ROOT/'docs'/(args.name+'.json')).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
