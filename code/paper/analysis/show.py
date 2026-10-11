import json,sys
S=json.load(open('audit_sample.json'))
for c in sys.argv[1:]:
    print('=====',c)
    for k,r in enumerate(S[c]):
        L=110 if r['strict'] is not None else 300
        t=r['tail'][-L:].replace('\n',' / ')
        print(f"[{k}] g={r['gold']} p={r['parsed']} s={r['strict']} gen={int(r['gen'])} :: {t}")
