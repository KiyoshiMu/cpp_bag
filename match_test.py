import random

pops = [True for _ in range(26)]
pops.extend(False for _ in range(34))
random.shuffle(pops)
options = [True, False]
test = []
times = 1000000
for _ in range(times):
    preds = [random.choice(options) for _ in range(len(pops))]
    test.append(sum(pred == pop for (pred, pop) in zip(preds, pops)))
print(sum(test) / times)
print(sum([t >= 46 for t in test]) / times)