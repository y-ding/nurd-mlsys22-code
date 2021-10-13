import pickle

outdirectory = '/scratch/avinashrao/causal-inference/clusterdata-2011-2/'
jobSCoutfile = 'jobSC.pickle'
jobparallelismoutfile = 'jobparallelism.pickle'

with open(outdirectory+jobSCoutfile, 'rb') as handle:
    schedulingclass = pickle.load(handle)

with open(outdirectory+jobparallelismoutfile, 'rb') as handle:
    parallelism = pickle.load(handle)

k = 0
tasktotal = 0
for x in schedulingclass:
    if x in parallelism:
        if schedulingclass[x] >=1 and parallelism [x] >= 20:
            print(schedulingclass[x],parallelism[x])
            k = k + 1
            tasktotal = tasktotal + parallelism[x]

print(k,"of",len(schedulingclass))
print(tasktotal, " tasks")