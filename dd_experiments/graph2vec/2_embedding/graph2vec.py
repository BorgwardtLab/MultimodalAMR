module load EasyBuild
module load Python/3.7.4-GCCcore-8.3.0

cd /massstorage/URT/GEN/BIO3/PRIV/Team/Diane/RESEARCH/Secondment/August/graphEmbedding/threshold/non_attributed/hackathon

python

import networkx as nx
from karateclub.graph_embedding import Graph2Vec
from ind_54 import ind

#Import graph
H = nx.Graph(ind)
graphs=[H]

# Graph2Vec generic example
model = Graph2Vec()
model.fit(graphs)
model.get_embedding()

#Export
f = open("results54.txt", "w")
f.write(str(model.get_embedding()))
f.close()
