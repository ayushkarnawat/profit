"""
Ideally we should compare many different models to (a) observe which gives the best results, 
(b) is efficent, and (c) explainable. 
1. Linear regression
2. gaussian processes
3. Feed-forward network (FFN)
4. Graph Convolution Network (GCN)
5. others

See https://www.biorxiv.org/content/biorxiv/early/2018/06/02/337154.full.pdf and
https://www.biorxiv.org/content/biorxiv/early/2019/03/26/589333.full.pdf
"""

from profit.models import gcn
from profit.models.gcn import GCN