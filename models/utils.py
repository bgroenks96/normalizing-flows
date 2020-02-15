import numpy as np

def update_metrics(metric_dict, **kwargs):
    for k,v in kwargs.items():
        if k in metric_dict:
            prev, n = metric_dict[k]
            metric_dict[k] = ((v + n*prev) / (n+1), n+1)
        else:
            metric_dict[k] = (v, 0)
