import numpy as np

def get_bootstrap_sample(sample, statistic, nResamples):
    func_values = []
    for i in range(nResamples):
        resample = np.random.choice(sample, size=len(sample), replace=True)
        func_values.append(statistic(resample))
    return func_values

def compute_bootstrap_CI(boot_sample, significance, two_sided=True, side="up"):
    if two_sided:
        ci = np.percentile(boot_sample, q=[100*significance/2,
                                           100*(1-significance/2)])
    else:
        if side == up:
            ci = np.percentile(boot_sample, p=100*significance)
        else:
            ci = np.percentile(boot_sample, p=100*(1-significance))
    return ci
