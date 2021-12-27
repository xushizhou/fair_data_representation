#!/usr/bin/env python
# coding: utf-8

# # Implement of "Fair Regression with Wasserstein Barycenters"

# In[ ]:


# train and test sensitive variable and prediction: Z_l(t), Y_l(t)
Z_l = trainZ_list # for CRIME data, use trainZ_list[i] = trainX_list[i][:,96]
Z_t = testZ_list # for CRIME data, use testZ_list[i] = testX_list[i][:,96]
Y_l = Y_hat_list_l # for ANN results, use Y_hat_list_l_ANN instead
Y_t = Y_hat_list_t # for ANN results, use Y_hat_list_t_ANN instead

# function that finds the barycenter by matching the cumulative distribution functions of two marginal/conditional random variables
def f_NIPS(Y_l, Y_t, Z_l, Z_t):
    u = np.unique(Z_l)
    nM = max([sum(Z_l==u[0]),sum(Z_l==u[1])])
    iM = np.argmax([sum(Z_l==u[0]),sum(Z_l==u[1])])
    nm = min([sum(Z_l==u[0]),sum(Z_l==u[1])])
    im = np.argmin([sum(Z_l==u[0]),sum(Z_l==u[1])])
    p = nm/len(Z_l)
    q = 1-p
    YF = np.zeros(len(Y_t))
    for i in range(0,len(Y_t)):
        print(i)
        if Z_t[i] == u[im]:
            dist_best = math.inf
            for t in np.linspace(min(Y_l),max(Y_l),100):
                tmp1 = sum(Y_l[Z_l==u[iM]] < t)/nM
                tmp2 = sum(Y_l[Z_l==u[im]] < Y_t[i])/nm
                dist = np.abs(tmp1-tmp2)
                if dist_best > dist:
                    dist_best = dist
                    ts = t
            YF[i] = p*Y_t[i]+q*ts
        else:
            dist_best = math.inf
            for t in np.linspace(min(Y_l),max(Y_l),100):
                tmp1 = sum(Y_l[Z_l==u[im]] < t)/nm
                tmp2 = sum(Y_l[Z_l==u[iM]] < Y_t[i])/nM
                dist = np.abs(tmp1-tmp2)
                if dist_best > dist:
                    dist_best = dist
                    ts = t
            YF[i] = q*Y_t[i]+p*ts
    return YF

# compute the predictions on the test sets via the post-processing approach
Y_hat_chzhen_list = []
time_chzhen_list = []

for i in range(len(trainX_list)):
    t = perf_counter()
    Y_hat_chzhen_list.append(f_NIPS(Y_l[i], Y_t[i], Z_l[i], Z_t[i]))
    time_chzhen_list.append(perf_counter() - t)
    
# compute MSE and KS
MSE_chzhen_list = []
KS_chzhen_list = []

for i in range(len(trainX_list)):
    MSE_chzhen_list.append(((testY_list[i] - Y_hat_chzhen_list[i])**2).mean(axis = 0))   # MSE
    KS_result_chzhen = stats.ks_2samp(Y_hat_chzhen_list[i][Z_t[i] == 0], Y_hat_chzhen_list[i][Z_t[i] == 1]) # KS
    KS_chzhen_list.append(KS_result_chzhen[0])

