### double machine learning plot
from dml import double_ml

dml_object = double_ml(1,2,3,1000)

YY = []
XX = []
DD = []
N = 1000
MC_no = 500

theta_est = dml_object.run(YY, XX, DD, N, MC_no)

theta_ols       = list(map(lambda x: x[0], theta_est))
theta_naive_dml = list(map(lambda x: x[1], theta_est))
theta_cross_dml = list(map(lambda x: x[2], theta_est))
    
sns.distplot(theta_ols, bins = MC_no)
sns.distplot(theta_naive_dml, bins = MC_no)
sns.distplot(theta_cross_dml, bins = MC_no)

plt.savefig('dml_estimator_distribution.png')

























