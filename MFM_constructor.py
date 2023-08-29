import numpy as np

# for the MFM model, we construct (Z_t, Y_{t+1})
# where Z_t = (r_t, F_t, ..., F_{t+1-k})
# for k = 2, an example would be
# Z_1 = (r_1, F_1, F_0), Y_2 = P_2 - P_1
# Z_2 = (r_2, F_2, F_1), Y_3 = P_3 - P_2
# ...
# Z_n = (r_n, F_n, F_{n-1}), Y_{n+1} = P_{n+1} - P_n 

def MFM_Bond_dataConstruct(
        b_vals,
        rates,
        factors,
        n_dependence
):
    full_N, K = factors.shape
    _, m = rates.shape
    assert len(b_vals) == full_N and len(rates) == full_N

    Z_data = np.zeros((full_N - n_dependence, n_dependence * K + m))
    Y_data = np.zeros(full_N - n_dependence)

    for i in range(n_dependence-1, full_N-1):
        r_part = rates[i]
        f_part = np.reshape(factors[i+1-n_dependence:i+1], -1)
        y = b_vals[i+1]-b_vals[i]

        Z_data[i+1-n_dependence,:m] = r_part
        Z_data[i+1-n_dependence, m:] = f_part

        Y_data[i+1-n_dependence] = y

    return Z_data, Y_data
