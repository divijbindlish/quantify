import sys

from math import exp
import numpy as np
import numpy.random as random

random.seed()

def func(sample, vol, T):
    return exp((vol * (T ** 0.5) * sample) - (((vol ** 2) * T) / 2))

def estimate_price(X, vol, T, K, rho, nsamples):
    dim = len(X)
    mean = np.zeros(dim)
    cov = np.zeros([dim, dim])
    cov.fill(rho)
    for i in range(dim):
        cov[i][i] = 1.0

    solution = 0

    for sample in random.multivariate_normal(mean, cov, nsamples):
        x = np.prod([X[i] * func(sample[i], vol[i], T) for i in range(dim)])
        if x > K:
            solution += x - K

    return solution / nsamples

for line in sys.stdin:
    parts = line.strip().split(',')

    test_case = int(parts[0])
    question_type = int(parts[1])

    if question_type == 1:
        expiration, option_type, X, strike, vol = map(float, parts[2:])

        price = X - strike
        if price < 0:
            price = 0.0

        print str(test_case) + ',' + str(question_type) + ',' \
            + '{0:.4f}'.format(round(price, 4))

    elif question_type == 2:
        correlation, expiration, option_type, X1, X2, strike, vol1, vol2 \
            = map(float, parts[2:])

        X = [X1, X2]
        vol = [vol1, vol2]

        initial_estimate = estimate_price(X, vol, expiration, strike, correlation, 10**3)
        if initial_estimate == 0.0:
            price = initial_estimate
        else:
            price = estimate_price(X, vol, expiration, strike, correlation, 10**5)

        print str(test_case) + ',' + str(question_type) + ',' \
            + '{0:.4f}'.format(round(price, 4))

    elif question_type == 3:
        correlation, expiration, option_type = map(float, parts[2:5])
        X = map(float, parts[5:14])
        strike = float(parts[14])
        vol = map(float, parts[15:])

        initial_estimate = estimate_price(X, vol, expiration, strike, correlation, 10**3)
        if initial_estimate == 0.0:
            price = initial_estimate
        else:
            price = estimate_price(X, vol, expiration, strike, correlation, 10**6)

        print str(test_case) + ',' + str(question_type) + ',' \
            + '{0:.4f}'.format(round(price, 4))

    elif question_type == 4:
        expiration, option_type, price = map(float, parts[2:5])
        X = map(float, parts[5:14])
        strike = float(parts[14])
        vol = map(float, parts[15:])

        initial_estimate = estimate_price(X, vol, expiration, strike, 0.0, 10**3)

        if initial_estimate == 0:
            print str(test_case) + ',' + str(question_type) + ',NA'
            continue

        lo = 0.0
        hi = 1.0
        prev = -1
        while True:
            mid = (lo + hi) / 2.0

            estimate = estimate_price(X, vol, expiration, strike, mid, 10**5)
            if estimate > price:
                hi = mid
            else:
                lo = mid

            drho = mid - prev
            prev = mid
            if drho < 0: drho = -drho
            if drho < 1e-4: break

        print str(test_case) + ',' + str(question_type) + ',' \
            + '{0:.4f}'.format(round(prev, 4))

    else:
        vol1, vol2, correlation, T, X1, X2, K1, K2, notional \
            = map(float, parts[2:])

        mean = np.zeros(2)
        cov = np.zeros([2, 2])
        cov.fill(correlation)
        cov[0][0] = cov[1][1] = 1.0

        solution = 0

        nsamples = 10**7
        for sample in random.multivariate_normal(mean, cov, nsamples):
            x1 = X1 * func(sample[0], vol1, T)
            if x1 <= K1:
                continue

            x2 = X2 * func(sample[1], vol2, T)
            if x2 > K2:
                solution += x2 - K2

        price = notional * solution / nsamples

        print str(test_case) + ',' + str(question_type) + ',' \
            + str(int(round(price)))
