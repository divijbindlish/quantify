import bisect
import math
import sys

lines = sys.stdin.readlines()

n = int(lines[0].strip().split(' ')[1])

table = {}

def getRowErrors(v, i):
    size = len(v)

    sumx = v[i][0]
    sumy = v[i][1]
    sumxy = v[i][0] * v[i][1]
    sumx2 = v[i][0] * v[i][0]
    sumy2 = v[i][1] * v[i][1]

    errors = [float(0.0) for _ in range(len(v))]

    for j in range(i+1, size):
        sumx = sumx + v[j][0]
        sumy = sumy + v[j][1]
        sumxy = sumxy + v[j][0] * v[j][1]
        sumx2 = sumx2 + v[j][0] * v[j][0]
        sumy2 = sumy2 + v[j][1] * v[j][1]

        n = float(j - i + 1.0)

        M = (n * sumxy - sumx * sumy) / (n * sumx2 - sumx * sumx)
        C = (sumy / n) - M * (sumx / n)

        errors[j] = sumy2 \
            + M * M * sumx2 \
            + C * C * n \
            - 2 * M * sumxy \
            - 2 * C * sumy \
            + 2 * M * C * sumx

    return errors

for i in range(1, n+1):
    parts = lines[i].strip().split()
    timestamp, symbol = int(parts[0]), parts[1]

    for j in range(len(parts[2:]) / 2):
        field, value = parts[2 + 2*j], int(parts[3 + 2*j])

        key = symbol + field
        if key in table.keys():
            table[key].append((timestamp, value, i))
        else:
            table[key] = [(timestamp, value, i)]

print 'tickfile completed'

for line in lines[n+1:]:
    parts = line.strip().split()
    query = parts[0]

    if query == 'sum':
        start_time, end_time, symbol, field \
            = int(parts[1]), int(parts[2]), parts[3], parts[4]

        result = 0
        if symbol+field in table.keys():
            vector = table[symbol+field]
            search, _, _ = zip(*vector)

            lo = bisect.bisect_left(search, start_time)
            hi = bisect.bisect_right(search, end_time)

            for i in range(lo, hi):
                result += vector[i][1]

        print result

    elif query == 'product':
        start_time, end_time, symbol, field1, field2 \
            = int(parts[1]), int(parts[2]), parts[3], parts[4], parts[5]

        result = 0
        if symbol+field1 in table.keys() and symbol+field2 in table.keys():
            vector1 = table[symbol+field1]
            vector2 = table[symbol+field2]

            search1, _, tick1 = zip(*vector1)
            search2, _, tick2 = zip(*vector2)

            lo1 = bisect.bisect_left(search1, start_time)
            hi1 = bisect.bisect_right(search1, end_time)

            lo2 = bisect.bisect_left(search2, start_time)
            hi2 = bisect.bisect_right(search2, end_time)

            j = lo2

            for i in range(lo1, hi1):
                if j == hi2: break

                if vector1[i][2] == vector2[j][2]:
                    result += vector1[i][1] * vector2[j][1]
                elif vector2[j][2] < vector1[i][2]:
                    while vector2[j][2] < vector1[i][2]:
                        j += 1
                        if j == hi2: break

                    if j == hi2: break
                    if vector1[i][2] == vector2[j][2]:
                        result += vector1[i][1] * vector2[j][1]

        print result

    elif query == 'max':
        start_time, end_time, symbol, field, k \
            = int(parts[1]), int(parts[2]), parts[3], parts[4], int(parts[5])

        result = 0
        if symbol+field in table.keys():
            vector = table[symbol+field]
            search, values, _ = zip(*vector)

            lo = bisect.bisect_left(search, start_time)
            hi = bisect.bisect_right(search, end_time)

            values = list(values)[lo:hi]
            values.sort(reverse=True)
            values = values[:k]

            print ' '.join(map(str, values)),

        print ''

    else:
        symbol, field, k = parts[1], parts[2], int(parts[3])

        vector = table[symbol+field]
        size = len(vector)

        e = [getRowErrors(vector, i) for i in range(size)]

        if size == 1:
            print k
        else:
            dp = [float(0.0) for _ in range(size)]
            for i in range(1, size):
                dp[i] = k + e[0][i]
                for j in range(1, i):
                    q = dp[j] + k + e[j+1][i]
                    if dp[i] > q:
                        dp[i] = q


        print int(math.ceil(round(dp[size-1], 2)))
