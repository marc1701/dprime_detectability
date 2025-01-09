from dprime_detectability import *

n, fs = sf.read('Park3_0002_0027_N2_BINREC.wav')
s, fs = sf.read('A_M300_F_1_R_BIN.wav')

dpd = DPrimeDetectability(s, n, fs)
_ = dpd.discounted_level(plot=True)
detectability = dpd.d_prime(plot=True)
p50, p95, pmax = dpd.d_prime_single_vals(plot=True)

print(f'Detectability, Flyover in Park Ambience -- 50th: {p50}, 95th: {p95}, Max:{pmax}')

n, fs = sf.read('BusyStreet8_0435_0500_N2_BINREC.wav')
s, fs = sf.read('A_M300_F_1_Rrun_labdata.py_BIN.wav')

dpd = DPrimeDetectability(s, n, fs)
_ = dpd.discounted_level(plot=True)
detectability = dpd.d_prime(plot=True)
p50, p95, pmax = dpd.d_prime_single_vals(plot=True)

print(f'Detectability, Flyover in BusyStreet Ambience -- 50th: {p50}, 95th: {p95}, Max:{pmax}')
