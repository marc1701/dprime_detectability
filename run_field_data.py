from dprime_detectability import *
import soundfile as sf

n, fs = sf.read('walk1_stop1.wav')
s, _ = sf.read('walk1_stop3.wav')

dpd = DPrimeDetectability(s, masker=n, fs=fs)
_ = dpd.discounted_level(plot=True)
detectability = dpd.d_prime(plot=True)
p50, p95, pmax = dpd.d_prime_single_vals(plot=True)

print(f'Detectability, Flyover in CityTraffic Ambience -- 50th: {p50}, 95th: {p95}, Max:{pmax}')

n, _ = sf.read('walk1_stop6.wav')
s, _ = sf.read('walk1_stop8.wav')

dpd = DPrimeDetectability(s, masker=n, fs=fs)
_ = dpd.discounted_level(plot=True)
detectability = dpd.d_prime(plot=True)
p50, p95, pmax = dpd.d_prime_single_vals(plot=True)

print(f'Detectability, Flyover in GreenSpace Ambience -- 50th: {p50}, 95th: {p95}, Max:{pmax}')
