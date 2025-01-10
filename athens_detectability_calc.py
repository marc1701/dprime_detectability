import numpy as np
from dprime_detectability import *
import soundfile as sf
import glob
import json

gs_files = glob.glob('Trimmed_Audio/drone_active/GreenSpace/*.wav')
ct_files = glob.glob('Trimmed_Audio/drone_active/CityTraffic/*.wav')

gs_masker_static = np.loadtxt('greenspace_avg_masker.csv')
ct_masker_static = np.loadtxt('citytraffic_avg_masker.csv')

dprime_levels = {}
for file in gs_files:
    s, fs = sf.read(file)
    dpd = DPrimeDetectability(s, masker_static=gs_masker_static, fs=fs)
    p50, p95, pmax, pint = dpd.d_prime_single_vals(plot=True)
    print(f'Detectability, {file} -- 50th: {p50}, 95th: {p95}, Max:{pmax}, Integrated:{pint}')

    dprime_levels[file] = [p50, p95, pmax, pint]

for file in ct_files:
    s, fs = sf.read(file)
    dpd = DPrimeDetectability(s, masker_static=ct_masker_static, fs=fs)
    p50, p95, pmax, pint = dpd.d_prime_single_vals(plot=True)
    print(f'Detectability, {file} -- 50th: {p50}, 95th: {p95}, Max:{pmax}, Integrated:{pint}')

    dprime_levels[file] = [p50, p95, pmax, pint]

with open('dprime_athens.json', 'w', encoding='utf-8') as f:
    json.dump(dprime_levels, f, ensure_ascii=False, indent=4)