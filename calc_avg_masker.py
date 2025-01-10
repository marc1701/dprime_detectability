from dprime_detectability import *
import soundfile as sf
import glob


city_traffic_ambience_files = glob.glob('Trimmed_Audio/drone_inactive/CityTraffic/*.wav')

# clunky as you want to init the object - needs sorting really
x, fs = sf.read(city_traffic_ambience_files[0])
dpd = DPrimeDetectability(x, x)

x_it = []
for file in city_traffic_ambience_files:
    x, fs = sf.read(file)
    x = x[:, 0]
    x_it.append(dpd.third_octave_spl(x, fs))

x_it = np.array(x_it)
x_it_mean_iter =  np.mean(x_it, axis=0)
avg_masker_ct = np.mean(x_it_mean_iter, axis=0)
plt.stairs(10*np.log10(avg_masker_ct), baseline=None)

green_space_ambience_files = glob.glob('Trimmed_Audio/drone_inactive/GreenSpace/*.wav')

gs_it = []
for file in green_space_ambience_files:
    gs, fs = sf.read(file)
    gs = gs[:, 0]
    gs_it.append(dpd.third_octave_spl(gs, fs))

gs_it = np.array(gs_it)
gs_it_mean_iter =  np.mean(gs_it, axis=0)
avg_masker_gs = np.mean(gs_it_mean_iter, axis=0)

plt.stairs(10*np.log10(avg_masker_gs), baseline=None)
plt.show()

ct_avg_arr = np.array([avg_masker_ct] * 120)
gs_avg_arr = np.array([avg_masker_gs] * 120)
