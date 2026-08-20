import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime
import io

# Energie functie voor kalibratie
def energie_functie(K, a, b, c):
    return a + b * K + c * (K ** 2)

# Fit-helpers
def fit_energy_calibration(calibration_points):
    if not calibration_points:
        return 0.0, 0.0, 0.0
    energies = np.array([p[0] for p in calibration_points])
    channels = np.array([p[1] for p in calibration_points])
    if len(calibration_points) < 3:
        if len(calibration_points) == 2:
            try:
                popt_linear, _ = curve_fit(lambda K, a_val, b_val: a_val + b_val * K, channels, energies)
                return popt_linear[0], popt_linear[1], 0.0
            except RuntimeError:
                return 0.0, 0.0, 0.0
        elif len(calibration_points) == 1:
            if channels[0] != 0:
                return 0.0, energies[0] / channels[0], 0.0
            else:
                return 0.0, 0.0, 0.0
    try:
        popt, pcov = curve_fit(energie_functie, channels, energies)
        return popt[0], popt[1], popt[2]
    except Exception:
        return 0.0, 0.0, 0.0

# Parser voor .PHD-inhoud (werkt met tekst)
def parse_phd_from_text(text):
    lines = text.splitlines()
    data = {
        'g_spectrum': [], 'b_spectrum': [], 'g_energy_cal': [], 'b_energy_cal': [], 'histogram': [],
        'air_volume': None, 'collection_start_datetime': None, 'collection_end_datetime': None,
        'xenon_volume': None, 'collection_date': None
    }
    current_section = None
    expect_collection_data = False
    expect_processing_data = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('#'):
            current_section = line
            expect_collection_data = (current_section == '#Collection')
            expect_processing_data = (current_section == '#Processing')
            continue
        parts = line.split()
        if not parts:
            continue
        if expect_collection_data:
            if len(parts) >= 5:
                try:
                    start_date_str_with_slash = parts[0]
                    start_time_str_with_sec = parts[1]
                    end_date_str_with_slash = parts[2]
                    end_time_str_with_sec = parts[3]
                    data['air_volume'] = float(parts[4])
                    start_dt_str = f"{start_date_str_with_slash} {start_time_str_with_sec}"
                    end_dt_str = f"{end_date_str_with_slash} {end_time_str_with_sec}"
                    data['collection_start_datetime'] = pd.to_datetime(start_dt_str, format='%Y/%m/%d %H:%M:%S.%f', errors='coerce')
                    data['collection_end_datetime'] = pd.to_datetime(end_dt_str, format='%Y/%m/%d %H:%M:%S.%f', errors='coerce')
                    if pd.notnull(data['collection_start_datetime']):
                        data['collection_date'] = data['collection_start_datetime'].date()
                except Exception:
                    pass
            else:
                try:
                    data['air_volume'] = float(parts[-1])
                except Exception:
                    pass
            expect_collection_data = False
        elif expect_processing_data:
            if len(parts) >= 1:
                try:
                    data['xenon_volume'] = float(parts[0])
                except Exception:
                    pass
            expect_processing_data = False
        elif current_section == '#g_Spectrum':
            if len(parts) > 2 and parts[0] != '256':
                try:
                    data['g_spectrum'].extend([int(x) for x in parts[1:]])
                except Exception:
                    pass
        elif current_section == '#b_Spectrum':
            if len(parts) > 2 and parts[0] != '256':
                try:
                    data['b_spectrum'].extend([int(x) for x in parts[1:]])
                except Exception:
                    pass
        elif current_section == '#g_Energy':
            try:
                data['g_energy_cal'].append((float(parts[0]), float(parts[1])))
            except Exception:
                pass
        elif current_section == '#b_Energy':
            if len(parts) >= 4 and parts[1] == 'C':
                try:
                    data['b_energy_cal'].append((float(parts[0]), float(parts[2])))
                except Exception:
                    pass
            elif len(parts) >= 3:
                try:
                    data['b_energy_cal'].append((float(parts[0]), float(parts[1])))
                except Exception:
                    pass
        elif current_section == '#Histogram':
            if parts[0] != '256':
                try:
                    data['histogram'].extend([int(x) for x in parts if x != 'STOP'])
                except Exception:
                    pass
    return data

# Functie om histogram te normaliseren en te berekenen voor g-diagram
# Now supports an energy range (min,max) instead of a single threshold
def compute_normalized_sums_for_files(parsed_list, energy_range_kev=(0, 300)):
    energy_min_kev, energy_max_kev = energy_range_kev
    collection_datetimes = []
    normalized_sums_total = []
    normalized_sums_in_range = []
    normalized_sums_outside_range = []

    for data in parsed_list:
        if not data['histogram']:
            continue
        if not data['air_volume'] or not data['xenon_volume']:
            continue
        if data['collection_start_datetime'] is None or data['collection_end_datetime'] is None:
            continue
        collection_duration_seconds = (data['collection_end_datetime'] - data['collection_start_datetime']).total_seconds()
        collection_duration_hours = collection_duration_seconds / 3600.0 if collection_duration_seconds else 0
        if collection_duration_hours == 0:
            continue
        normalization_factor = (data['air_volume'] * collection_duration_hours * data['xenon_volume'])
        if normalization_factor == 0:
            continue
        hist_array = np.array(data['histogram'])
        try:
            hist_array = hist_array.reshape((256, 256))
        except Exception:
            continue
        total_histogram_sum = np.sum(hist_array)
        normalized_sums_total.append(total_histogram_sum / normalization_factor)
        if not data['g_spectrum'] or not data['g_energy_cal']:
            normalized_sums_in_range.append(0.0)
            normalized_sums_outside_range.append(0.0)
        else:
            g_coeffs = fit_energy_calibration(data['g_energy_cal'])
            g_spectrum_arr = np.array(data['g_spectrum'])
            g_channels = np.arange(len(g_spectrum_arr))
            g_energies = energie_functie(g_channels, *g_coeffs)
            if len(g_energies) == 0:
                start_idx = 0
                end_idx = 0
            else:
                # find closest channel indices for min and max energies
                idx_min = int(np.argmin(np.abs(g_energies - energy_min_kev)))
                idx_max = int(np.argmin(np.abs(g_energies - energy_max_kev)))
                start_idx = min(idx_min, idx_max)
                end_idx = max(idx_min, idx_max) + 1
            sum_in_range = np.sum(g_spectrum_arr[start_idx:end_idx])
            sum_outside = np.sum(g_spectrum_arr[:start_idx]) + np.sum(g_spectrum_arr[end_idx:])
            normalized_sums_in_range.append(sum_in_range / normalization_factor)
            normalized_sums_outside_range.append(sum_outside / normalization_factor)
        collection_datetimes.append(data['collection_start_datetime'])

    if not collection_datetimes:
        return None
    df = pd.DataFrame({
        'Collection Datetime': collection_datetimes,
        'Normalized Sum (Total)': normalized_sums_total,
        'Normalized Sum (In Range)': normalized_sums_in_range,
        'Normalized Sum (Outside Range)': normalized_sums_outside_range
    })
    df = df.sort_values(by='Collection Datetime').reset_index(drop=True)
    return df

# Streamlit UI
st.set_page_config(page_title='Data-analyse gamma-beta (Streamlit)', layout='wide')
st.title('Data-analyse gamma-beta')
st.markdown('Upload één of meerdere .PHD-bestanden. Eén bestand → coïncidentiespectrum; meerdere bestanden → g-diagrammen.')

uploaded_files = st.file_uploader('Upload .PHD bestanden', type=['PHD', 'phd', 'PHD.txt'], accept_multiple_files=True)
# Replace slider with two number inputs so users can type min and max
# Two separate number inputs for min and max energy (keV)
energy_min = st.sidebar.number_input('Energie min (keV) voor g-diagrammen', min_value=0, max_value=2000, value=0, step=1)
energy_max = st.sidebar.number_input('Energie max (keV) voor g-diagrammen', min_value=0, max_value=2000, value=300, step=1)
# If the user accidentally sets min > max, swap them for processing but inform the user
if energy_min > energy_max:
    st.sidebar.warning('Min is groter dan max — waarden worden omgewisseld voor verwerking')
    energy_range = (int(energy_max), int(energy_min))
else:
    energy_range = (int(energy_min), int(energy_max))

show_dummy = st.sidebar.checkbox('Toon demo met gegenereerde dummy bestanden', value=False)

parsed_files = []
if uploaded_files:
    for uploaded in uploaded_files:
        try:
            raw = uploaded.read()
            # decode; try utf-8 then latin-1
            try:
                text = raw.decode('utf-8')
            except Exception:
                text = raw.decode('latin-1')
            data = parse_phd_from_text(text)
            # If collection datetimes are pandas Timestamp, convert to python datetime
            if isinstance(data.get('collection_start_datetime'), pd.Timestamp):
                data['collection_start_datetime'] = data['collection_start_datetime'].to_pydatetime()
            if isinstance(data.get('collection_end_datetime'), pd.Timestamp):
                data['collection_end_datetime'] = data['collection_end_datetime'].to_pydatetime()
            parsed_files.append(data)
        except Exception as e:
            st.warning(f'Kon bestand niet verwerken: {uploaded.name} — {e}')

# Demo mode: genereer enkele dummy datasets (lichtgewicht)
if show_dummy and not uploaded_files:
    import random
    from datetime import timedelta
    dummy_list = []
    base_date = datetime(2023, 1, 1, 8, 0, 0)
    for i in range(6):
        # maak een eenvoudige dummy data-structuur
        start_dt = base_date + timedelta(days=i)
        end_dt = start_dt + timedelta(hours=12)
        air_volume = 0.2 + i * 0.01
        xenon_volume = 0.02 + i * 0.002
        g_spectrum = list((np.random.poisson(10 + i, size=256)).astype(int))
        histogram = (np.random.poisson(5 + i / 2, size=(256, 256))).astype(int).flatten().tolist()
        g_energy_cal = [(50.0, 20.0), (200.0, 80.0), (600.0, 240.0)]
        dummy_list.append({
            'g_spectrum': g_spectrum,
            'b_spectrum': [],
            'g_energy_cal': g_energy_cal,
            'b_energy_cal': [],
            'histogram': histogram,
            'air_volume': air_volume,
            'xenon_volume': xenon_volume,
            'collection_start_datetime': start_dt,
            'collection_end_datetime': end_dt
        })
    parsed_files = dummy_list

if not parsed_files:
    st.info('Upload een of meerdere .PHD-bestanden of schakel demo-modus in.')
else:
    # --- SINGLE FILE VIEW: show gamma, beta and coincidence spectra in three columns ---
    if len(parsed_files) == 1:
        data = parsed_files[0]
        st.subheader('Coïncidentiespectrum en metadata')
        st.write('Collectie start:', data.get('collection_start_datetime'))
        st.write('Collectie eind :', data.get('collection_end_datetime'))
        st.write('Luchtvolume (m3):', data.get('air_volume'))
        st.write('Xenonvolume (m3):', data.get('xenon_volume'))

        # Maak drie kolommen: gamma | beta | coincidentie
        col_g, col_b, col_h = st.columns([1, 1, 1.2])

        # --- GAMMA SPECTRUM ---
        with col_g:
            st.markdown("**Gammaspectrum**")
            if data['g_spectrum']:
                g_spectrum_arr = np.array(data['g_spectrum'])
                g_channels = np.arange(len(g_spectrum_arr))
                if data['g_energy_cal']:
                    coeffs = fit_energy_calibration(data['g_energy_cal'])
                    g_energies = energie_functie(g_channels, *coeffs)
                    fig_g, ax_g = plt.subplots(figsize=(6, 3.5))
                    ax_g.step(g_energies, g_spectrum_arr, where='mid')
                    ax_g.set_xlabel('Energie (keV)')
                    ax_g.set_ylabel('Counts')
                    ax_g.set_title('G-spectrum (gekalibreerd)')
                    ax_g.grid(True)
                    st.pyplot(fig_g)
                else:
                    fig_g, ax_g = plt.subplots(figsize=(6, 3.5))
                    ax_g.step(g_channels, g_spectrum_arr, where='mid')
                    ax_g.set_xlabel('Kanaal')
                    ax_g.set_ylabel('Counts')
                    ax_g.set_title('G-spectrum (kanaal)')
                    ax_g.grid(True)
                    st.pyplot(fig_g)
            else:
                st.warning('Geen G-spectrum gevonden in het bestand.')

        # --- BETA SPECTRUM ---
        with col_b:
            st.markdown("**Betaspectrum**")
            if data['b_spectrum']:
                b_spectrum_arr = np.array(data['b_spectrum'])
                b_channels = np.arange(len(b_spectrum_arr))
                # probeer b-calibratie als aanwezig
                if data.get('b_energy_cal'):
                    try:
                        b_coeffs = fit_energy_calibration(data['b_energy_cal'])
                    except Exception:
                        b_coeffs = (0.0, 1.0, 0.0)
                    b_energies = energie_functie(b_channels, *b_coeffs)
                    fig_b, ax_b = plt.subplots(figsize=(6, 3.5))
                    ax_b.step(b_energies, b_spectrum_arr, where='mid')
                    ax_b.set_xlabel('Energie (keV)')
                    ax_b.set_ylabel('Counts')
                    ax_b.set_title('B-spectrum (gekalibreerd)')
                    ax_b.grid(True)
                    st.pyplot(fig_b)
                else:
                    fig_b, ax_b = plt.subplots(figsize=(6, 3.5))
                    ax_b.step(b_channels, b_spectrum_arr, where='mid')
                    ax_b.set_xlabel('Kanaal')
                    ax_b.set_ylabel('Counts')
                    ax_b.set_title('B-spectrum (kanaal)')
                    ax_b.grid(True)
                    st.pyplot(fig_b)
            else:
                st.warning('Geen B-spectrum gevonden in het bestand.')

        # --- COINCIDENTIE HISTOGRAM (2D) ---
        with col_h:
            st.markdown("**Coïncidentie-histogram (2D)**")
            if data['histogram']:
                hist_array = np.array(data['histogram'])
                try:
                    hist_array = hist_array.reshape((256, 256))
                    fig_h, ax_h = plt.subplots(figsize=(5.5, 5.0))
                    cax = ax_h.imshow(hist_array, origin='lower', cmap='inferno', aspect='auto')
                    ax_h.set_title('Coïncidentie-histogram (256x256)')
                    fig_h.colorbar(cax, ax=ax_h, fraction=0.046, pad=0.04)
                    st.pyplot(fig_h)
                except Exception:
                    st.warning('Histogram kon niet naar 256x256 gereshaped worden.')
            else:
                st.warning('Geen histogramdata gevonden in het bestand.')
    else:
        st.subheader('G-diagrammen voor meerdere bestanden')
        df = compute_normalized_sums_for_files(parsed_files, energy_range_kev=energy_range)
        if df is None:
            st.warning('Geen geldige datapunten gevonden om g-diagrammen te maken.')
        else:
            # Plot totale genormaliseerde som
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df['Collection Datetime'], df['Normalized Sum (Total)'], marker='o', linestyle='-', color='blue')
            ax.set_title('G-diagram: Totale Genormaliseerde Som van Coïncidenties')
            ax.set_xlabel('Collectiestartdatum en -tijd')
            ax.set_ylabel('Genormaliseerde Som')
            ax.grid(True)
            fig.autofmt_xdate()
            st.pyplot(fig)

            # In-range
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(df['Collection Datetime'], df['Normalized Sum (In Range)'], marker='o', linestyle='-', color='green')
            ax2.set_title(f'G-diagram: Genormaliseerde Som van Coïncidenties binnen {energy_range[0]}–{energy_range[1]} keV')
            ax2.set_xlabel('Collectiestartdatum en -tijd')
            ax2.set_ylabel('Genormaliseerde Som (Counts binnen bereik)')
            ax2.grid(True)
            fig2.autofmt_xdate()
            st.pyplot(fig2)

            # Outside-range
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.plot(df['Collection Datetime'], df['Normalized Sum (Outside Range)'], marker='o', linestyle='-', color='red')
            ax3.set_title(f'G-diagram: Genormaliseerde Som van Coïncidenties buiten {energy_range[0]}–{energy_range[1]} keV')
            ax3.set_xlabel('Collectiestartdatum en -tijd')
            ax3.set_ylabel('Genormaliseerde Som (Counts buiten bereik)')
            ax3.grid(True)
            fig3.autofmt_xdate()
            st.pyplot(fig3)

st.markdown('---')
st.markdown('Tips: draai lokaal met `streamlit run streamlit_app.py`. Voor deployment op Streamlit Cloud kun je dit repository verbinden en de app starten.')
