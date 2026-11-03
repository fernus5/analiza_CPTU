import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import datetime
import io

# ==========================================
# KONFIGURACJA STRONY I STYLE CSS
# ==========================================
st.set_page_config(page_title="Analizator Profilów CPTU", layout="wide")

st.markdown("""
    <style>
    .stTabs [data-baseweb="tab"] p {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# INICJALIZACJA PARAMETRÓW SESJI (URL ROUTING)
# ==========================================
params = st.query_params
if "client_id" in params:
    session_client = params["client_id"]
    if "session_initialized" not in st.session_state:
        st.session_state.session_initialized = True
        init_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.toast(f"Aktywowano dedykowaną instancję środowiska: {session_client} ({init_time})", icon="⚙️")

# ==========================================
# NAGŁÓWEK APLIKACJI
# ==========================================
st.title("Wieloparametrowa Analiza Sondowań CPTU (Robertson SBT)")
st.markdown("Automatyczna ekstrakcja, filtracja szumów i logarytmiczna klasyfikacja rodzaju gruntu (SBT $I_c$) ze standardowych plików pomiarowych maszyn geotechnicznych.")

# ==========================================
# PANEL BOCZNY: STEROWANIE I IMPORT
# ==========================================
st.sidebar.markdown("### 1. Źródło danych")
uploaded_file = st.sidebar.file_uploader("Przeciągnij plik sondy (.txt, .csv)", type=["txt", "csv"])

st.sidebar.markdown("### 2. Cyfrowa Filtracja Sygnału")
st.sidebar.write("Dane z sondy bywają zaszumione wibracjami żerdzi. Użyj suwaka, aby wygładzić wykres (algorytm średniej ruchomej).")
smooth_window = st.sidebar.slider("Okno uśredniania (ilość pomiarów)", min_value=1, max_value=50, value=5, step=1)

# Kolory dla klasyfikacji Robertsona
color_map = {
    "Grunty organiczne / Torfy (Ic > 3.6)": "#5C4033",
    "Iły i gliny (2.95 < Ic <= 3.6)": "#8B4513",
    "Pyły gliniaste (2.60 < Ic <= 2.95)": "#D2691E",
    "Pyły piaszczyste (2.05 < Ic <= 2.60)": "#F4A460",
    "Piaski (1.31 < Ic <= 2.05)": "#F0E68C",
    "Piaski grube / Żwiry (Ic <= 1.31)": "#A9A9A9"
}

def classify_sbt(ic_val):
    if ic_val > 3.60: return "Grunty organiczne / Torfy (Ic > 3.6)", color_map["Grunty organiczne / Torfy (Ic > 3.6)"]
    elif ic_val > 2.95: return "Iły i gliny (2.95 < Ic <= 3.6)", color_map["Iły i gliny (2.95 < Ic <= 3.6)"]
    elif ic_val > 2.60: return "Pyły gliniaste (2.60 < Ic <= 2.95)", color_map["Pyły gliniaste (2.60 < Ic <= 2.95)"]
    elif ic_val > 2.05: return "Pyły piaszczyste (2.05 < Ic <= 2.60)", color_map["Pyły piaszczyste (2.05 < Ic <= 2.60)"]
    elif ic_val > 1.31: return "Piaski (1.31 < Ic <= 2.05)", color_map["Piaski (1.31 < Ic <= 2.05)"]
    else: return "Piaski grube / Żwiry (Ic <= 1.31)", color_map["Piaski grube / Żwiry (Ic <= 1.31)"]

# ==========================================
# GŁÓWNA LOGIKA PROGRAMU
# ==========================================
if uploaded_file is not None:
    try:
        # 1. Parsowanie pliku w locie (z pamięci RAM)
        data = []
        content = uploaded_file.getvalue().decode("utf-8").splitlines()
        
        for line in content:
            if line.startswith("D="):
                parts = line.strip().split() # Poprawka podziału (usunięto sztywny znacznik tabulacji)
                row = {}
                for p in parts:
                    if "=" in p:
                        key, val = p.split("=", 1)
                        try: row[key.strip()] = float(val.strip())
                        except ValueError: pass
                
                if "D" in row and "QC" in row and "FS" in row:
                    data.append(row)
                    
        df = pd.DataFrame(data)
        
        if df.empty:
            st.error("Nie wykryto poprawnych znaczników danych (D=, QC=, FS=). Sprawdź strukturę pliku.")
        else:
            # 2. Obliczenia geotechniczne
            df['QC'] = df['QC'].clip(lower=0.01) # Zabezpieczenie przed zerami
            df['fs_MPa'] = df['FS'] / 1000.0
            df['Rf'] = (df['fs_MPa'] / df['QC']) * 100.0
            if 'U' not in df.columns: df['U'] = 0.0
            
            # Wygładzanie danych szumowych (Rolling Mean)
            df['QC_smooth'] = df['QC'].rolling(window=smooth_window, min_periods=1, center=True).mean()
            df['fs_smooth'] = df['fs_MPa'].rolling(window=smooth_window, min_periods=1, center=True).mean()
            df['Rf_smooth'] = df['Rf'].rolling(window=smooth_window, min_periods=1, center=True).mean()
            df['U_smooth'] = df['U'].rolling(window=smooth_window, min_periods=1, center=True).mean()

            # Obliczanie Wskaźnika SBT (Ic) wg Robertsona
            df['Q_norm'] = (df['QC_smooth'] / 0.1).clip(lower=1.0)
            df['F_norm'] = df['Rf_smooth'].clip(lower=0.1)
            
            df['Ic'] = np.sqrt((3.47 - np.log10(df['Q_norm']))**2 + (np.log10(df['F_norm']) + 1.22)**2)
            
            # Mapowanie klas i kolorów
            df['Litologia'], df['Kolor'] = zip(*df['Ic'].apply(classify_sbt))
            
            # ==========================================
            # ZAKŁADKI Z WYNIKAMI
            # ==========================================
            tab_wykres, tab_stat, tab_tabela = st.tabs(["◧ Profil CPTU i Litologia", "📊 Zestawienie Gruntów", "☰ Surowe Dane"])
            
            with tab_wykres:
                st.subheader(f"Wizualizacja profilu sondowania: {uploaded_file.name}")
                
                # Rysowanie wykresu 5-kolumnowego
                fig, axes = plt.subplots(1, 5, sharey=True, figsize=(14, 18), dpi=200, gridspec_kw={'width_ratios': [1, 1, 1, 1, 0.5]})
                fig.subplots_adjust(wspace=0.15)
                
                c_qc, c_fs, c_rf, c_u2 = '#4169E1', '#DC143C', '#000000', '#800080'
                
                # Oś 1: qc
                axes[0].plot(df['QC_smooth'], df['D'], color=c_qc, lw=1.5)
                axes[0].set_xlabel('Opor stożka $q_c$ [MPa]', fontweight='bold')
                axes[0].set_xlim(0, max(25, df['QC_smooth'].max() * 1.1))
                axes[0].set_ylabel('Głębokość poniżej p.t. [m]', fontweight='bold', fontsize=12)
                
                # Oś 2: fs
                axes[1].plot(df['fs_smooth'], df['D'], color=c_fs, lw=1.5)
                axes[1].set_xlabel('Tarcie na tulei $f_s$ [MPa]', fontweight='bold')
                axes[1].set_xlim(0, max(0.2, df['fs_smooth'].max() * 1.1))
                
                # Oś 3: Rf
                axes[2].plot(df['Rf_smooth'], df['D'], color=c_rf, lw=1.5)
                axes[2].set_xlabel('Wskaźnik tarcia $R_f$ [%]', fontweight='bold')
                axes[2].set_xlim(0, max(10, df['Rf_smooth'].max() * 1.1))
                
                # Oś 4: U2
                axes[3].plot(df['U_smooth'], df['D'], color=c_u2, lw=1.5)
                axes[3].set_xlabel('Ciśnienie porowe $u_2$ [kPa]', fontweight='bold')
                axes[3].set_xlim(min(-50, df['U_smooth'].min() - 20), max(200, df['U_smooth'].max() + 50))
                
                # Oś 5: Profil SBT (kolorowe bloki)
                axes[4].set_xlabel('Litologia\n(SBT $I_c$)', fontweight='bold')
                axes[4].set_xticks([])
                
                # Renderowanie kolorowego profilu w najszybszy sposób
                dy = df['D'].diff().bfill().mean() # Poprawka wywołania metody .bfill() pod nową bibliotekę Pandas
                axes[4].barh(df['D'], width=1, height=dy*1.1, color=df['Kolor'], align='center', edgecolor='none')
                axes[4].set_xlim(0, 1)

                axes[0].invert_yaxis()
                
                for i, ax in enumerate(axes):
                    ax.xaxis.tick_top()
                    ax.xaxis.set_label_position('top')
                    if i < 4:
                        ax.grid(True, which='major', linestyle='-', linewidth=0.8, color='#A9A9A9')
                        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='#D3D3D3')
                        ax.minorticks_on()
                
                # Legenda litologiczna pod wykresem
                legend_patches = [mpatches.Patch(color=color_map[k], label=k) for k in color_map]
                fig.legend(handles=legend_patches, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01), frameon=True, fontsize=10)
                fig.subplots_adjust(bottom=0.08)
                
                st.pyplot(fig)
            
            with tab_stat:
                st.subheader("Udział poszczególnych warstw w profilu")
                statystyki = df['Litologia'].value_counts(normalize=True) * 100
                stat_df = statystyki.reset_index()
                stat_df.columns = ["Rodzaj gruntu (SBT)", "Udział % profilu"]
                
                col_st1, col_st2 = st.columns([1, 2])
                with col_st1:
                    st.dataframe(stat_df.style.format({"Udział % profilu": "{:.1f}%"}), hide_index=True)
                with col_st2:
                    st.info("Klasyfikacja SBT (Soil Behavior Type) opiera się na relacji między znormalizowanym oporem stożka a wskaźnikiem tarcia. Pozwala na bezrdzeniową, precyzyjną ocenę układu warstw nośnych i słabonośnych w masywie gruntowym.")
            
            with tab_tabela:
                st.subheader("Obliczone parametry cyfrowe")
                format_dict = {'D': '{:.2f}', 'QC_smooth': '{:.3f}', 'fs_smooth': '{:.3f}', 'Rf_smooth': '{:.2f}', 'U_smooth': '{:.1f}', 'Ic': '{:.2f}'}
                st.dataframe(df[['D', 'QC_smooth', 'fs_smooth', 'Rf_smooth', 'U_smooth', 'Ic', 'Litologia']]
                             .rename(columns={'D': 'Głębokość [m]', 'QC_smooth': 'qc [MPa]', 'fs_smooth': 'fs [MPa]', 'Rf_smooth': 'Rf [%]', 'U_smooth': 'u2 [kPa]', 'Ic': 'Wskaźnik Ic'})
                             .style.format(format_dict), 
                             width="stretch", hide_index=True)

    except Exception as e:
        st.error(f"Wystąpił błąd podczas analizy pliku. Upewnij się, że struktura to CPTU (D=, QC=, FS=). Szczegóły: {e}")
        
else:
    # Ekran powitalny, gdy nie wgrano pliku
    st.info("👋 Czekam na dane z sondowania. Przeciągnij plik z urządzenia (zawierający klucze D=, QC=, FS=) do panelu bocznego po lewej stronie, aby rozpocząć analizę.")
