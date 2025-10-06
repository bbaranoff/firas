#!/usr/bin/env python3
"""
Compare un spectre expérimental du rayonnement du corps noir (CMB mesuré par COBE/FIRAS)
à la loi de Planck, et quantifie les écarts.

Données : FIRAS CMB Monopole Spectrum (NASA LAMBDA)
- Page : https://lambda.gsfc.nasa.gov/product/cobe/firas_monopole_get.html
- Fichier : https://lambda.gsfc.nasa.gov/data/cobe/firas/monopole_spec/firas_monopole_spec_v1.txt

Ce script :
1) Télécharge le fichier (ou utilise une copie embarquée si le réseau est indisponible).
2) Convertit le nombre d'onde (cm^-1) en fréquence (Hz).
3) Calcule B_ν(T) (loi de Planck) en MJy/sr et ajuste T par moindres carrés pondérés.
4) Affiche des statistiques (T ajustée, χ² réduit) et sauvegarde deux figures PNG :
   - spectre_mesure_vs_planck.png (spectre + fit)
   - residus_kJy.png (résidus expérimentaux et résidus du fit)

Dépendances : numpy, scipy, pandas (optionnel), matplotlib, requests
   pip install numpy scipy pandas matplotlib requests

Usage :
   python script_blackbody_firas_compare.py

"""
from __future__ import annotations
import io
import sys
import math
import textwrap
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import requests
except ImportError:  # requests n'est pas indispensable si on utilise le fallback
    requests = None

# --- Constantes physiques ---
h = 6.62607015e-34       # J s
kB = 1.380649e-23        # J/K
c = 2.99792458e8         # m/s
c_cm_s = 2.99792458e10   # cm/s (pour convertir cm^-1 -> Hz)
JY = 1e-26               # 1 Jansky = 1e-26 W/m^2/Hz

FIRAS_URL = "https://lambda.gsfc.nasa.gov/data/cobe/firas/monopole_spec/firas_monopole_spec_v1.txt"

# --- Copie embarquée du tableau FIRAS (si le téléchargement échoue) ---
# Colonnes : (wavenumber_cm^-1, I_MJy_per_sr, residual_kJy_per_sr, sigma_kJy_per_sr, galaxy_kJy_per_sr)
FALLBACK_DATA = """
2.27 200.723 5 14 4
2.72 249.508 9 19 3
3.18 293.024 15 25 -1
3.63 327.770 4 23 -1
4.08 354.081 19 22 3
4.54 372.079 -30 21 6
4.99 381.493 -30 18 8
5.45 383.478 -10 18 8
5.90 378.901 32 16 10
6.35 368.833 4 14 10
6.81 354.063 -2 13 12
7.26 336.278 13 12 20
7.71 316.076 -22 11 25
8.17 293.924 8 10 30
8.62 271.432 8 11 36
9.08 248.239 -21 12 41
9.53 225.940 9 14 46
9.98 204.327 12 16 57
10.44 183.262 11 18 65
10.89 163.830 -29 22 73
11.34 145.750 -46 22 93
11.80 128.835 58 23 98
12.25 113.568 6 23 105
12.71 99.451 -6 23 121
13.16 87.036 6 22 135
13.61 75.876 -17 21 147
14.07 65.766 6 20 160
14.52 57.008 26 19 178
14.97 49.223 -12 19 199
15.43 42.267 -19 19 221
15.88 36.352 8 21 227
16.34 31.062 7 23 250
16.79 26.580 14 26 275
17.24 22.644 -33 28 295
17.70 19.255 6 30 312
18.15 16.391 26 32 336
18.61 13.811 -26 33 363
19.06 11.716 -6 35 405
19.51 9.921 8 41 421
19.97 8.364 26 55 435
20.42 7.087 57 88 477
20.87 5.801 -116 155 519
21.33 4.523 -432 282 573
""".strip()

@dataclass
class FIRASSample:
    wn_cm: float           # nombre d'onde [cm^-1]
    I_MJy_sr: float        # intensité (monopole) [MJy/sr]
    resid_kJy_sr: float    # résidu [kJy/sr]
    sigma_kJy_sr: float    # incertitude [kJy/sr]
    galaxy_kJy_sr: float   # modèle Galaxie [kJy/sr]


def load_firas_text() -> np.ndarray:
    """Télécharge le fichier FIRAS (texte) et renvoie un tableau (N,5).
    Si le téléchargement échoue, utilise FALLBACK_DATA.
    """
    text = None
    if requests is not None:
        try:
            resp = requests.get(FIRAS_URL, timeout=10)
            resp.raise_for_status()
            # Extraire uniquement les nombres (certaines versions mettent des commentaires sur 1-2 lignes)
            # On supprime toutes les lignes commençant par '#'
            lines = [ln for ln in resp.text.splitlines() if not ln.strip().startswith('#') and ln.strip()]
            text = "\n".join(lines)
        except Exception:
            text = None
    if not text:
        text = FALLBACK_DATA
    # Certaines versions du fichier mettent tout sur 1 ligne après les commentaires → split générique
    arr = np.fromstring(text, sep=' ')
    if arr.size % 5 != 0:
        raise RuntimeError("Format inattendu des données FIRAS (pas multiple de 5 colonnes)")
    data = arr.reshape((-1, 5))
    return data


def planck_Bnu_MJy_sr(nu_hz: np.ndarray, T: float) -> np.ndarray:
    """Loi de Planck B_ν(T) en MJy/sr, pour un tableau de fréquences en Hz."""
    # B_ν = 2 h ν^3 / c^2 / (exp(hν/kT) - 1)  [W m^-2 Hz^-1 sr^-1]
    x = (h * nu_hz) / (kB * T)
    # gérer les grandes valeurs de x numériquement (éviter overflow)
    # np.expm1(x) est plus stable pour petits x
    denom = np.expm1(x)
    Bnu_W = (2.0 * h * nu_hz**3) / (c**2 * denom)
    # Conversion W m^-2 Hz^-1 sr^-1 -> MJy/sr (1 MJy = 1e6 * 1e-26 W/m^2/Hz)
    Bnu_MJy_sr = Bnu_W / (1e-20)
    return Bnu_MJy_sr


def fit_temperature(nu_hz: np.ndarray, I_MJy_sr: np.ndarray, sigma_kJy_sr: np.ndarray) -> Tuple[float, float, float]:
    """Ajuste T (K) par moindres carrés pondérés.
    Retourne (T_fit, chi2_red, dof).
    """
    from scipy.optimize import minimize_scalar

    # Convertir incertitudes en MJy/sr
    sigma_MJy_sr = sigma_kJy_sr / 1000.0

    def chi2_of_T(T: float) -> float:
        model = planck_Bnu_MJy_sr(nu_hz, T)
        r = (I_MJy_sr - model) / sigma_MJy_sr
        return float(np.sum(r**2))

    res = minimize_scalar(chi2_of_T, bounds=(2.0, 3.5), method='bounded')
    T_fit = float(res.x)
    chi2 = chi2_of_T(T_fit)
    dof = max(1, I_MJy_sr.size - 1)
    chi2_red = chi2 / dof
    return T_fit, chi2_red, dof


def main() -> None:
    data = load_firas_text()
    wn = data[:, 0]  # cm^-1
    I_MJy = data[:, 1]
    resid_kJy = data[:, 2]
    sigma_kJy = data[:, 3]
    # galaxy_kJy = data[:, 4]

    # Conversion en fréquence [Hz] : f = c(cm/s) * wn(cm^-1)
    nu = c_cm_s * wn  # Hz

    # Ajuster T
    T_fit, chi2_red, dof = fit_temperature(nu, I_MJy, sigma_kJy)

    # Modèles
    I_fit = planck_Bnu_MJy_sr(nu, T_fit)
    I_planck_2725 = planck_Bnu_MJy_sr(nu, 2.725)

    # Résidus (MJy/sr)
    resid_fit_MJy = I_MJy - I_fit
    resid_tab_MJy = resid_kJy / 1000.0

    # Stats sur les résidus (kJy/sr pour lisibilité)
    def stats_kJy(x_MJy):
        x = x_MJy * 1000.0
        return float(np.mean(x)), float(np.std(x)), float(np.max(np.abs(x)))

    m_fit, s_fit, a_fit = stats_kJy(resid_fit_MJy)
    m_tab, s_tab, a_tab = stats_kJy(resid_tab_MJy)

    # Impressions console
    print("\n=== Comparaison FIRAS vs Planck ===")
    print(f"N points         : {nu.size}")
    print(f"T ajustée (K)    : {T_fit:.6f}")
    print(f"χ² réduit        : {chi2_red:.3f} (dof={dof})")
    print("-- Résidus (kJy/sr) --")
    print(f"Fit : moyenne={m_fit:.3f}, sigma={s_fit:.3f}, max|.|={a_fit:.3f}")
    print(f"Tab : moyenne={m_tab:.3f}, sigma={s_tab:.3f}, max|.|={a_tab:.3f}")

    # Figure 1 : Spectre mesuré vs Planck
    fig1 = plt.figure(figsize=(7, 5))
    # Abscisse : on affiche en nombre d'onde (cm^-1) comme dans FIRAS
    order = np.argsort(wn)
    wn_s = wn[order]
    I_MJy_s = I_MJy[order]
    I_fit_s = I_fit[order]
    I_2725_s = I_planck_2725[order]

    plt.plot(wn_s, I_MJy_s, marker='o', linestyle='none', label='FIRAS (monopole)')
    plt.plot(wn_s, I_fit_s, label=f'Planck T_fit={T_fit:.4f} K')
    plt.plot(wn_s, I_2725_s, linestyle='--', label='Planck 2.725 K')
    plt.xlabel('Nombre d\'onde $\~\nu$ (cm$^{-1}$)')
    plt.ylabel('Intensité $B_\nu$ (MJy/sr)')
    plt.title('Spectre CMB (COBE/FIRAS) vs loi de Planck')
    plt.legend()
    plt.tight_layout()
    plt.savefig('spectre_mesure_vs_planck.png', dpi=150)

    # Figure 2 : Résidus en kJy/sr
    fig2 = plt.figure(figsize=(7, 4))
    plt.axhline(0.0, linestyle='--')
    plt.plot(wn_s, resid_fit_MJy[order] * 1000.0, marker='o', linestyle='none', label='Résidus (mesure - fit)')
    plt.plot(wn_s, resid_tab_MJy[order] * 1000.0, linestyle='-', label='Résidus tableau FIRAS')
    plt.xlabel('Nombre d\'onde $\~\nu$ (cm$^{-1}$)')
    plt.ylabel('kJy/sr')
    plt.title('Résidus du spectre monopole')
    plt.legend()
    plt.tight_layout()
    plt.savefig('residus_kJy.png', dpi=150)

    print("\nFichiers écrits : spectre_mesure_vs_planck.png, residus_kJy.png")


if __name__ == '__main__':
    main()
