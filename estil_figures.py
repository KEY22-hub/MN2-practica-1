"""
Aquesta pagina defineix l'aspecte  de totes les gràfiques de l'informe. Per importar-lo on les volem, cal afegir al codi: 

    import estil_figures
    from estil_figures import desa
"""
import os

import matplotlib.pyplot as plt

# Carpeta de sortida. Es calcula a partir de la ubicació d'aquest fitxer, no del directori des d'on s'executa el programa, perquè les figures
# acabin sempre a figures/ dins del repositori encara que algú executi un script des d'una altra carpeta.
CARPETA_FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# plt.rcParams és el diccionari de configuració global de matplotlib. Tot el que posem aquí s'aplica a totes les figures creades després d'importar el fitxer.
plt.rcParams.update({
    # mida 
    "figure.figsize": (3.4, 2.6),

    # tipus de lletra
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],

    # Fa que les fórmules entre $...$ (per exemple \hat{x}) es dibuixin amb la
    # mateixa tipografia que les de LaTeX, sense necessitar tenir LaTeX instal·lat.
    "mathtext.fontset": "cm",

    # Fa que els números dels eixos també es dibuixin amb la tipografia
    # matemàtica. Sense això, amb cmr10 el signe menys surt malament.
    "axes.formatter.use_mathtext": True,

    # Mides de lletra llegibles a la mida final de 3.4 polzades.
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,

    # Ticks cap a dins als quatre costats: direction="in" els gira cap a
    # dins de la figura, i top/right els afegeixen a dalt i a la dreta.
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,

    # Figura tancada: els quatre eixos (spines) visibles formen un requadre.
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.bottom": True,
    "axes.spines.left": True,

    # Punt decimal i no coma. 
    "axes.formatter.use_locale": False,

    # Línies una mica més fines que les de per defecte, i llegenda sense marc per no tapar les corbes.
    "lines.linewidth": 1.2,
    "legend.frameon": False,

    # Format de sortida per defecte: PDF
    "savefig.format": "pdf",
})


def desa(fig, nom):
    """
    Desa la figura fig a figures/nom.pdf i retorna la ruta del fitxer.

    nom va sense extensió, per exemple desa(fig, "explicit_errors").
    """
    # Crea la carpeta figures/ si encara no existeix. exist_ok=True evita
    # l'error si ja hi és.
    os.makedirs(CARPETA_FIGURES, exist_ok=True)

    # Reajusta els marges perquè les etiquetes dels eixos no es tallin.
    fig.tight_layout()

    ruta = os.path.join(CARPETA_FIGURES, nom + ".pdf")
    fig.savefig(ruta)
    return ruta
