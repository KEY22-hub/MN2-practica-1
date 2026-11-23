# Pràctica 1 – Mètodes Numèrics II: ablació cardíaca IVS

Codi de la pràctica 1 de Mètodes Numèrics II.
Resol l'equació de la calor adimensional amb Euler explícit, Euler implícit i
Crank–Nicolson, i calcula durant quant de temps cal aplicar el senyal de 40 V.

**Autors:** Paula López Batista, Martí Cornacchionne Lliteres i Mark Krakkai Poza.

## Requisits

- Python 3
- NumPy i Matplotlib:

```
pip install -r requeriments.txt
```

## Com executar-ho

Tots els fitxers s'executen des de la carpeta arrel del repositori, per exemple:

```
python euler_explicit.py
```

Les figures es desen automàticament a la carpeta `figures/` en format PDF.

| Fitxer | Què fa | Sortida |

| `parametres.py` | Dades físiques, escales de l'adimensionalització i funcions de conversió | Imprimeix els valors derivats |

| `estil_figures.py` | Estil comú de totes les figures | — (l'importen els altres fitxers) |

| `gauss_seidel.py` | Mètode de Gauss–Seidel per a sistemes tridiagonals | — (l'importen els altres fitxers) |

| `sol_analitica.py` | Solució analítica i error de truncament de la sèrie | Comprovacions per pantalla |

| `euler_explicit.py` | Euler explícit amb Δt = 0.51, 0.49 i 0.25 (Δx)² | `explicit_solucio_error.pdf` |

| `euler_imp.py` | Euler implícit amb Δt = 0.25, 0.5 i 1 (Δx)² | `implicit_solucio_error.pdf` |

| `crank_nicolson.py` | Crank–Nicolson amb Δt = 0.5 i 1 (Δx)² | `cn_solucio.pdf`, `cn_error.pdf` |

| `comparacio.py` | Comparació dels tres mètodes | `comparacio_error.pdf`, `comparacio_convergencia.pdf` |

| `iteracions_gs.py` | Iteracions de Gauss–Seidel segons el punt de partida | Taula per pantalla |

| `problema.py` | Solució del problema: temps òptim d'aplicació | `problema_perfils.pdf`, `problema_evolucio.pdf` |

`comparacio.py` i `problema.py` triguen una mica més que la resta en executar.

## Notes

- Totes les variables són adimensionals, excepte a `problema.py`, on els resultats es
  passen a °C i segons amb les funcions de `parametres.py`.
- No es fa servir cap funció de `numpy.linalg` ni de SciPy: els sistemes lineals es
  resolen amb el nostre Gauss–Seidel.